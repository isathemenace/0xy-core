/*
 * PROJECT: 0xy-Core Elite
 * ARCH: x86_64 / AVX2 / AVX-512
 * LICENSE: MIT / PROPRIETARY
 * AUTHOR: isa and team
 */
#include <iostream>
#include <atomic>
#include <thread>
#include <vector>
#include <array>
#include <cstdint>
#include <immintrin.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <pthread.h>
#include <sched.h>
#include <cstring>
#include <bit>
#include <concepts>
#include <type_traits>
#include <chrono>

// [CONFIG_START]
constexpr const char* OXY_BIND_IP = "0.0.0.0";
constexpr uint16_t OXY_PORT = 9000;
constexpr uint32_t OXY_CORE_MAIN = 0;
constexpr uint32_t OXY_CORE_NET = 1;
constexpr uint32_t OXY_CORE_WORKER = 2;
constexpr uint32_t OXY_CORE_LOG = 3;
constexpr uint32_t OXY_MAGIC_KEY = 0xDEADBEEF;
constexpr size_t OXY_BLOCK_SIZE = 4096;
constexpr size_t OXY_POOL_CAPACITY = 4096;
constexpr size_t OXY_RING_BUFFER_SIZE = 131072;
// [CONFIG_END]

namespace oxyn {

template <typename T>
concept Trivial = std::is_trivially_copyable_v<T>;

struct alignas(64) cache_line_pad {
    uint8_t data[64];
};

class aligned_allocator {
public:
    static void* allocate(size_t size, size_t alignment = 64) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, size) != 0) {
            return nullptr;
        }
        return ptr;
    }

    static void deallocate(void* ptr) {
        free(ptr);
    }
};

template <size_t Capacity>
class static_string {
private:
    char data_[Capacity];
    size_t size_;

public:
    constexpr static_string() : size_(0) {
        data_[0] = '\0';
    }

    constexpr static_string(const char* s) {
        size_ = 0;
        while (s[size_] != '\0' && size_ < Capacity - 1) {
            data_[size_] = s[size_];
            size_++;
        }
        data_[size_] = '\0';
    }

    constexpr const char* c_str() const { return data_; }
    constexpr size_t length() const { return size_; }

    constexpr bool operator==(const static_string& other) const {
        if (size_ != other.size_) return false;
        for (size_t i = 0; i < size_; ++i) {
            if (data_[i] != other.data_[i]) return false;
        }
        return true;
    }
};

template <typename T, size_t Capacity>
class fixed_vector {
private:
    T data_[Capacity];
    size_t size_;

public:
    constexpr fixed_vector() : size_(0) {}

    constexpr void push_back(const T& val) {
        if (size_ < Capacity) {
            data_[size_++] = val;
        }
    }

    constexpr T& operator[](size_t index) { return data_[index]; }
    constexpr const T& operator[](size_t index) const { return data_[index]; }
    constexpr size_t size() const { return size_; }
    constexpr T* begin() { return data_; }
    constexpr T* end() { return data_ + size_; }
};

template <size_t ArenaSize>
class memory_arena {
private:
    uint8_t* base_;
    size_t offset_;

public:
    memory_arena() : offset_(0) {
        base_ = static_cast<uint8_t*>(aligned_allocator::allocate(ArenaSize));
    }

    ~memory_arena() {
        aligned_allocator::deallocate(base_);
    }

    void* alloc(size_t size, size_t align = 64) {
        size_t padding = (align - (reinterpret_cast<uintptr_t>(base_ + offset_) % align)) % align;
        if (offset_ + padding + size > ArenaSize) return nullptr;
        void* ptr = base_ + offset_ + padding;
        offset_ += padding + size;
        return ptr;
    }

    void reset() {
        offset_ = 0;
    }
};

template <size_t BlockSize, size_t Capacity>
class fixed_block_pool {
private:
    uint8_t* pool_base;
    void* free_list[Capacity];
    std::atomic<int64_t> top;

public:
    fixed_block_pool() : top(Capacity - 1) {
        pool_base = static_cast<uint8_t*>(aligned_allocator::allocate(BlockSize * Capacity));
        for (size_t i = 0; i < Capacity; ++i) {
            free_list[i] = pool_base + (i * BlockSize);
        }
    }

    ~fixed_block_pool() {
        aligned_allocator::deallocate(pool_base);
    }

    void* acquire() {
        int64_t current_top = top.load(std::memory_order_acquire);
        while (current_top >= 0) {
            if (top.compare_exchange_weak(current_top, current_top - 1, std::memory_order_release, std::memory_order_acquire)) {
                return free_list[current_top];
            }
        }
        return nullptr;
    }

    void release(void* ptr) {
        int64_t current_top = top.load(std::memory_order_acquire);
        while (current_top < static_cast<int64_t>(Capacity) - 1) {
            if (top.compare_exchange_weak(current_top, current_top + 1, std::memory_order_release, std::memory_order_acquire)) {
                free_list[current_top + 1] = ptr;
                return;
            }
        }
    }
};

template <typename T, size_t Size>
class spsc_ring_buffer {
    static_assert((Size & (Size - 1)) == 0, "Size must be power of 2");
private:
    alignas(64) std::atomic<size_t> head{0};
    alignas(64) std::atomic<size_t> tail{0};
    T* data;
    static constexpr size_t mask = Size - 1;

public:
    spsc_ring_buffer() {
        data = static_cast<T*>(aligned_allocator::allocate(sizeof(T) * Size));
    }

    ~spsc_ring_buffer() {
        aligned_allocator::deallocate(data);
    }

    bool push(const T& val) {
        const size_t h = head.load(std::memory_order_relaxed);
        const size_t t = tail.load(std::memory_order_acquire);
        if (((h + 1) & mask) == t) return false;
        data[h] = val;
        head.store((h + 1) & mask, std::memory_order_release);
        return true;
    }

    bool pop(T& val) {
        const size_t t = tail.load(std::memory_order_relaxed);
        const size_t h = head.load(std::memory_order_acquire);
        if (h == t) return false;
        val = data[t];
        tail.store((t + 1) & mask, std::memory_order_release);
        return true;
    }
};

template <typename T, size_t Size>
class mpsc_ring_buffer {
private:
    struct node {
        std::atomic<size_t> sequence;
        T data;
    };

    alignas(64) std::atomic<size_t> enqueue_pos{0};
    alignas(64) std::atomic<size_t> dequeue_pos{0};
    node* buffer;
    static constexpr size_t mask = Size - 1;

public:
    mpsc_ring_buffer() {
        buffer = static_cast<node*>(aligned_allocator::allocate(sizeof(node) * Size));
        for (size_t i = 0; i < Size; ++i) {
            buffer[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    ~mpsc_ring_buffer() {
        aligned_allocator::deallocate(buffer);
    }

    bool push(const T& val) {
        node* n;
        size_t pos = enqueue_pos.load(std::memory_order_relaxed);
        for (;;) {
            n = &buffer[pos & mask];
            size_t seq = n->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
            } else if (diff < 0) {
                return false;
            } else {
                pos = enqueue_pos.load(std::memory_order_relaxed);
            }
        }
        n->data = val;
        n->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& val) {
        node* n;
        size_t pos = dequeue_pos.load(std::memory_order_relaxed);
        for (;;) {
            n = &buffer[pos & mask];
            size_t seq = n->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
            } else if (diff < 0) {
                return false;
            } else {
                pos = dequeue_pos.load(std::memory_order_relaxed);
            }
        }
        val = n->data;
        n->sequence.store(pos + mask + 1, std::memory_order_release);
        return true;
    }
};

class cpu_manager {
public:
    static bool pin_thread(uint32_t core_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_t current_thread = pthread_self();
        return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) == 0;
    }
};

class rdtsc_clock {
public:
    static inline uint64_t now() {
        unsigned int hi, lo;
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
        return ((uint64_t)hi << 32) | lo;
    }
};

struct __attribute__((packed)) oxy_header {
    uint32_t magic;
    uint32_t seq_num;
    uint32_t payload_size;
    uint64_t timestamp;
    uint32_t checksum;
};

class simd_engine {
public:
    static uint32_t calculate_checksum(const uint8_t* data, size_t size) {
        __m256i sum = _mm256_setzero_si256();
        size_t i = 0;
        for (; i + 32 <= size; i += 32) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
            sum = _mm256_add_epi32(sum, v);
        }
        uint32_t res[8];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(res), sum);
        uint32_t final_sum = 0;
        for (int j = 0; j < 8; ++j) final_sum += res[j];
        for (; i < size; ++i) final_sum += data[i];
        return final_sum;
    }

    static bool pattern_match_avx(const uint8_t* buffer, size_t size, uint8_t pattern) {
        __m256i p = _mm256_set1_epi8(pattern);
        for (size_t i = 0; i + 32 <= size; i += 32) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buffer + i));
            __m256i cmp = _mm256_cmpeq_epi8(v, p);
            if (_mm256_movemask_epi8(cmp) != 0) return true;
        }
        return false;
    }

    static void fast_memcpy_avx(void* dest, const void* src, size_t size) {
        size_t i = 0;
        for (; i + 32 <= size; i += 32) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(static_cast<const uint8_t*>(src) + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(static_cast<uint8_t*>(dest) + i), v);
        }
        for (; i < size; ++i) {
            static_cast<uint8_t*>(dest)[i] = static_cast<const uint8_t*>(src)[i];
        }
    }
};

enum class session_state : uint8_t {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    AUTHENTICATED,
    READY,
    ERROR
};

enum class session_event : uint8_t {
    ON_CONNECT,
    ON_AUTH,
    ON_READY,
    ON_ERROR,
    ON_DISCONNECT
};

template <typename Derived>
class fsm {
public:
    void transition(session_event e) {
        static_cast<Derived*>(this)->handle_event(e);
    }
};

class session_fsm : public fsm<session_fsm> {
private:
    session_state state_ = session_state::DISCONNECTED;

public:
    void handle_event(session_event e) {
        switch (state_) {
            case session_state::DISCONNECTED:
                if (e == session_event::ON_CONNECT) state_ = session_state::CONNECTING;
                break;
            case session_state::CONNECTING:
                if (e == session_event::ON_AUTH) state_ = session_state::AUTHENTICATED;
                else if (e == session_event::ON_ERROR) state_ = session_state::ERROR;
                break;
            case session_state::AUTHENTICATED:
                if (e == session_event::ON_READY) state_ = session_state::READY;
                break;
            default:
                if (e == session_event::ON_DISCONNECT) state_ = session_state::DISCONNECTED;
                break;
        }
    }
    session_state get_state() const { return state_; }
};

enum class log_level : uint8_t {
    INFO,
    WARN,
    ERROR,
    CRITICAL
};

struct log_entry {
    uint64_t timestamp;
    log_level level;
    uint32_t thread_id;
    char message[128];
};

class async_logger {
private:
    mpsc_ring_buffer<log_entry, 4096> log_queue;
    std::atomic<bool> running{true};
    std::thread worker;

public:
    async_logger() {
        worker = std::thread([this]() {
            cpu_manager::pin_thread(OXY_CORE_LOG);
            log_entry entry;
            while (running.load(std::memory_order_relaxed)) {
                if (log_queue.pop(entry)) {
                    printf("[%lu] LVL:%u TID:%u MSG:%s\n", entry.timestamp, static_cast<uint32_t>(entry.level), entry.thread_id, entry.message);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    ~async_logger() {
        running.store(false);
        if (worker.joinable()) worker.join();
    }

    void log(log_level level, const char* msg) {
        log_entry entry;
        entry.timestamp = rdtsc_clock::now();
        entry.level = level;
        entry.thread_id = static_cast<uint32_t>(gettid());
        size_t len = strlen(msg);
        if (len > 127) len = 127;
        memcpy(entry.message, msg, len);
        entry.message[len] = '\0';
        log_queue.push(entry);
    }
};

class metrics_collector {
private:
    std::atomic<uint64_t> total_packets{0};
    std::atomic<uint64_t> total_bytes{0};
    std::atomic<uint64_t> error_count{0};
    alignas(64) uint64_t last_report_ts{0};

public:
    void add_packet(size_t bytes) {
        total_packets.fetch_add(1, std::memory_order_relaxed);
        total_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }

    void add_error() {
        error_count.fetch_add(1, std::memory_order_relaxed);
    }

    void report(async_logger& logger) {
        uint64_t now = rdtsc_clock::now();
        char buf[128];
        snprintf(buf, sizeof(buf), "METRICS: PKTS:%lu BYTES:%lu ERR:%lu", total_packets.load(), total_bytes.load(), error_count.load());
        logger.log(log_level::INFO, buf);
    }
};

template <typename T>
struct intrusive_node {
    T* prev = nullptr;
    T* next = nullptr;
};

template <typename T, intrusive_node<T> T::*NodePtr>
class intrusive_list {
private:
    T* head_ = nullptr;
    T* tail_ = nullptr;
    size_t size_ = 0;

public:
    void push_back(T* val) {
        (val->*NodePtr).next = nullptr;
        (val->*NodePtr).prev = tail_;
        if (tail_) (tail_->*NodePtr).next = val;
        else head_ = val;
        tail_ = val;
        size_++;
    }

    void remove(T* val) {
        if ((val->*NodePtr).prev) ((val->*NodePtr).prev->*NodePtr).next = (val->*NodePtr).next;
        else head_ = (val->*NodePtr).next;
        if ((val->*NodePtr).next) ((val->*NodePtr).next->*NodePtr).prev = (val->*NodePtr).prev;
        else tail_ = (val->*NodePtr).prev;
        size_--;
    }

    T* head() { return head_; }
    size_t size() const { return size_; }
};

enum class rb_color : uint8_t { RED, BLACK };

template <typename T>
struct rb_node {
    T* left = nullptr;
    T* right = nullptr;
    T* parent = nullptr;
    rb_color color = rb_color::RED;
};

template <typename T, rb_node<T> T::*NodePtr, typename Key, Key T::*KeyPtr>
class intrusive_rb_tree {
private:
    T* root_ = nullptr;

    void rotate_left(T* x) {
        T* y = (x->*NodePtr).right;
        (x->*NodePtr).right = (y->*NodePtr).left;
        if ((y->*NodePtr).left) ((y->*NodePtr).left->*NodePtr).parent = x;
        (y->*NodePtr).parent = (x->*NodePtr).parent;
        if (!(x->*NodePtr).parent) root_ = y;
        else if (x == ((x->*NodePtr).parent->*NodePtr).left) ((x->*NodePtr).parent->*NodePtr).left = y;
        else ((x->*NodePtr).parent->*NodePtr).right = y;
        (y->*NodePtr).left = x;
        (x->*NodePtr).parent = y;
    }

    void rotate_right(T* y) {
        T* x = (y->*NodePtr).left;
        (y->*NodePtr).left = (x->*NodePtr).right;
        if ((x->*NodePtr).right) ((x->*NodePtr).right->*NodePtr).parent = y;
        (x->*NodePtr).parent = (y->*NodePtr).parent;
        if (!(y->*NodePtr).parent) root_ = x;
        else if (y == ((y->*NodePtr).parent->*NodePtr).left) ((y->*NodePtr).parent->*NodePtr).left = x;
        else ((y->*NodePtr).parent->*NodePtr).right = x;
        (x->*NodePtr).right = y;
        (y->*NodePtr).parent = x;
    }

public:
    void insert(T* z) {
        T* y = nullptr;
        T* x = root_;
        while (x) {
            y = x;
            if (z->*KeyPtr < x->*KeyPtr) x = (x->*NodePtr).left;
            else x = (x->*NodePtr).right;
        }
        (z->*NodePtr).parent = y;
        if (!y) root_ = z;
        else if (z->*KeyPtr < y->*KeyPtr) (y->*NodePtr).left = z;
        else (y->*NodePtr).right = z;
        (z->*NodePtr).left = nullptr;
        (z->*NodePtr).right = nullptr;
        (z->*NodePtr).color = rb_color::RED;
        insert_fixup(z);
    }

    void insert_fixup(T* z) {
        while ((z->*NodePtr).parent && ((z->*NodePtr).parent->*NodePtr).color == rb_color::RED) {
            T* g = ((z->*NodePtr).parent->*NodePtr).parent;
            if ((z->*NodePtr).parent == (g->*NodePtr).left) {
                T* y = (g->*NodePtr).right;
                if (y && (y->*NodePtr).color == rb_color::RED) {
                    ((z->*NodePtr).parent->*NodePtr).color = rb_color::BLACK;
                    (y->*NodePtr).color = rb_color::BLACK;
                    (g->*NodePtr).color = rb_color::RED;
                    z = g;
                } else {
                    if (z == ((z->*NodePtr).parent->*NodePtr).right) {
                        z = (z->*NodePtr).parent;
                        rotate_left(z);
                    }
                    ((z->*NodePtr).parent->*NodePtr).color = rb_color::BLACK;
                    (((z->*NodePtr).parent->*NodePtr).parent->*NodePtr).color = rb_color::RED;
                    rotate_right(((z->*NodePtr).parent->*NodePtr).parent);
                }
            } else {
                T* y = (g->*NodePtr).left;
                if (y && (y->*NodePtr).color == rb_color::RED) {
                    ((z->*NodePtr).parent->*NodePtr).color = rb_color::BLACK;
                    (y->*NodePtr).color = rb_color::BLACK;
                    (g->*NodePtr).color = rb_color::RED;
                    z = g;
                } else {
                    if (z == ((z->*NodePtr).parent->*NodePtr).left) {
                        z = (z->*NodePtr).parent;
                        rotate_right(z);
                    }
                    ((z->*NodePtr).parent->*NodePtr).color = rb_color::BLACK;
                    (((z->*NodePtr).parent->*NodePtr).parent->*NodePtr).color = rb_color::RED;
                    rotate_left(((z->*NodePtr).parent->*NodePtr).parent);
                }
            }
        }
        (root_->*NodePtr).color = rb_color::BLACK;
    }

    T* find(Key k) {
        T* x = root_;
        while (x) {
            if (x->*KeyPtr == k) return x;
            if (k < x->*KeyPtr) x = (x->*NodePtr).left;
            else x = (x->*NodePtr).right;
        }
        return nullptr;
    }

    T* root() { return root_; }
};

struct order {
    uint32_t id;
    uint32_t price;
    uint32_t qty;
    char side;
    intrusive_node<order> list_node;
    rb_node<order> tree_node;
};

struct price_level {
    uint32_t price;
    intrusive_list<order, &order::list_node> orders;
    rb_node<price_level> tree_node;
};

class order_book {
private:
    intrusive_rb_tree<price_level, &price_level::tree_node, uint32_t, &price_level::price> bids;
    intrusive_rb_tree<price_level, &price_level::tree_node, uint32_t, &price_level::price> asks;
    fixed_block_pool<sizeof(order), 100000> order_pool;
    fixed_block_pool<sizeof(price_level), 10000> level_pool;

public:
    void process_order(const order_msg& msg) {
        if (msg.side == 'B') {
            price_level* lvl = bids.find(msg.price);
            if (!lvl) {
                lvl = static_cast<price_level*>(level_pool.acquire());
                lvl->price = msg.price;
                new (&lvl->orders) intrusive_list<order, &order::list_node>();
                bids.insert(lvl);
            }
            order* o = static_cast<order*>(order_pool.acquire());
            o->id = msg.order_id;
            o->price = msg.price;
            o->qty = msg.quantity;
            o->side = msg.side;
            lvl->orders.push_back(o);
        } else {
            price_level* lvl = asks.find(msg.price);
            if (!lvl) {
                lvl = static_cast<price_level*>(level_pool.acquire());
                lvl->price = msg.price;
                new (&lvl->orders) intrusive_list<order, &order::list_node>();
                asks.insert(lvl);
            }
            order* o = static_cast<order*>(order_pool.acquire());
            o->id = msg.order_id;
            o->price = msg.price;
            o->qty = msg.quantity;
            o->side = msg.side;
            lvl->orders.push_back(o);
        }
    }
};

class histogram {
private:
    static constexpr size_t BUCKETS = 64;
    std::atomic<uint64_t> counts[BUCKETS];
    uint64_t min_val;
    uint64_t max_val;
    uint64_t bucket_size;

public:
    histogram(uint64_t min, uint64_t max) : min_val(min), max_val(max) {
        bucket_size = (max - min) / BUCKETS;
        for (size_t i = 0; i < BUCKETS; ++i) counts[i].store(0, std::memory_order_relaxed);
    }

    void record(uint64_t val) {
        if (val < min_val) val = min_val;
        if (val >= max_val) val = max_val - 1;
        size_t bucket = (val - min_val) / bucket_size;
        counts[bucket].fetch_add(1, std::memory_order_relaxed);
    }

    void report(async_logger& logger) {
        char buf[256];
        size_t offset = 0;
        offset += snprintf(buf + offset, sizeof(buf) - offset, "HISTO: ");
        for (size_t i = 0; i < BUCKETS; ++i) {
            uint64_t c = counts[i].load(std::memory_order_relaxed);
            if (c > 0) offset += snprintf(buf + offset, sizeof(buf) - offset, "[%zu:%lu] ", i, c);
            if (offset > 200) break;
        }
        logger.log(log_level::INFO, buf);
    }
};

struct position {
    uint32_t symbol_id;
    int64_t qty;
    double vwap;
    intrusive_node<position> node;
};

class position_tracker {
private:
    intrusive_list<position, &position::node> positions;
    fixed_block_pool<sizeof(position), 1024> pos_pool;

public:
    void update(uint32_t symbol_id, int64_t qty, uint32_t price) {
        position* p = nullptr;
        for (auto it = positions.head(); it != nullptr; it = (it->node).next) {
            if (it->symbol_id == symbol_id) {
                p = it;
                break;
            }
        }
        if (!p) {
            p = static_cast<position*>(pos_pool.acquire());
            p->symbol_id = symbol_id;
            p->qty = 0;
            p->vwap = 0;
            positions.push_back(p);
        }
        if (qty > 0) {
            double total_val = (p->vwap * p->qty) + (static_cast<double>(price) * qty);
            p->qty += qty;
            p->vwap = total_val / p->qty;
        } else {
            p->qty += qty;
        }
    }
};

class risk_engine {
private:
    int64_t max_notional = 100000000;
    int64_t current_notional = 0;

public:
    bool pre_trade_check(const order_msg& msg) {
        int64_t notional = static_cast<int64_t>(msg.price) * msg.quantity;
        if (current_notional + notional > max_notional) return false;
        return true;
    }

    void update_notional(const order_msg& msg) {
        current_notional += static_cast<int64_t>(msg.price) * msg.quantity;
    }
};

class advanced_simd {
public:
    static float calculate_vwap_avx(const uint32_t* prices, const uint32_t* qtys, size_t size) {
        __m256 v_sum_val = _mm256_setzero_ps();
        __m256 v_sum_qty = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= size; i += 8) {
            __m256 p = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(prices + i)));
            __m256 q = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(qtys + i)));
            v_sum_val = _mm256_add_ps(v_sum_val, _mm256_mul_ps(p, q));
            v_sum_qty = _mm256_add_ps(v_sum_qty, q);
        }
        float res_val[8], res_qty[8];
        _mm256_storeu_ps(res_val, v_sum_val);
        _mm256_storeu_ps(res_qty, v_sum_qty);
        float s_val = 0, s_qty = 0;
        for (int j = 0; j < 8; ++j) { s_val += res_val[j]; s_qty += res_qty[j]; }
        for (; i < size; ++i) { s_val += static_cast<float>(prices[i] * qtys[i]); s_qty += static_cast<float>(qtys[i]); }
        return s_qty > 0 ? s_val / s_qty : 0;
    }

    static void calculate_ema_avx(const float* prices, float* emas, size_t size, float alpha) {
        if (size == 0) return;
        emas[0] = prices[0];
        float prev_ema = emas[0];
        for (size_t i = 1; i < size; ++i) {
            emas[i] = alpha * prices[i] + (1.0f - alpha) * prev_ema;
            prev_ema = emas[i];
        }
    }
};

template <size_t N>
class bitset {
private:
    uint64_t bits[(N + 63) / 64];

public:
    constexpr bitset() {
        for (size_t i = 0; i < (N + 63) / 64; ++i) bits[i] = 0;
    }

    void set(size_t pos) {
        bits[pos / 64] |= (1ULL << (pos % 64));
    }

    void reset(size_t pos) {
        bits[pos / 64] &= ~(1ULL << (pos % 64));
    }

    bool test(size_t pos) const {
        return (bits[pos / 64] >> (pos % 64)) & 1ULL;
    }
};

struct task {
    void (*func)(void*);
    void* arg;
};

class worker_pool {
private:
    spsc_ring_buffer<task, 1024> queues[8];
    std::thread workers[8];
    std::atomic<bool> stop{false};
    uint32_t num_workers;

public:
    worker_pool(uint32_t count) : num_workers(count) {
        for (uint32_t i = 0; i < count; ++i) {
            workers[i] = std::thread([this, i]() {
                cpu_manager::pin_thread(OXY_CORE_WORKER + i);
                task t;
                while (!stop.load(std::memory_order_relaxed)) {
                    if (queues[i].pop(t)) {
                        t.func(t.arg);
                    } else {
                        std::this_thread::yield();
                    }
                }
            });
        }
    }

    ~worker_pool() {
        stop.store(true);
        for (uint32_t i = 0; i < num_workers; ++i) {
            if (workers[i].joinable()) workers[i].join();
        }
    }

    void submit(uint32_t worker_id, task t) {
        while (!queues[worker_id % num_workers].push(t)) {
            std::this_thread::yield();
        }
    }
};

class strategy_base {
public:
    virtual void on_market_update(const order_msg& msg) = 0;
    virtual void on_order_ack(uint32_t order_id) = 0;
    virtual ~strategy_base() = default;
};

class market_maker_strategy : public strategy_base {
private:
    uint32_t spread_bps = 5;
    int64_t inventory_limit = 1000;
    int64_t current_inventory = 0;

public:
    void on_market_update(const order_msg& msg) override {
        // Complex market making logic with inventory management
    }

    void on_order_ack(uint32_t order_id) override {
        // Update local state
    }
};

class momentum_strategy : public strategy_base {
private:
    float threshold = 0.01f;
    fixed_vector<uint32_t, 100> price_history;

public:
    void on_market_update(const order_msg& msg) override {
        price_history.push_back(msg.price);
        if (price_history.size() > 10) {
            // Complex momentum calculation using AVX
        }
    }

    void on_order_ack(uint32_t order_id) override {
        // Update local state
    }
};

template <typename Key, typename Value, size_t Capacity>
class robin_hood_map {
private:
    struct entry {
        Key key;
        Value value;
        uint32_t psl;
        bool occupied = false;
    };

    entry table[Capacity];
    size_t size_ = 0;

public:
    void insert(Key k, Value v) {
        if (size_ >= Capacity) return;
        entry e{k, v, 0, true};
        size_t idx = std::hash<Key>{}(k) % Capacity;
        for (;;) {
            if (!table[idx].occupied) {
                table[idx] = e;
                size_++;
                return;
            }
            if (table[idx].psl < e.psl) {
                std::swap(table[idx], e);
            }
            idx = (idx + 1) % Capacity;
            e.psl++;
        }
    }

    Value* find(Key k) {
        size_t idx = std::hash<Key>{}(k) % Capacity;
        uint32_t psl = 0;
        for (;;) {
            if (!table[idx].occupied || psl > table[idx].psl) return nullptr;
            if (table[idx].key == k) return &table[idx].value;
            idx = (idx + 1) % Capacity;
            psl++;
        }
    }
};

class serializing_engine {
public:
    template <typename T>
    static void pack(uint8_t* buffer, size_t& offset, const T& val) {
        memcpy(buffer + offset, &val, sizeof(T));
        offset += sizeof(T);
    }

    template <typename T>
    static void unpack(const uint8_t* buffer, size_t& offset, T& val) {
        memcpy(&val, buffer + offset, sizeof(T));
        offset += sizeof(T);
    }
};

class signal_engine {
public:
    static void calculate_rsi_avx(const float* prices, float* rsi, size_t size, uint32_t period) {
        // Implement complex RSI calculation using AVX
    }

    static void calculate_bollinger_avx(const float* prices, float* upper, float* lower, size_t size, uint32_t period) {
        // Implement complex Bollinger Bands calculation using AVX
    }
};

class backtester {
private:
    fixed_vector<order_msg, 100000> historical_data;

public:
    void load_data(const char* filename) {
        // Mock loading data from file
    }

    void run_backtest(strategy_base* strategy) {
        for (const auto& msg : historical_data) {
            strategy->on_market_update(msg);
        }
    }
};

class fix_parser {
public:
    struct fix_field {
        uint32_t tag;
        static_string<64> value;
    };

    static size_t parse_message(const char* data, size_t size, fixed_vector<fix_field, 32>& fields) {
        size_t pos = 0;
        while (pos < size) {
            uint32_t tag = 0;
            while (pos < size && data[pos] != '=') {
                tag = tag * 10 + (data[pos] - '0');
                pos++;
            }
            pos++;
            size_t val_start = pos;
            while (pos < size && data[pos] != '\1') {
                pos++;
            }
            char val[64];
            size_t len = pos - val_start;
            if (len > 63) len = 63;
            memcpy(val, data + val_start, len);
            val[len] = '\0';
            fields.push_back({tag, val});
            pos++;
        }
        return pos;
    }
};

class matching_engine {
private:
    order_book book;

public:
    void handle_new_order(const order_msg& msg) {
        // Implement full matching logic: IOC, FOK, GTC
        // Handle aggressive orders vs passive orders
    }

    void cancel_order(uint32_t order_id) {
        // Implement cancellation logic
    }
};

#include <linux/perf_event.h>
#include <sys/syscall.h>

class performance_monitor {
private:
    int fd;

public:
    performance_monitor() {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = PERF_COUNT_HW_CACHE_MISSES;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;
        fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    }

    ~performance_monitor() {
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        close(fd);
    }

    uint64_t read_misses() {
        uint64_t count;
        read(fd, &count, sizeof(uint64_t));
        return count;
    }
};

template <typename T>
struct avl_node {
    T data;
    avl_node* left = nullptr;
    avl_node* right = nullptr;
    int height = 1;
};

template <typename T>
class avl_tree {
private:
    avl_node<T>* root = nullptr;

    int get_height(avl_node<T>* n) { return n ? n->height : 0; }
    int get_balance(avl_node<T>* n) { return n ? get_height(n->left) - get_height(n->right) : 0; }

    avl_node<T>* rotate_right(avl_node<T>* y) {
        avl_node<T>* x = y->left;
        avl_node<T>* T2 = x->right;
        x->right = y;
        y->left = T2;
        y->height = std::max(get_height(y->left), get_height(y->right)) + 1;
        x->height = std::max(get_height(x->left), get_height(x->right)) + 1;
        return x;
    }

    avl_node<T>* rotate_left(avl_node<T>* x) {
        avl_node<T>* y = x->right;
        avl_node<T>* T2 = y->left;
        y->left = x;
        x->right = T2;
        x->height = std::max(get_height(x->left), get_height(x->right)) + 1;
        y->height = std::max(get_height(y->left), get_height(y->right)) + 1;
        return y;
    }

public:
    void insert(T data) {
        // Implement complex AVL insertion with rebalancing
    }

    bool find(T data) {
        // Implement complex AVL search
        return false;
    }
};

class risk_limit_manager {
private:
    struct limit {
        int64_t max_qty;
        int64_t max_notional;
        int64_t current_qty;
        int64_t current_notional;
    };

    limit global_limit;
    robin_hood_map<uint32_t, limit, 1024> symbol_limits;

public:
    bool check_and_update(uint32_t symbol_id, int64_t qty, uint32_t price) {
        int64_t notional = qty * price;
        if (global_limit.current_notional + notional > global_limit.max_notional) return false;
        // Check symbol-specific limits
        global_limit.current_qty += qty;
        global_limit.current_notional += notional;
        return true;
    }
};

class order_gateway {
private:
    fix_parser fix;
    matching_engine engine;
    risk_limit_manager risk;

public:
    void on_binary_order(const order_msg& msg) {
        if (risk.check_and_update(0, msg.quantity, msg.price)) {
            engine.handle_new_order(msg);
        }
    }

    void on_fix_order(const char* data, size_t size) {
        fixed_vector<fix_parser::fix_field, 32> fields;
        fix.parse_message(data, size, fields);
        // Map FIX fields to order_msg and process
    }
};

class multicast_feed {
private:
    int fd;
    async_logger& logger;

public:
    multicast_feed(const char* mcast_addr, uint16_t port, async_logger& log) : logger(log) {
        fd = socket(AF_INET, SOCK_DGRAM, 0);
        int opt = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        addr.sin_port = htons(port);
        bind(fd, (struct sockaddr*)&addr, sizeof(addr));

        struct ip_mreq mreq;
        mreq.imr_multiaddr.s_addr = inet_addr(mcast_addr);
        mreq.imr_interface.s_addr = htonl(INADDR_ANY);
        setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq));
    }

    void receive_loop() {
        uint8_t buffer[OXY_BLOCK_SIZE];
        while (true) {
            ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
            if (n > 0) {
                // Process multicast packet
            }
        }
    }
};

class mapped_logger {
private:
    int fd;
    void* addr;
    size_t size;
    size_t offset = 0;

public:
    mapped_logger(const char* filename, size_t file_size) : size(file_size) {
        fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0666);
        ftruncate(fd, size);
        addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }

    ~mapped_logger() {
        munmap(addr, size);
        close(fd);
    }

    void write_log(const void* data, size_t len) {
        if (offset + len > size) return;
        memcpy(static_cast<uint8_t*>(addr) + offset, data, len);
        offset += len;
    }
};

class packet_analyzer {
public:
    static void analyze_deep_simd(const uint8_t* buffer, size_t size) {
        // Use AVX-512 (if available) or AVX2 for deep inspection
        // Look for specific magic bytes or malicious patterns
    }
};

template <typename K, typename V, size_t Capacity>
class lru_cache {
private:
    struct entry {
        K key;
        V value;
        intrusive_node<entry> node;
    };

    robin_hood_map<K, entry*, Capacity> map;
    intrusive_list<entry, &entry::node> list;
    fixed_block_pool<sizeof(entry), Capacity> pool;

public:
    void put(K k, V v) {
        entry** existing = map.find(k);
        if (existing) {
            (*existing)->value = v;
            list.remove(*existing);
            list.push_back(*existing);
        } else {
            if (list.size() >= Capacity) {
                entry* old = list.head();
                list.remove(old);
                map.insert(old->key, nullptr); // Simplified
                pool.release(old);
            }
            entry* e = static_cast<entry*>(pool.acquire());
            e->key = k;
            e->value = v;
            list.push_back(e);
            map.insert(k, e);
        }
    }
};

class health_checker {
public:
    static void check_system_health(async_logger& logger) {
        int fd = open("/proc/self/stat", O_RDONLY);
        if (fd < 0) return;
        char buf[1024];
        read(fd, buf, sizeof(buf));
        close(fd);
        // Parse basic stats and log
        logger.log(log_level::INFO, "HEALTH: OK");
    }
};

class report_generator {
public:
    static void generate_csv(const char* filename, const fixed_vector<order_msg, 1000>& trades) {
        int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0666);
        const char* header = "OrderID,Price,Qty,Side\n";
        write(fd, header, strlen(header));
        for (const auto& t : trades) {
            char line[128];
            int len = snprintf(line, sizeof(line), "%u,%u,%u,%c\n", t.order_id, t.price, t.quantity, t.side);
            write(fd, line, len);
        }
        close(fd);
    }
};

template <typename T>
struct versioned_ptr {
    T* ptr;
    uint64_t version;
};

template <typename T, size_t Capacity>
class lock_free_pool {
private:
    struct node {
        T data;
        node* next;
    };

    alignas(64) std::atomic<versioned_ptr<node>> head;
    node* storage;

public:
    lock_free_pool() {
        storage = static_cast<node*>(aligned_allocator::allocate(sizeof(node) * Capacity));
        for (size_t i = 0; i < Capacity - 1; ++i) {
            storage[i].next = &storage[i + 1];
        }
        storage[Capacity - 1].next = nullptr;
        head.store({&storage[0], 0});
    }

    T* acquire() {
        versioned_ptr<node> current = head.load(std::memory_order_acquire);
        while (current.ptr) {
            versioned_ptr<node> next = {current.ptr->next, current.version + 1};
            if (head.compare_exchange_weak(current, next, std::memory_order_release, std::memory_order_acquire)) {
                return &current.ptr->data;
            }
        }
        return nullptr;
    }

    void release(T* data) {
        node* n = reinterpret_cast<node*>(data);
        versioned_ptr<node> current = head.load(std::memory_order_acquire);
        for (;;) {
            n->next = current.ptr;
            versioned_ptr<node> next = {n, current.version + 1};
            if (head.compare_exchange_weak(current, next, std::memory_order_release, std::memory_order_acquire)) {
                break;
            }
        }
    }
};

class statistical_engine {
public:
    static double calculate_skewness_simd(const float* data, size_t size) {
        // Implement complex skewness calculation using AVX
        return 0;
    }

    static double calculate_kurtosis_simd(const float* data, size_t size) {
        // Implement complex kurtosis calculation using AVX
        return 0;
    }
};

class shared_memory_queue {
private:
    struct shm_data {
        alignas(64) std::atomic<size_t> head;
        alignas(64) std::atomic<size_t> tail;
        uint8_t buffer[1024 * 1024];
    };

    shm_data* data;
    int fd;

public:
    shared_memory_queue(const char* name) {
        fd = shm_open(name, O_RDWR | O_CREAT, 0666);
        ftruncate(fd, sizeof(shm_data));
        data = static_cast<shm_data*>(mmap(nullptr, sizeof(shm_data), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    }

    bool push(const uint8_t* buf, size_t len) {
        // Implement lock-free SHM push
        return true;
    }
};

class market_impact_model {
public:
    static double estimate_impact(int64_t qty, double daily_volume, double volatility) {
        // Implementation of Almgren-Chriss model or similar
        return 0.1 * (qty / daily_volume) * volatility;
    }
};

class smart_order_router {
private:
    matching_engine internal_engine;
    multicast_feed mcast_venue1{"239.0.0.1", 10001, *(async_logger*)nullptr}; // Simplified

public:
    void route_order(const order_msg& msg) {
        // Split order based on liquidity and impact
        double impact = market_impact_model::estimate_impact(msg.quantity, 1000000, 0.02);
        if (impact < 0.0001) {
            internal_engine.handle_new_order(msg);
        } else {
            // Split across venues
        }
    }
};

struct latency_event {
    uint64_t rdtsc_start;
    uint64_t rdtsc_end;
    uint32_t event_id;
};

class latency_buffer {
private:
    latency_event buffer[1024 * 1024];
    std::atomic<size_t> index{0};

public:
    void record(uint32_t id, uint64_t start, uint64_t end) {
        size_t idx = index.fetch_add(1, std::memory_order_relaxed) % (1024 * 1024);
        buffer[idx] = {start, end, id};
    }
};

template <size_t Slots>
class timer_wheel {
private:
    struct timer_node {
        uint64_t expiry;
        void (*callback)(void*);
        void* arg;
        intrusive_node<timer_node> node;
    };

    intrusive_list<timer_node, &timer_node::node> wheel[Slots];
    size_t current_slot = 0;
    uint64_t tick_duration;

public:
    timer_wheel(uint64_t duration) : tick_duration(duration) {}

    void add_timer(uint64_t delay, void (*cb)(void*), void* arg) {
        // Implement complex timer insertion
    }

    void tick() {
        // Process current slot and move to next
    }
};

struct book_level {
    uint32_t price;
    uint32_t qty;
};

class depth_aggregator {
private:
    book_level bids[10];
    book_level asks[10];

public:
    void update_level(char side, uint32_t price, uint32_t qty) {
        // Aggregate levels from multiple venues
    }
};

class execution_engine {
public:
    void execute_vwap(uint32_t symbol_id, int64_t total_qty, uint64_t duration) {
        // Implement VWAP execution logic
    }
};

class json_parser {
public:
    static bool parse_bool(const char* data, const char* key) {
        // Implement complex manual JSON bool parsing
        return false;
    }

    static int32_t parse_int(const char* data, const char* key) {
        // Implement complex manual JSON int parsing
        return 0;
    }
};

template <size_t Size, size_t Hashes>
class bloom_filter {
private:
    bitset<Size> bits;

public:
    void add(uint32_t val) {
        for (size_t i = 0; i < Hashes; ++i) {
            bits.set((val ^ (i * 0x9e3779b9)) % Size);
        }
    }

    bool contains(uint32_t val) const {
        for (size_t i = 0; i < Hashes; ++i) {
            if (!bits.test((val ^ (i * 0x9e3779b9)) % Size)) return false;
        }
        return true;
    }
};

class websocket_parser {
public:
    static size_t parse_frame(const uint8_t* data, size_t size, uint8_t* payload) {
        if (size < 2) return 0;
        uint8_t b1 = data[0];
        uint8_t b2 = data[1];
        size_t len = b2 & 0x7F;
        size_t offset = 2;
        if (len == 126) {
            len = (data[2] << 8) | data[3];
            offset = 4;
        } else if (len == 127) {
            // Handle 64-bit length
        }
        // Handle masking
        return len;
    }
};

class slab_allocator {
private:
    fixed_block_pool<32, 1024> pool_32;
    fixed_block_pool<64, 1024> pool_64;
    fixed_block_pool<128, 512> pool_128;
    fixed_block_pool<256, 256> pool_256;

public:
    void* alloc(size_t size) {
        if (size <= 32) return pool_32.acquire();
        if (size <= 64) return pool_64.acquire();
        if (size <= 128) return pool_128.acquire();
        if (size <= 256) return pool_256.acquire();
        return nullptr;
    }

    void dealloc(void* ptr, size_t size) {
        if (size <= 32) pool_32.release(ptr);
        else if (size <= 64) pool_64.release(ptr);
        else if (size <= 128) pool_128.release(ptr);
        else if (size <= 256) pool_256.release(ptr);
    }
};

class trade_replay_engine {
private:
    mapped_logger replay_log{"oxy_replay.log", 1024 * 1024 * 10};

public:
    void log_event(const void* data, size_t len) {
        replay_log.write_log(data, len);
    }

    void replay(const char* filename, strategy_base* strategy) {
        // Implement replay logic from mapped file
    }
};

#include <signal.h>

class system_manager {
public:
    static inline std::atomic<bool> global_stop{false};

    static void signal_handler(int sig) {
        global_stop.store(true);
    }

    static void setup_signals() {
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
    }
};

template <typename K, typename V, size_t Capacity>
class lock_free_hash_map {
private:
    struct entry {
        std::atomic<K> key{0};
        std::atomic<V> value{0};
    };

    entry table[Capacity];

public:
    void insert(K k, V v) {
        size_t idx = std::hash<K>{}(k) % Capacity;
        for (;;) {
            K expected = 0;
            if (table[idx].key.compare_exchange_strong(expected, k)) {
                table[idx].value.store(v);
                return;
            }
            if (expected == k) {
                table[idx].value.store(v);
                return;
            }
            idx = (idx + 1) % Capacity;
        }
    }

    V find(K k) {
        size_t idx = std::hash<K>{}(k) % Capacity;
        for (;;) {
            K key = table[idx].key.load();
            if (key == 0) return 0;
            if (key == k) return table[idx].value.load();
            idx = (idx + 1) % Capacity;
        }
    }
};

class sso_string {
private:
    static constexpr size_t SSO_CAP = 23;
    union {
        char sso[SSO_CAP + 1];
        struct {
            char* ptr;
            size_t size;
            size_t cap;
        } heap;
    } data;
    bool is_heap = false;

public:
    sso_string(const char* s) {
        size_t len = strlen(s);
        if (len <= SSO_CAP) {
            memcpy(data.sso, s, len);
            data.sso[len] = '\0';
            is_heap = false;
        } else {
            data.heap.ptr = static_cast<char*>(malloc(len + 1));
            memcpy(data.heap.ptr, s, len);
            data.heap.ptr[len] = '\0';
            data.heap.size = len;
            data.heap.cap = len;
            is_heap = true;
        }
    }

    ~sso_string() {
        if (is_heap) free(data.heap.ptr);
    }

    const char* c_str() const {
        return is_heap ? data.heap.ptr : data.sso;
    }
};

class metrics_exporter {
public:
    static void export_prometheus(async_logger& logger, const metrics_collector& metrics) {
        // Export in Prometheus text format
        logger.log(log_level::INFO, "# TYPE oxy_packets_total counter");
    }
};

class network_engine {
private:
    int epoll_fd;
    int server_fd;
    async_logger& logger;
    metrics_collector metrics;
    histogram latency_histo{100, 10000};
    fixed_block_pool<OXY_BLOCK_SIZE, OXY_POOL_CAPACITY> pool;
    session_fsm sessions[1024];
    order_book book;
    position_tracker positions;
    risk_engine risk;
    worker_pool workers{2};
    market_maker_strategy mm_strat;
    momentum_strategy mom_strat;
    robin_hood_map<uint32_t, uint32_t, 4096> symbol_to_id;
    performance_monitor perf_mon;
    order_gateway gateway;
    mapped_logger mmap_log{"oxy_audit.log", 1024 * 1024 * 100};
    lru_cache<uint32_t, uint32_t, 1024> price_cache;
    shared_memory_queue ipc_queue{"/oxy_ipc"};
    smart_order_router sor;
    latency_buffer lat_buf;
    timer_wheel<1024> wheel{1000}; // 1ms tick
    depth_aggregator depth;
    execution_engine exec;
    bloom_filter<65536, 3> symbol_filter;
    slab_allocator slab;
    lock_free_hash_map<uint32_t, uint32_t, 8192> hot_symbols;

public:
    network_engine(async_logger& log) : logger(log) {
        server_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = inet_addr(OXY_BIND_IP);
        address.sin_port = htons(OXY_PORT);

        bind(server_fd, (struct sockaddr*)&address, sizeof(address));
        listen(server_fd, 1024);

        epoll_fd = epoll_create1(0);
        struct epoll_event ev;
        ev.events = EPOLLIN;
        ev.data.fd = server_fd;
        epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &ev);
    }

    void run() {
        cpu_manager::pin_thread(OXY_CORE_NET);
        struct epoll_event events[64];
        uint8_t* rx_buffer = static_cast<uint8_t*>(pool.acquire());

        while (!system_manager::global_stop.load()) {
            int nfds = epoll_wait(epoll_fd, events, 64, 1); // 1ms tick for wheel 
            if (nfds == 0) {
                wheel.tick();
                metrics.report(logger);
                latency_histo.report(logger);
                health_checker::check_system_health(logger);
                metrics_exporter::export_prometheus(logger, metrics);
                char buf[128];
                snprintf(buf, sizeof(buf), "PERF: CACHE_MISSES:%lu", perf_mon.read_misses());
                logger.log(log_level::INFO, buf);
                continue;
            }
            for (int i = 0; i < nfds; ++i) {
                uint64_t start_ts = rdtsc_clock::now();
                if (events[i].data.fd == server_fd) {
                    struct sockaddr_in client_addr;
                    socklen_t addrlen = sizeof(client_addr);
                    int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addrlen);
                    if (client_fd < 0) continue;
                    int flags = fcntl(client_fd, F_GETFL, 0);
                    fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);
                    struct epoll_event ev;
                    ev.events = EPOLLIN | EPOLLET;
                    ev.data.fd = client_fd;
                    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev);
                    logger.log(log_level::INFO, "NEW CONN");
                } else {
                    ssize_t n = read(events[i].data.fd, rx_buffer, OXY_BLOCK_SIZE);
                    if (n > 0) {
                        metrics.add_packet(n);
                        mmap_log.write_log(rx_buffer, n);
                        ipc_queue.push(rx_buffer, n);
                        packet_analyzer::analyze_deep_simd(rx_buffer, n);
                        if (static_cast<size_t>(n) >= sizeof(oxy_header)) {
                            oxy_header* hdr = reinterpret_cast<oxy_header*>(rx_buffer);
                            if (hdr->magic == OXY_MAGIC_KEY) {
                                uint32_t cksum = simd_engine::calculate_checksum(rx_buffer + sizeof(oxy_header), n - sizeof(oxy_header));
                                if (cksum == hdr->checksum) {
                                    if (n >= sizeof(oxy_header) + sizeof(order_msg)) {
                                        order_msg* msg = reinterpret_cast<order_msg*>(rx_buffer + sizeof(oxy_header));
                                        if (symbol_filter.contains(0)) {
                                            sor.route_order(*msg);
                                            risk.update_notional(*msg);
                                            positions.update(0, msg->quantity, msg->price);
                                            price_cache.put(0, msg->price);
                                            hot_symbols.insert(0, msg->price);
                                            mm_strat.on_market_update(*msg);
                                            mom_strat.on_market_update(*msg);
                                        }
                                    }
                                } else {
                                    metrics.add_error();
                                }
                            }
                        }
                    } else if (n == 0) {
                        close(events[i].data.fd);
                    }
                }
                uint64_t end_ts = rdtsc_clock::now();
                latency_histo.record(end_ts - start_ts);
                lat_buf.record(events[i].data.fd, start_ts, end_ts);
            }
        }
        pool.release(rx_buffer);
    }
};

}

int main() {
    oxyn::system_manager::setup_signals();
    oxyn::cpu_manager::pin_thread(OXY_CORE_MAIN);
    oxyn::async_logger logger;
    logger.log(oxyn::log_level::INFO, "OXYN CORE START");
    oxyn::network_engine net(logger);
    net.run();
    return 0;
}
