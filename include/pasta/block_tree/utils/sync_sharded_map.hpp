#pragma once

#include <barrier>
#include <cassert>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <omp.h>
#include <pasta/block_tree/utils/concepts.hpp>
#include <pasta/block_tree/utils/mpsc_queue/jiffy.hpp>
#include <span>
#include <syncstream>
#include <unordered_map>
#include <vector>

namespace pasta {

enum Whereabouts { NOWHERE, IN_MAP, IN_QUEUE };

///
/// @brief An update function which on update just overwrites the value.
///
/// @tparam K The key type saved in the hash map.
/// @tparam V The value type saved in the hash map.
template <typename K, typename V>
struct [[maybe_unused]] Overwrite {
  using InputValue [[maybe_unused]] = V;

  inline static void update(const K&, V& value, V&& input_value) {
    value = input_value;
  }

  inline static V init(const K&, V&& input_value) {
    return input_value;
  }
};

///
/// @brief An update function which upon update does nothing besides
/// inserting the value if it doesn't exist.
///
/// @tparam K The key type saved in the hash map.
/// @tparam V The value type saved in the hash map.
///
template <typename K, typename V>
struct [[maybe_unused]] Keep {
  using InputValue [[maybe_unused]] = V;
  inline static void update(const K&, V&, V&&) {}

  inline static V init(const K&, V&& input_value) {
    return input_value;
  }
};

/// @brief A hash map that must be used by multiple threads, each thread having
///     only having write access to a certain segment of the input space.
/// @tparam K The type of the keys in the hash map.
/// @tparam V The type of the values in the hash map.
/// @tparam SeqHashMapType The type of the hash map used internally.
///     This should be compatible with std::unordered_map.
/// @tparam UpdateFn The update function deciding how to insert or update values
///     in the map.
template <std::copy_constructible K,
          std::move_constructible V,
          template <typename, typename> typename SeqHashMapType =
              std::unordered_map,
          UpdateFunction<K, V> UpdateFn = Overwrite<K, V>>
  requires std::movable<typename UpdateFn::InputValue>
class SyncShardedMap {
  /// The sequential backing hash map type
  using SeqHashMap = SeqHashMapType<K, V>;
  /// The sequential hash map's hasher
  using Hasher = SeqHashMap::hasher;
  /// The type used for updates
  using InputValue = UpdateFn::InputValue;

  /// The actual pair of key and value stored in the map
  using StoredValue = std::pair<K, InputValue>;

  using Queue = std::span<StoredValue>;

  /// The memory order namespace from the standard library
  using mem = std::memory_order;

  /// @brief The number of threads operating on this map.
  const size_t thread_count_;
  /// @brief Contains a hash map for each thread
  std::vector<SeqHashMap> map_;
  /// @brief Contains a task queue for each thread, holding insert
  ///   operations for each thread.
  std::vector<Queue> task_queue_;
  std::vector<Queue> task_queue_swap_;
  /// @brief Contains the number of tasks in each thread's queue.
  std::span<std::atomic_size_t> task_count_;
  /// @brief Contains the number of threads currently handling their queues.
  ///   This is used 1. signal to other threads that they should handle their
  ///   queue, and 2. to keep track of whether all threads have handled their
  ///   queues.
  std::atomic_size_t threads_handling_queue_;

  constexpr static std::invocable auto FN = []() noexcept {
  };

  std::barrier<decltype(FN)> barrier_;

  std::mutex mtx_;

  /// https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html
  inline uint64_t mix_select(uint64_t key) {
    key ^= (key >> 31);
    key *= 0x7fb5d329728ea185;
    key ^= (key >> 27);
    key *= 0x81dadef4bc2dd44d;
    key ^= (key >> 33);
    return key % thread_count_;
  }

public:
  std::atomic_size_t num_updates_;
  std::atomic_size_t num_inserts_;
  //
  /// @brief Creates a new sharded map.
  ///
  /// @param fill_threshold The fill percentage (between 0 and 1) above which
  /// a thread is signaled to handle its own tasks.
  /// @param thread_count The exact number of threads working on this map.
  /// @param queue_capacity The maximum amount of tasks allowed in each queue.
  ///
  SyncShardedMap(size_t thread_count, size_t queue_capacity)
      : thread_count_(thread_count),
        map_(),
        task_queue_(),
        task_count_(),
        threads_handling_queue_(0),
        barrier_(thread_count, FN),
        num_updates_(0),
        num_inserts_(0),
        mtx_() {
    map_.reserve(thread_count);
    task_queue_.reserve(thread_count);
    task_queue_swap_.reserve(thread_count);
    task_count_ =
        std::span<std::atomic_size_t>(new std::atomic_size_t[thread_count],
                                      thread_count);
    for (size_t i = 0; i < thread_count; i++) {
      map_.emplace_back();
      task_queue_.emplace_back(new StoredValue[queue_capacity], queue_capacity);
      task_queue_swap_.emplace_back(new StoredValue[queue_capacity],
                                    queue_capacity);
      task_count_[i] = 0;
    }
  }

  ~SyncShardedMap() {
    delete[] task_count_.data();
    for (auto& queue : task_queue_) {
      delete[] queue.data();
    }
    for (auto& queue : task_queue_swap_) {
      delete[] queue.data();
    }
  }

  class Shard {
    SyncShardedMap& sharded_map_;
    const size_t thread_id_;
    SeqHashMap& map_;
    Queue& task_queue_;
    Queue& task_queue_swap_;
    std::atomic_size_t& task_count_;

  public:
    Shard(SyncShardedMap& sharded_map, size_t thread_id)
        : sharded_map_(sharded_map),
          thread_id_(thread_id),
          map_(sharded_map_.map_[thread_id]),
          task_queue_(sharded_map_.task_queue_[thread_id]),
          task_queue_swap_(sharded_map_.task_queue_swap_[thread_id]),
          task_count_(sharded_map.task_count_[thread_id]) {}

    /// @brief Inserts or updates a new value in the map, depending on whether
    /// @param k The key to insert or update a value for.
    /// @param in_value The value with which to insert or update.
    inline void insert_or_update_direct(const K& k, InputValue&& in_value) {
      auto res = map_.find(k);
      assert(k.hash_ != 0);
      if (res == map_.end()) {
        // If the value does not exist, insert it
        K key = k;
        V initial = UpdateFn::init(key, std::move(in_value));
        auto [a, b] = map_.emplace(key, std::move(initial));
        assert(b);
        sharded_map_.num_inserts_.fetch_add(1, mem::acq_rel);
      } else {
        // Otherwise, update it.
        V& val = res->second;
        UpdateFn::update(k, val, std::move(in_value));
        sharded_map_.num_updates_.fetch_add(1, mem::acq_rel);
      }
    }

    void handle_queue_sync(bool make_others_wait = true) {
      if (make_others_wait) {
        // If this value is >0 then other threads will also handle their queue
        // when trying to insert
        sharded_map_.threads_handling_queue_.fetch_add(1, mem::seq_cst);
      }
      sharded_map_.barrier_.arrive_and_wait();

      handle_queue();

      sharded_map_.barrier_.arrive_and_wait();
      if (make_others_wait) {
        sharded_map_.threads_handling_queue_.fetch_sub(1, mem::seq_cst);
      }
    }

    /// @brief Handles this thread's queue, inserting or updating all values in
    ///     its queue, waiting for other threads to be
    ///     done with their handle_queue call.
    void handle_queue() {
      const size_t num_tasks_raw = task_count_.exchange(0, mem::seq_cst);
      assert(num_tasks_raw <= task_queue_.size());
      const size_t num_tasks = std::min(num_tasks_raw, task_queue_.size());
      if (num_tasks == 0) {
        return;
      }
      // std::swap(task_queue_, task_queue_swap_);

      static unsigned char zeroed[sizeof(StoredValue)];
      memset(&zeroed, 0, sizeof(StoredValue));
      //  Handle all tasks in the queue
      for (size_t i = 0; i < num_tasks; ++i) {
        // bool is_eq = memcmp(zeroed, &task_queue_swap_[i],
        // sizeof(StoredValue)); assert(!"hello" || is_eq);
        auto& entry = task_queue_[i];
        insert_or_update_direct(entry.first, std::move(entry.second));
        // memset(&task_queue_swap_[i], 0, sizeof(StoredValue));
      }
    }

    /// @brief Inserts or updates a new value in the map.
    ///
    /// If the value is inserted into the current thread's map,
    /// it is inserted immediately. If not, then it is added to that thread's
    /// queue. It will only be inserted into the map, once the thread comes
    /// around to handle its queue using the handle_queue method.
    ///
    /// @param pair The key-value pair to insert or update.
    void insert(StoredValue&& pair) {
      if (sharded_map_.threads_handling_queue_.load(mem::seq_cst) > 0) {
        handle_queue_sync();
      }
      const size_t hash = Hasher{}(pair.first);
      const size_t target_thread_id = sharded_map_.mix_select(hash);
      if (target_thread_id == thread_id_) {
        // If the target thread is this thread, insert the value directly
        insert_or_update_direct(pair.first, std::move(pair.second));
        return;
      }

      // Otherwise enqueue the new value in the target thread
      Queue* q = &sharded_map_.task_queue_[target_thread_id];
      std::atomic_size_t& target_task_count =
          sharded_map_.task_count_[target_thread_id];
      // size_t task_idx = target_task_count.fetch_add(1, mem::seq_cst);

      // sharded_map_.mtx_.lock();
      size_t task_idx = target_task_count.fetch_add(1, mem::seq_cst);
      // If the target queue is full, signal to the other threads, that they
      // need to handle their queue and handle this thread's queue
      if (task_idx >= sharded_map_.task_queue_[target_thread_id].size()) {
        // sharded_map_.mtx_.unlock();
        //  Since we incremented that thread's task count, but didn't insert
        //  anything, we need to decrement it again so that it has the correct
        //  value
        target_task_count.fetch_sub(1, mem::seq_cst);
        handle_queue_sync();
        // Since the queue was handled, the task count is now 0
        // task_idx = target_task_count.fetch_add(1, mem::seq_cst);
        insert(std::move(pair));
        return;
      }
      // assert(task_idx < q->size());

      // size_t num_tasks_raw = target_task_count.fetch_add(1);
      // sharded_map_.mtx_.unlock();

      // if (num_tasks_raw >= task_queue_.size()) {
      //   std::cerr << "man: " << num_tasks_raw << std::endl;
      // }
      assert(task_idx < task_queue_.size());
      // Insert the value into the queue
      sharded_map_.task_queue_[target_thread_id][task_idx] = std::move(pair);
    }

    /// @brief Inserts or updates a new value in the map.
    ///
    /// If the value is inserted into the current thread's map,
    /// it is inserted immediately. If not, then it is added to that thread's
    /// queue. It will only be inserted into the map, once the thread comes
    /// around to handle its queue using the handle_queue method.
    ///
    /// @param key The key of the value to insert.
    /// @param value The value to associate with the key.
    inline void insert(K& key, InputValue value) {
      insert(StoredValue(key, value));
    }
  };

  Shard get_shard(const size_t thread_id) {
    return Shard(*this, thread_id);
  }

  /// @brief Returns the number of key-value pairs in the map.
  ///
  /// Note, that this method calculates the size for each map separately and
  ///     is therefore not O(1).
  /// @return The number of key-value pairs in the map.
  [[nodiscard]] size_t size() const {
    size_t size = 0;
    for (const SeqHashMap& map : map_) {
      size += map.size();
    }
    return size;
  }

  Whereabouts where(const K& k) {
    const size_t hash = Hasher{}(k);
    const size_t target_thread_id = mix_select(hash);
    SeqHashMap& map = map_[target_thread_id];
    typename SeqHashMap::iterator it = map.find(k);
    if (it != map.end()) {
      return IN_MAP;
    }
    Queue& queue = task_queue_[target_thread_id];
    for (size_t i = 0; i < task_count_[target_thread_id]; ++i) {
      if (queue[i].first == k) {
        return IN_QUEUE;
      }
    }
    return NOWHERE;
  }

  /// @brief Runs a method for each value in the map.
  ///
  /// The given function must take const references to a key and a value
  ///     respectively.
  /// @param f The function or lambda to run for each value.
  void for_each(std::invocable<const K&, const V&> auto f) const {
    for (const SeqHashMap& map : map_) {
      for (const auto& [k, v] : map) {
        f(k, v);
      }
    }
  }

  SeqHashMap::iterator end() {
    return map_.back().end();
  }

  SeqHashMap::iterator find(const K& key) {
    const size_t hash = Hasher{}(key);
    const size_t target_thread_id = mix_select(hash);
    SeqHashMap& map = map_[target_thread_id];
    typename SeqHashMap::iterator it = map.find(key);
    if (it == map.end()) {
      return end();
    }
    return it;
  }

  void print_map_loads() {
    for (size_t i = 0; i < map_.size(); ++i) {
      std::cout << "Map " << i << " load: " << map_[i].size() << std::endl;
    }
  }

  void print_queue_upd() {
    auto so = std::osyncstream(std::cout);

    for (size_t i = 0; i < map_.size(); ++i) {
      so << "Queue " << i << " load: " << task_count_[i].load(mem::acquire)
         << "\n";
    }
    so << std::endl;
  }

  void print_ins_upd() {
    std::osyncstream(std::cout)
        << "Inserts: " << num_inserts_.load()
        << "\nUpdates: " << num_updates_.load() << std::endl;
  }

  std::barrier<decltype(FN)>& barrier() {
    return barrier_;
  }

}; // namespace pasta

} // namespace pasta
