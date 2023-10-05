#pragma once

#include <barrier>
#include <cassert>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <latch>
#include <omp.h>
#include <pasta/block_tree/utils/concepts.hpp>
#include <pasta/block_tree/utils/mpsc_queue/jiffy.hpp>
#include <semaphore>
#include <span>
#include <unordered_map>
#include <vector>

namespace pasta {

///
/// @brief An update function which on update just overwrites the value.
///
/// @tparam K The key type saved in the hash map.
/// @tparam V The value type saved in the hash map.
///
template <typename K, typename V>
struct Overwrite {
  using InputValue = V;

  inline static void update(K&, V& value, V&& input_value) {
    value = input_value;
  }

  inline static V init(K&, V&& input_value) {
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
struct Keep {
  using InputValue = V;
  inline static void update(K&, V&, V&&) {}

  inline static V init(K&, V&& input_value) {
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
template <typename K,
          typename V,
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

  using StoredValue = std::pair<K, InputValue>;

  using mem = std::memory_order;

  /// @brief A value between 0 and 1, determining to which extent
  /// each thread's queue should be filled, before the thread is signaled to
  /// handle its queued operations.
  ///
  /// If this value is 0.25, then the thread's value in threshold_met_
  /// is set to true, signaling that the thread should handle its requests
  /// in task_queue_
  const double fill_threshold_;
  /// @brief The number of threads operating on this map.
  const size_t thread_count_;
  /// @brief Contains a hash map for each thread
  std::vector<SeqHashMap> map_;
  /// @brief Contains a task queue for each thread, holding insert
  /// operations for each thread.
  std::vector<std::vector<StoredValue>> task_queue_;
  /// @brief Contains the number of tasks in each thread's queue.
  std::span<std::atomic_size_t> task_count_;
  /// @brief Contains the number of threads currently handling their queues.
  /// This is used 1. signal to other threads that they should handle their
  /// queue, and 2. to keep track of whether all threads have handled their
  /// queues.
  std::atomic_size_t threads_handling_queue_;

  constexpr static std::invocable auto FN = []() noexcept {
  };

  std::barrier<decltype(FN)> barrier_;

  /// https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html
  inline uint64_t mix_select(uint64_t key) {
    key ^= (key >> 33);
    key *= 0xff51afd7ed558ccd;
    key ^= (key >> 33);
    key *= 0xc4ceb9fe1a85ec53;
    key ^= (key >> 33);
    return key % thread_count_;
  }

public:
  std::atomic_size_t num_updates_;
  std::atomic_size_t num_inserts_;

  std::condition_variable aa;
  //
  /// @brief Creates a new sharded map.
  ///
  /// @param fill_threshold The fill percentage (between 0 and 1) above which
  /// a thread is signaled to handle its own tasks.
  /// @param thread_count The exact number of threads working on this map.
  /// @param queue_capacity The maximum amount of tasks allowed in each queue.
  ///
  SyncShardedMap(double fill_threshold,
                 size_t thread_count,
                 size_t queue_capacity)
      : fill_threshold_(fill_threshold),
        thread_count_(thread_count),
        map_(),
        task_queue_(),
        task_count_(),
        threads_handling_queue_(0),
        barrier_(thread_count, FN),
        aa() {
    assert(0 <= fill_threshold && fill_threshold <= 1);
    map_.reserve(thread_count);
    task_queue_.reserve(thread_count);
    auto* task_arr = new std::atomic_size_t[thread_count];
    task_count_ = std::span<std::atomic_size_t>(task_arr, thread_count);
    for (size_t i = 0; i < thread_count; i++) {
      map_.emplace_back();
      task_queue_.emplace_back(queue_capacity);
      task_count_[i] = 0;
    }
  }

  ~SyncShardedMap() {
    delete[] task_count_.data();
  }

  class Shard {
    SyncShardedMap& sharded_map_;
    const size_t thread_id_;
    SeqHashMap& map_;
    std::vector<StoredValue>& task_queue_;
    std::atomic_size_t& task_count_;
    size_t last_cycle;

  public:
    Shard(SyncShardedMap& sharded_map, size_t thread_id)
        : sharded_map_(sharded_map),
          thread_id_(thread_id),
          map_(sharded_map_.map_[thread_id]),
          task_queue_(sharded_map_.task_queue_[thread_id]),
          task_count_(sharded_map.task_count_[thread_id]),
          last_cycle(0) {}

    /// @brief Inserts or updates a new value in the map, depending on whether
    /// @param k The key to insert or update a value for.
    /// @param in_value The value with which to insert or update.
    inline void insert_or_update_direct(K& k, InputValue&& in_value) {
      auto res = map_.find(k);
      if (res == map_.end()) {
        // If the value does not exist, insert it
        V initial = UpdateFn::init(k, std::move(in_value));
        K key = k;
        map_.emplace(key, std::move(initial));
        sharded_map_.num_inserts_.fetch_add(1, mem::acq_rel);
      } else {
        // Otherwise, update it.
        V& val = res->second;
        UpdateFn::update(k, val, std::move(in_value));
        sharded_map_.num_updates_.fetch_add(1, mem::acq_rel);
      }
    }

    void handle_queue_sync() {
      sharded_map_.threads_handling_queue_.fetch_add(1, mem::seq_cst);
      sharded_map_.barrier_.arrive_and_wait();

      handle_queue();

      sharded_map_.barrier_.arrive_and_wait();
      sharded_map_.threads_handling_queue_.fetch_sub(1, mem::seq_cst);
    }

    /// @brief Handles this thread's queue, inserting or updating all values in
    ///     its queue, waiting for other threads to be
    ///     done with their handle_queue call.
    void handle_queue() {
      const size_t num_tasks =
          std::min(task_count_.exchange(0, mem::acq_rel), task_queue_.size());
      // Handle all tasks in the queue
      for (size_t i = 0; i < num_tasks; ++i) {
        auto entry = task_queue_[i];
        insert_or_update_direct(entry.first, std::move(entry.second));
      }
      // All tasks are handled and this thread is done
    }

    /// @brief Inserts or updates a new value in the map.
    ///
    /// If the value is inserted into the current thread's map,
    /// it is inserted immediately. If not, then it is added to that thread's
    /// queue. It will only be inserted into the map, once the thread comes
    /// around to handle its queue using the handle_queue method.
    ///
    /// @param pair The key-value pair to insert or update.
    void insert(StoredValue&& pair, std::condition_variable& cv) {
      if (sharded_map_.threads_handling_queue_.load(mem::acquire) > 0) {
        handle_queue_sync();
      }
      const size_t hash = Hasher{}(pair.first);
      const size_t target_thread_id = sharded_map_.mix_select(hash);

      // Otherwise enqueue the new value in the target thread
      std::vector<StoredValue>& q = sharded_map_.task_queue_[target_thread_id];
      std::atomic_size_t& target_task_count =
          sharded_map_.task_count_[target_thread_id];
      size_t task_idx = target_task_count.fetch_add(1, mem::seq_cst);
      // If the target queue is full, signal to the other threads, that they
      // need to handle their queue and handle this thread's queue
      if (task_idx >= q.size() ||
          sharded_map_.threads_handling_queue_.load(mem::acquire)) {
        handle_queue_sync();
        // Since the queue was handled, the task count is now 0
        task_idx =
            sharded_map_.task_count_[target_thread_id].fetch_add(1,
                                                                 mem::acq_rel);
        // std::cout << "e" << task_idx << std::endl;
        // assert(prev_task_idx == 0 || prev_task_idx > task_idx);
      }
      if (task_idx >= q.size()) {
        std::cout << "i: " << task_idx << ", qsize: " << q.size() << std::endl;
      }
      assert(task_idx < q.size());
      // Insert the value into the queue
      q.at(task_idx) = std::move(pair);
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
    inline void insert(K& key, InputValue value, std::condition_variable& cv) {
      insert(StoredValue(key, value), cv);
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

  /// @brief Runs a method for each value in the map.
  ///
  /// The given function must take const references to a key and a value
  ///     respectively.
  /// @param f The function or lambda to run for each value.
  void for_each(std::invocable<const K&, const V&> auto f) {
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

  void print_ins_upd() {
    std::cout << "Inserts: " << num_inserts_.load() << std::endl;
    std::cout << "Updates: " << num_updates_.load() << std::endl;
  }

  std::barrier<decltype(FN)>& barrier() {
    return barrier_;
  }

}; // namespace pasta

} // namespace pasta
