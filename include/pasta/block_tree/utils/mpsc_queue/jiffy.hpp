#pragma once

#include "queue.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>

namespace pasta {

namespace jiffy {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wreorder"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <MpScQueue.h>
#pragma GCC diagnostic pop
} // namespace jiffy

template <std::copyable T>
class JiffyQueue {
  std::unique_ptr<jiffy::MpScQueue<T>> queue_;
  std::atomic_size_t size_;
  size_t capacity_;
  bool space_available_;
  std::condition_variable enqueue_cv_;
  std::mutex enqueue_mtx_;

public:
  explicit JiffyQueue(size_t capacity, size_t)
      : queue_(std::make_unique<jiffy::MpScQueue<T>>(capacity)),
        size_(0),
        capacity_(capacity),
        space_available_(capacity > 0),
        enqueue_cv_(),
        enqueue_mtx_() {}

  JiffyQueue(JiffyQueue&& other) noexcept
      : queue_(std::move(other.queue_)),
        size_(other.size()),
        capacity_(other.capacity_),
        space_available_(other.space_available_),
        enqueue_cv_(),
        enqueue_mtx_() {}

  ~JiffyQueue() {
    while (size() > 0) {
      dequeue().release();
    }
  };

  bool enqueue(std::unique_ptr<T>&& elem) {
    std::unique_lock<std::mutex> lock(enqueue_mtx_);
    enqueue_cv_.wait(lock, [this] { return this->space_available_; });
    const size_t size = size_.load();
    space_available_ = size + 1 < capacity_;
    assert(size < capacity_);
    size_.fetch_add(1);
    queue_->enqueue(*elem.release());
    return true;
  }

  std::unique_ptr<T> dequeue() {
    T* t = new T;
    size_.fetch_sub(1);
    queue_->dequeue(*t);
    this->space_available_ = true;
    enqueue_cv_.notify_one();
    return std::unique_ptr<T>(static_cast<T*>(t));
  }

  [[nodiscard]] __attribute_noinline__ size_t size() const {
    return size_.load();
  }

  [[nodiscard]] __attribute_noinline__ size_t capacity() const {
    return capacity_;
  }
};

} // namespace pasta