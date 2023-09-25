#pragma once

#include <concepts>
#include <cstddef>
#include <memory>

namespace pasta {

namespace internal {
using Capacity = size_t;
using ThreadCount = size_t;
} // namespace internal

template <typename Queue, typename Elem>
// Should have a constructor that allows the construction with a given capacity
// and thread count, should the queue need it
concept MpscQueue =
    std::constructible_from<Queue, internal::Capacity, internal::ThreadCount> &&
    requires(Queue q, std::unique_ptr<Elem>&& e) {
      // enqueue should enqueue a value and return true, if the element was
      //   enqueued (i.e. there was space)
      { q.enqueue(std::move(e)) } -> std::convertible_to<bool>; // aa
      // dequeue dequeue the oldest value and return it
      { q.dequeue() } -> std::convertible_to<std::unique_ptr<Elem>>;
      // size should return the current number of elements
      { q.size() } -> std::convertible_to<size_t>;
      // capacity should return the maximum number of elements
      // this queue can hold
      { q.capacity() } -> std::convertible_to<size_t>;
    };

} // namespace pasta
