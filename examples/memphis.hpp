// Be very careful when including this header! It will overwrite memory allocation operators! 
// Only include it, if you want to track your heap memory.
// Source: https://en.cppreference.com/w/cpp/memory/new/operator_new

#pragma once
#pragma GCC system_header

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>

#ifndef MEMPHIS_ENABLED
#  define MEMPHIS_ENABLED
#endif // !MEMPHIS_ENABLED

namespace memphis {
/// @brief The current heap usage in bytes.
std::atomic_size_t current_heap_usage = 0;
/// @brief The number of heap allocations since the start of the program.
std::atomic_size_t num_allocations = 0;
/// @brief The maximum heap usage over the course of the program's runtime in
/// bytes.
std::atomic_size_t peak_heap_usage = 0;

///
/// @brief Retrieves the size of the allocation of the given pointer in bytes.
///
/// @param ptr A pointer to some allocation.
/// @return The size of the allocation in bytes.
///
[[nodiscard("allocation size determined but discarded")]] inline size_t
allocation_size(const void* ptr);
} // namespace memphis

#ifdef _WIN32
#  include <malloc.h>
inline size_t memphis::allocation_size(const void* ptr) {
  return _msize(const_cast<void*>(ptr));
}
#elif defined unix
#  include <malloc.h>
inline size_t memphis::allocation_size(const void* ptr) {
  return malloc_usable_size(const_cast<void*>(ptr));
}
#elif defined __APPLE__
#  include <malloc/malloc.h>
inline size_t memphis::allocation_size(const void* ptr) {
  return malloc_size(ptr);
}

#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
// no inline, required by [replacement.functions]/3
void* operator new(std::size_t sz) {
  if (sz == 0)
    ++sz; // avoid std::malloc(0) which may return nullptr on success

  if (void* ptr = std::malloc(sz)) {
    const std::size_t current =
        memphis::current_heap_usage.fetch_add(sz, std::memory_order_relaxed) +
        sz;
    const std::size_t old_peak =
        memphis::peak_heap_usage.load(std::memory_order_acquire);
    memphis::peak_heap_usage.store(std::max(current, old_peak),
                                   std::memory_order_release);
    memphis::num_allocations.fetch_add(1, std::memory_order_relaxed);
    return ptr;
  }

  throw std::bad_alloc{}; // required by [new.delete.single]/3
}

// no inline, required by [replacement.functions]/3
void* operator new[](std::size_t sz) {
  if (sz == 0)
    ++sz; // avoid std::malloc(0) which may return nullptr on success

  if (void* ptr = std::malloc(sz)) {
    const std::size_t current =
        memphis::current_heap_usage.fetch_add(sz, std::memory_order_relaxed) +
        sz;
    const std::size_t old_peak =
        memphis::peak_heap_usage.load(std::memory_order_acquire);
    memphis::peak_heap_usage.store(std::max(current, old_peak),
                                   std::memory_order_release);
    memphis::num_allocations.fetch_add(1, std::memory_order_relaxed);
    return ptr;
  }

  throw std::bad_alloc{}; // required by [new.delete.single]/3
}

void operator delete(void* ptr) noexcept {
  std::size_t sz = memphis::allocation_size(ptr);
  memphis::current_heap_usage.fetch_sub(sz, std::memory_order_relaxed);
  std::free(ptr);
}

void operator delete(void* ptr, std::size_t size) noexcept {
  memphis::current_heap_usage.fetch_sub(size, std::memory_order_relaxed);
  std::free(ptr);
}

void operator delete[](void* ptr) noexcept {
  std::size_t sz = memphis::allocation_size(ptr);
  memphis::current_heap_usage.fetch_sub(sz, std::memory_order_relaxed);
  std::free(ptr);
}

void operator delete[](void* ptr, std::size_t size) noexcept {
  memphis::current_heap_usage.fetch_sub(size, std::memory_order_relaxed);
  std::free(ptr);
}
#pragma GCC diagnostic pop
