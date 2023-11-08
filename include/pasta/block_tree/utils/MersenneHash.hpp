/*******************************************************************************
 * This file is part of pasta::block_tree
 *
 * Copyright (C) 2022 Daniel Meyer
 * Copyright (C) 2023 Etienne Palanga
 *
 * pasta::block_tree is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * pasta::block_tree is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with pasta::block_tree.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <syncstream>
#include <type_traits>
#include <vector>
#include <emmintrin.h>

namespace pasta {

#ifdef BT_INSTRUMENT
static std::atomic_size_t mersenne_hash_comparisons = 0;
static std::atomic_size_t mersenne_hash_equals = 0;
static std::atomic_size_t mersenne_hash_collisions = 0;
#endif

template <typename T>
class MersenneHash {
public:
  __extension__ typedef unsigned __int128 uint128_t;
  const std::vector<T>* text_;
  uint128_t hash_;
  uint32_t start_;
  uint32_t length_;
  MersenneHash(std::vector<T> const& text,
               uint128_t hash,
               uint64_t start,
               uint64_t length)
      : text_(&text),
        hash_(hash),
        start_(start),
        length_(length){};

  constexpr MersenneHash() : text_(nullptr), hash_(0), start_(0), length_(0){};

  constexpr MersenneHash(const MersenneHash& other) = default;
  constexpr MersenneHash(MersenneHash&& other) = default;

  MersenneHash<T>& operator=(const MersenneHash<T>& other) = default;
  MersenneHash<T>& operator=(MersenneHash<T>&& other) = default;

  bool operator==(const MersenneHash& other) const {
#ifdef BT_INSTRUMENT
    mersenne_hash_comparisons++;
#endif
    // if (length_ != other.length_)
    // return false;
    if (hash_ != other.hash_)
      return false;

#define MH_MEMCMP

#ifndef MH_NOCMP
    const std::vector<T>& text = *text_;
    const std::vector<T>& other_text = *other.text_;
#endif

#ifdef MH_LOOP
    for (uint64_t i = 0; i < length_; i++) {
      if (text[start_ + i] != other_text[other.start_ + i]) {
        return false;
      }
    }
    return true;
#elif defined MH_PACKED_LOOP
    const size_t num_blocks = length_ / 8;
    for (size_t block = 0; block < num_blocks; ++block) {
      uint64_t a =
          *reinterpret_cast<const uint64_t*>(text.data() + start_ + block * 8);
      uint64_t b = *reinterpret_cast<const uint64_t*>(other_text.data() +
                                                      other.start_ + block * 8);
      if (a != b) {
        return false;
      }
    }
    return memcmp(text_->data() + start_ + num_blocks * 8,
                  other.text_->data() + other.start_ + num_blocks * 8,
                  length_ - num_blocks * 8) == 0;
#elif defined MH_PACKED_LOOP_UNROLL
    const size_t num_blocks = length_ / 16;
    for (size_t block = 0; block < num_blocks; ++block) {
      uint64_t a1 = *reinterpret_cast<const uint64_t*>(text.data() + start_ +
                                                       2 * block * 8);
      uint64_t a2 = *reinterpret_cast<const uint64_t*>(text.data() + start_ +
                                                       (2 * block + 1) * 8);
      uint64_t b1 = *reinterpret_cast<const uint64_t*>(
          other_text.data() + other.start_ + 2 * block * 8);
      uint64_t b2 = *reinterpret_cast<const uint64_t*>(
          other_text.data() + other.start_ + (2 * block + 1) * 8);
      if (a1 != b1 || a2 != b2) {
        return false;
      }
    }
    return memcmp(text_->data() + start_ + num_blocks * 16,
                  other.text_->data() + other.start_ + num_blocks * 16,
                  length_ - num_blocks * 16) == 0;
#elif defined MH_SSE
    // Requires SSE2
    const size_t num_blocks = length_ / 16;
    for (size_t block = 0; block < num_blocks; ++block) {
      __m128i_u a = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(
          text.data() + start_ + block * 16));
      __m128i_u b = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(
          other_text.data() + other.start_ + block * 16));
      __m128i_u res = _mm_cmpeq_epi8(a, b);
      if (0xFFFF != _mm_movemask_epi8(res)) {
        return false;
      }
    }
    return memcmp(text_->data() + start_ + num_blocks * 16,
                  other.text_->data() + other.start_ + num_blocks * 16,
                  length_ - num_blocks * 16) == 0;
#elif defined MH_AVX
    // Requires AVX-2
    const size_t num_blocks = length_ / 32;
    for (size_t block = 0; block < num_blocks; ++block) {
      __m256i a = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(text.data() + start_ + block * 32));
      __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
          other_text.data() + other.start_ + block * 32));
      __m256i res = _mm256_cmpeq_epi8(a, b);
      if (static_cast<int>(0xFFFFFFFF) != _mm256_movemask_epi8(res)) {
        return false;
      }
    }
    return memcmp(text_->data() + start_ + num_blocks * 32,
                  other.text_->data() + other.start_ + num_blocks * 32,
                  length_ - num_blocks * 32) == 0;

#elif defined MH_MEMCMP
    const bool b = memcmp(text.data() + start_,
                          other_text.data() + other.start_,
                          length_) == 0;

#  ifdef BT_INSTRUMENT
    if (!b) {
      mersenne_hash_collisions++;
    } else {
      mersenne_hash_equals++;
    }
#  endif
    return b;

#elif defined MH_SPARSECMP
    // Just compare as much data as fits into a word, choosing the largest
    // integer type that fits into the length of this window
    static constexpr size_t bytes_per_t = sizeof(T);
    const T* text_ptr = text.data() + start_;
    const T* other_text_ptr = other_text.data() + other.start_;
    bool b;
    switch (length_ * bytes_per_t) {
      case 1:
        b = text[0] == other_text[0];
        break;
      case 2:
      case 3: {
        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(text_ptr);
        const uint16_t* other_ptr =
            reinterpret_cast<const uint16_t*>(other_text_ptr);
        b = *ptr == *other_ptr;
        break;
      }
      case 4:
      case 5:
      case 6:
      case 7: {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(text_ptr);
        const uint32_t* other_ptr =
            reinterpret_cast<const uint32_t*>(other_text_ptr);
        b = *ptr == *other_ptr;
        break;
      }
      default: {
        const uint64_t* ptr = reinterpret_cast<const uint64_t*>(text_ptr);
        const uint64_t* other_ptr =
            reinterpret_cast<const uint64_t*>(other_text_ptr);
        b = *ptr == *other_ptr;
      }
    }
#  ifdef BT_INSTRUMENT
    if (!b) {
      mersenne_hash_collisions++;
    } else {
      mersenne_hash_equals++;
    }
#  endif
    return b;
#elif defined MH_NOCMP
    return true;
#endif
  }
};

} // namespace pasta

namespace std {
template <typename T>
struct hash<pasta::MersenneHash<T>> {
  pasta::MersenneHash<T>::uint128_t
  operator()(const pasta::MersenneHash<T>& hS) const {
    return hS.hash_;
  }
};
} // namespace std

/******************************************************************************/
