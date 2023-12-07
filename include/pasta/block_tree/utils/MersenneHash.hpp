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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <pasta/bit_vector/bit_vector.hpp>
#include <robin_hood.h>
#include <span>
#include <vector>

#ifdef BT_INSTRUMENT
#  include <iostream>
#endif

namespace pasta {

#ifdef BT_INSTRUMENT
static std::atomic_size_t mersenne_hash_comparisons = 0;
static std::atomic_size_t mersenne_hash_equals = 0;
static std::atomic_size_t mersenne_hash_collisions = 0;

void print_hash_data() {
  std::cout << "comparisons: " << mersenne_hash_comparisons
            << ", equals: " << mersenne_hash_equals
            << ", collisions: " << mersenne_hash_collisions
            << ", percent equals: "
            << 100 * mersenne_hash_equals / ((double)mersenne_hash_comparisons)
            << ", percent collisions: "
            << 100 * mersenne_hash_collisions /
                   ((double)mersenne_hash_comparisons)
            << std::endl;
}
#endif

template <typename T>
class MersenneHash {
public:
  __extension__ typedef unsigned __int128 uint128_t;
  std::span<const T> text_;
  uint128_t hash_;
  uint32_t start_;
  uint32_t length_;
  MersenneHash(std::vector<T> const& text,
               const uint128_t hash,
               const uint64_t start,
               const uint64_t length)
      : text_(text),
        hash_(hash),
        start_(start),
        length_(length){};

  MersenneHash(const std::span<const T> text,
               const uint128_t hash,
               const uint64_t start,
               const uint64_t length)
      : text_(text),
        hash_(hash),
        start_(start),
        length_(length){};

  constexpr MersenneHash() : text_(), hash_(0), start_(0), length_(0){};

  constexpr MersenneHash(const MersenneHash& other) = default;
  constexpr MersenneHash(MersenneHash&& other) = default;

  MersenneHash& operator=(const MersenneHash& other) = default;
  MersenneHash& operator=(MersenneHash&& other) = default;

  bool operator==(const MersenneHash& other) const {
#ifdef BT_INSTRUMENT
    ++mersenne_hash_comparisons;
#endif
    // if (length_ != other.length_)
    // return false;
    // std::cout << static_cast<uint64_t>(hash_) << ", "
    //          << static_cast<uint64_t>(other.hash_) << std::endl;
    if (hash_ != other.hash_)
      return false;

    const bool is_same = memcmp(text_.data() + start_,
                                other.text_.data() + other.start_,
                                length_) == 0;

#ifdef BT_INSTRUMENT
    if (!is_same) {
      // The hash is the same but the substring isn't => collision
      ++mersenne_hash_collisions;
    } else {
      // The substrings are the same
      ++mersenne_hash_equals;
    }
#endif
    return is_same;
  };
};

template <>
class MersenneHash<bool> {
public:
  __extension__ typedef unsigned __int128 uint128_t;
  /// @brief The whole string in which the substring lies
  std::span<const std::byte> text_;
  uint128_t hash_;
  /// @brief The bit start-position of the hashed substring
  uint32_t start_;
  /// @brief The number of bits in the hashed substring
  uint32_t length_;

  template <typename T>
  constexpr MersenneHash(const std::span<T> text,
                         const uint128_t hash,
                         const uint64_t start,
                         const uint64_t length)
      : text_(std::as_bytes(text)),
        hash_(hash),
        start_(start),
        length_(length) {}

  MersenneHash(const pasta::BitVector& text,
               const uint128_t hash,
               const uint64_t start,
               const uint64_t length)
      : MersenneHash(text.data(), hash, start, length){};

  constexpr MersenneHash() : hash_(0), start_(0), length_(0){};

  constexpr MersenneHash(const MersenneHash& other) = default;
  constexpr MersenneHash(MersenneHash&& other) = default;

  constexpr MersenneHash& operator=(const MersenneHash& other) = default;
  constexpr MersenneHash& operator=(MersenneHash&& other) = default;

  constexpr bool operator==(const MersenneHash& other) const {
#ifdef BT_INSTRUMENT
    ++mersenne_hash_comparisons;
#endif
    if (hash_ != other.hash_) {
      return false;
    }
    size_t byte_index = start_ / 8;
    size_t bit_index = start_ % 8;
    size_t other_byte_index = other.start_ / 8;
    size_t other_bit_index = other.start_ % 8;

    bool is_same = true;
    for (size_t i = 0; i < length_; ++i) {
      if (get_bit(text_, byte_index, bit_index) !=
          get_bit(other.text_, other_byte_index, other_bit_index)) {
        is_same = false;
        break;
      }
      ++bit_index;
      ++other_bit_index;
      if (bit_index == 8) {
        bit_index = 0;
        ++byte_index;
      }

      if (other_bit_index == 8) {
        other_bit_index = 0;
        ++other_byte_index;
      }
    }

#ifdef BT_INSTRUMENT
    if (!is_same) {
      // The hash is the same but the substring isn't => collision
      ++mersenne_hash_collisions;
    } else {
      // The substrings are the same
      ++mersenne_hash_equals;
    }
#endif
    return is_same;
  };

  [[nodiscard]] constexpr std::span<const std::byte> overlapping_range() const {
    return text_.subspan(start_ / 8, ((length_ - 1) / 8) + 1);
  }

private:
  constexpr static bool get_bit(const std::span<const std::byte> v,
                                const size_t byte_index,
                                const size_t bit_index) {
    return (v[byte_index] & std::byte{static_cast<uint8_t>(1 << bit_index)}) >
           std::byte{0};
  }
};

} // namespace pasta

template <typename T>
struct std::hash<pasta::MersenneHash<T>> {
  typename pasta::MersenneHash<T>::uint128_t
  operator()(const pasta::MersenneHash<T>& hS) const {
    return hS.hash_;
  }
}; // namespace std

/******************************************************************************/
