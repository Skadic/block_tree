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
#include <robin_hood.h>
#include <vector>

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
               const uint128_t hash,
               const uint64_t start,
               const uint64_t length)
      : text_(&text),
        hash_(hash),
        start_(start),
        length_(length){};

  constexpr MersenneHash() : text_(nullptr), hash_(0), start_(0), length_(0){};

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

    const bool is_same = memcmp(text_->data() + start_,
                                other.text_->data() + other.start_,
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

} // namespace pasta

namespace std {
template <typename T>
struct hash<pasta::MersenneHash<T>> {
  typename pasta::MersenneHash<T>::uint128_t
  operator()(const pasta::MersenneHash<T>& hS) const {
    return hS.hash_;
  }
};
} // namespace std

/******************************************************************************/
