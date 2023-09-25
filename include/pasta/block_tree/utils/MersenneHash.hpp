/*******************************************************************************
 * This file is part of pasta::block_tree
 *
 * Copyright (C) 2022 Daniel Meyer
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
#include <functional>
#include <iostream>
#include <type_traits>
#include <vector>

namespace pasta {

template <typename T>
class MersenneHash {
public:
  const std::vector<T>* text_;
  uint64_t hash_;
  uint32_t start_;
  uint32_t length_;
  MersenneHash(std::vector<T> const& text,
               size_t hash,
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
    //            std::cout << start_ << " " << other.start_ << std::endl;
    if (length_ != other.length_)
      return false;

    const std::vector<T>& text = *text_;
    const std::vector<T>& other_text = *other.text_;

    for (uint64_t i = 0; i < length_; i++) {
      if (text[start_ + i] != other_text[other.start_ + i]) {
        return false;
      }
    }
    return hash_ == other.hash_;
  }
};

} // namespace pasta

namespace std {
template <typename T>
struct hash<pasta::MersenneHash<T>> {
  std::size_t operator()(const pasta::MersenneHash<T>& hS) const {
    return hS.hash_;
  }
};
} // namespace std

/******************************************************************************/
