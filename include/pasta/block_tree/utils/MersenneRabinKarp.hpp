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

#include "pasta/block_tree/utils/MersenneHash.hpp"

#include <iostream>

namespace pasta {

///
/// @brief A Rabin-Karp rolling hasher.
///
/// @tparam T The type of the characters in the text.
/// @tparam size_type The type to use for indexing etc.
/// @tparam mersenne_exponent If using a mersenne prime 2^p-1, then this should
/// be p. If this is 0, a normal modulus operation will be used
///
template <class T, class size_type, uint8_t mersenne_exponent = 0>
class MersenneRabinKarp {
  __extension__ typedef unsigned __int128 uint128_t;

public:
  /// The text being hashed
  std::vector<T> const& text_;
  uint128_t sigma_;
  /// The start index of the currently hashed window
  uint64_t init_;
  /// The window size of this hasher
  uint64_t length_;
  /// A large prime used for modulus operations
  uint128_t prime_;
  /// The current hash value
  uint64_t hash_;
  uint128_t max_sigma_;

  /// @brief Construct a new Rabin Karp hasher.
  /// @param text The text to hash.
  /// @param sigma The alphabet size.
  /// @param init The start index of the first hashed window in the text.
  /// @param length The window size.
  /// @param prime A large prime used for modulus operations.
  MersenneRabinKarp(std::vector<T> const& text,
                    uint64_t sigma,
                    uint64_t init,
                    uint64_t length,
                    uint128_t prime)
      : text_(text),
        sigma_(sigma),
        init_(init),
        length_(length),
        prime_(prime) {
    max_sigma_ = 1;
    uint128_t fp = 0;
    uint128_t sigma_c = 1;
    for (uint64_t i = init_; i < init_ + length_; i++) {
      fp = fp * sigma;
      fp = mersenneModulo(fp + text_[i]);
    }
    for (uint64_t i = 0; i < length_ - 1; i++) {
      sigma_c = mersenneModulo(sigma_c * sigma_);
    }
    hash_ = (uint64_t)(fp);
    max_sigma_ = (uint64_t)(sigma_c);
  };

  void restart(uint64_t index) {
    if (index + length_ >= text_.size()) {
      return;
    }
    init_ = index;
    uint128_t fp = 0;
    for (uint64_t i = init_; i < init_ + length_; i++) {
      fp = fp * sigma_;
      fp = mersenneModulo(fp + text_[i]);
    }
    hash_ = (uint64_t)(fp);
  };

  inline uint128_t mersenneModulo(uint128_t k) {
    if constexpr (mersenne_exponent == 0) {
      return k % prime_;
    } else {
      uint128_t i = (k & prime_) + (k >> mersenne_exponent);
      return (i >= prime_) ? i - prime_ : i;
    }
  };

  inline MersenneHash<T> current_hash() const {
    return MersenneHash<T>(text_, hash_, init_, length_);
  }

  void next() {
    if (text_.size() <= init_ + length_) {
      return;
    }

    uint128_t fp = hash_;
    T out_char = text_[init_];
    T in_char = text_[init_ + length_];
    const uint128_t out_char_influence = mersenneModulo(out_char * max_sigma_);
    // Conditionally add the prime, of the out_char_influence is too large
    fp += prime_ * (out_char_influence > hash_) - out_char_influence;
    fp *= sigma_;
    fp += in_char;
    fp = mersenneModulo(fp);
    hash_ = (uint64_t)(fp);
    init_++;
  };
};

} // namespace pasta

/******************************************************************************/
