/*******************************************************************************
 * This file is part of pasta::block_tree
 *
 * Copyright (C) 2022 Daniel Meyer
 * Copyright (C) 2023 Etienne Palanga <etienne.palanga@tu-dortmund.de>
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

#include <ankerl/unordered_dense.h>
#include <bit>
#include <iostream>
#include <omp.h>
#include <pasta/bit_vector/bit_vector.hpp>
#include <pasta/bit_vector/support/flat_rank_select.hpp>
#include <pasta/bit_vector/support/optimized_for.hpp>
#include <pasta/bit_vector/support/rank.hpp>
#include <pasta/bit_vector/support/rank_select.hpp>
#include <pasta/bit_vector/support/wide_rank_select.hpp>
#include <sdsl/int_vector.hpp>
#include <vector>

namespace pasta {

template <std::signed_integral size_type>
class BitBlockTree {
public:
  /// If this is true, then the only levels of the tree start to be
  ///   included starting at the first level that contains a back block
  ///
  /// For example, if levels 0 to 5 do not contain any back blocks, then the
  /// tree will only contain levels 6 and below.
  bool CUT_FIRST_LEVELS = true;

  using BVStoreType = pasta::BitVector::RawDataType;

  /// The arity of the tree
  size_type tau_;
  size_type max_leaf_length_;
  /// The arity of the tree's root
  size_type s_ = 1;
  size_type leaf_size = 0;
  size_type amount_of_leaves = 0;
  size_type num_bits;
  bool rank_support = false;
  /// Bit vectors for each level determining whether a block is internal
  /// (=1) or not (=0)
  std::vector<pasta::BitVector*> block_tree_types_;
  std::vector<pasta::RankSelect<pasta::OptimizedFor::ONE_QUERIES>*>
      block_tree_types_rs_;
  /// For each level and each back block, contains the index of the
  /// block's source
  std::vector<sdsl::int_vector<>*> block_tree_pointers_;
  std::vector<sdsl::int_vector<>*> block_tree_offsets_;
  //    std::vector<sdsl::int_vector<>*> block_tree_encoded_;
  std::vector<int64_t> block_size_lvl_;
  std::vector<int64_t> block_per_lvl_;
  std::vector<uint8_t> leaves_;

  std::vector<uint8_t> compress_map_;
  std::vector<uint8_t> decompress_map_;
  sdsl::int_vector<> compressed_leaves_;

  ankerl::unordered_dense::map<uint8_t, size_type> chars_index_;
  std::vector<uint8_t> chars_;
  std::vector<std::vector<sdsl::int_vector<>>> c_ranks_;
  std::vector<std::vector<sdsl::int_vector<>>> pointer_c_ranks_;

  /// @brief For each level and each block, contains the number of 1s up to (and
  /// including) the block.
  std::vector<sdsl::int_vector<>> one_ranks_;
  /// @brief For each level and each back block,
  ///   contains the number of 1s up to (and including) the pointed-to area of
  ///   the back-block.
  std::vector<sdsl::int_vector<>> pointer_prefix_one_counts_;

  [[nodiscard]] size_t height() const {
    return block_tree_types_.size();
  }

  bool access(const size_type bit_index) const {
    // FIXME: As of now this works on little endian systems only
    const int64_t byte_index = bit_index / 8;
    const int64_t bit_offset = bit_index % 8;

    int64_t block_size = block_size_lvl_[0];
    int64_t block_index = byte_index / block_size;
    int64_t off = byte_index % block_size;
    for (size_t i = 0; i < height(); i++) {
      const auto& is_internal = *block_tree_types_[i];
      const auto& is_internal_rank = *block_tree_types_rs_[i];
      const auto& pointers = *block_tree_pointers_[i];
      const auto& offsets = *block_tree_offsets_[i];
      if (!is_internal[block_index]) {
        // If this block is not internal, go to its pointed-to block
        const size_t back_block_index = is_internal_rank.rank0(block_index);
        off = off + offsets[back_block_index];
        block_index = pointers[back_block_index];
        if (off >= block_size) {
          ++block_index;
          off -= block_size;
        }
      }
      block_size /= tau_;
      const int64_t child = off / block_size;
      off %= block_size;
      block_index = is_internal_rank.rank1(block_index) * tau_ + child;
    }
    const uint8_t byte =
        decompress_map_[compressed_leaves_[block_index * leaf_size + off]];
    return ((1 << bit_offset) & byte) != 0;
  };

  int64_t select(uint8_t c, size_type j) {
    auto c_index = chars_index_[c];
    auto& top_level = *block_tree_types_[0];

    auto& top_level_rs = *block_tree_types_rs_[0];
    auto& top_level_ptr = *block_tree_pointers_[0];
    auto& top_level_off = *block_tree_offsets_[0];
    size_type current_block = (j - 1) / block_size_lvl_[0];
    size_type end_block = c_ranks_[c_index][0].size() - 1;
    int64_t block_size = block_size_lvl_[0];
    // find first level block containing the jth occurrence of c with a bin
    // search
    while (current_block != end_block) {
      size_type m = current_block + (end_block - current_block) / 2;

      size_type f = (m == 0) ? 0 : c_ranks_[c_index][0][m - 1];
      if (f < j) {
        if (end_block - current_block == 1) {
          if (c_ranks_[c_index][0][m] < static_cast<uint64_t>(j)) {
            current_block = m + 1;
          }
          break;
        }
        current_block = m;
      } else {
        end_block = m - 1;
      }
    }

    // accumulator
    int64_t s = current_block * block_size - 1;
    // index that indicates how many c's are still unaccounted for
    j -= (current_block == 0) ? 0 : c_ranks_[c_index][0][current_block - 1];
    // we translate unmarked blocks on the top level independently as it differs
    // from the other levels
    if (!top_level[current_block]) {
      int64_t blk = top_level_rs.rank0(current_block);
      current_block = top_level_ptr[blk];
      int64_t g = top_level_off[blk];
      int64_t rank_d = (current_block == 0) ?
                           c_ranks_[c_index][0][0] :
                           c_ranks_[c_index][0][current_block] -
                               c_ranks_[c_index][0][current_block - 1];
      rank_d -= pointer_c_ranks_[c_index][0][blk];
      if (rank_d < j) {
        j -= rank_d;
        s += (block_size - g);
        current_block++;
      } else {
        j += pointer_c_ranks_[c_index][0][blk];
        s -= g;
      }
    }
    uint64_t i = 1;
    while (i < height()) {
      auto& current_level = *block_tree_types_[i];
      auto& current_level_rs = *block_tree_types_rs_[i];
      auto& current_level_ptr = *block_tree_pointers_[i];
      auto& current_level_off = *block_tree_offsets_[i];
      auto& prev_level_rs = *block_tree_types_rs_[i - 1];
      current_block = prev_level_rs.rank1(current_block) * tau_;
      block_size /= tau_;
      int64_t k = current_block;
      while ((int64_t)c_ranks_[c_index][i][current_block] < j) {
        current_block++;
      }
      j -= (current_block == k) ? 0 : c_ranks_[c_index][i][current_block - 1];
      s += (current_block - k) * block_size;
      if (!current_level[current_block]) {
        int64_t blk = current_level_rs.rank0(current_block);
        current_block = current_level_ptr[blk];
        int64_t g = current_level_off[blk];
        int64_t rank_d = (current_block % tau_ == 0) ?
                             c_ranks_[c_index][i][current_block] :
                             c_ranks_[c_index][i][current_block] -
                                 c_ranks_[c_index][i][current_block - 1];
        rank_d -= pointer_c_ranks_[c_index][i][blk];
        if (rank_d < j) {
          j -= rank_d;
          s += (block_size - g);
          current_block++;
        } else {
          j += pointer_c_ranks_[c_index][i][blk];
          s -= g;
        }
      }
      i++;
    }

    current_block = (*block_tree_types_rs_[i - 1]).rank1(current_block) * tau_;
    int64_t l = 0;
    while (j > 0) {
      if (compressed_leaves_[current_block * leaf_size + l] == compress_map_[c])
        j--;
      l++;
    }
    return s + l;
  }

  /// @brief Counts the number of 1-bits up to (and excluding) an index.
  size_t rank1(const size_type bit_index) const {
    const size_t byte_index = bit_index / 8;
    const auto& top_is_internal = *block_tree_types_[0];
    const auto& top_is_internal_rank = *block_tree_types_rs_[0];
    const auto& top_pointers = *block_tree_pointers_[0];
    const auto& top_offsets = *block_tree_offsets_[0];
    size_t block_size = block_size_lvl_[0];
    size_t block_index = byte_index / block_size;
    size_t block_offset = byte_index % block_size;
    size_t rank = (block_index == 0) ? 0 : one_ranks_[0][block_index - 1];
    if (!top_is_internal[block_index]) {
      // If the top block is a back block, go to it and adjust the offset
      const size_t back_block_index = top_is_internal_rank.rank0(block_index);
      rank -= pointer_prefix_one_counts_[0][back_block_index];
      block_offset += top_offsets[back_block_index];
      block_index = top_pointers[back_block_index];
      if (block_offset >= block_size) {
        // If we're exceeding the pointed-to block's offset,
        // add the ones inside of it
        rank +=
            (block_index == 0) ?
                one_ranks_[0][block_index] :
                (one_ranks_[0][block_index] - one_ranks_[0][block_index - 1]);
        ++block_index;
        block_offset -= block_size;
      }
    }

    // Go down to the next level
    block_size /= tau_;
    // How many children are we 'skipping over'
    size_t child = block_offset / block_size;
    block_offset %= block_size;
    block_index = top_is_internal_rank.rank1(block_index) * tau_ + child;

    size_t level = 1;
    while (level < height()) {
      const auto& ranks = one_ranks_[level];
      const auto& pointer_ranks = pointer_prefix_one_counts_[level];
      const auto& is_internal = *block_tree_types_[level];
      const auto& is_internal_rank = *block_tree_types_rs_[level];
      rank += (child == 0) ? 0 : ranks[block_index - 1];
      // If this block is internal, just go to the correct child
      if (is_internal[block_index]) {
        block_size /= tau_;
        child = block_offset / block_size;
        block_offset %= block_size;
        block_index = is_internal_rank.rank1(block_index) * tau_ + child;
        level++;
        continue;
      }

      // If we have a back block, we need to go to the pointed-to block
      const size_t back_block_index = is_internal_rank.rank0(block_index);
      rank -= pointer_ranks[back_block_index];
      block_offset += (*block_tree_offsets_[level])[back_block_index];
      block_index = (*block_tree_pointers_[level])[back_block_index];
      child = block_index % tau_;

      if (block_offset >= block_size) {
        // If we're exceeding the pointed-to block's offset,
        // add the ones inside of it and go to the next block
        rank += (child == 0) ? ranks[block_index] :
                               (ranks[block_index] - ranks[block_index - 1]);
        ++block_index;
        child = block_index % tau_;
        block_offset -= block_size;
      }
      const size_t remove_prefix = (child == 0) ? 0 : ranks[block_index - 1];
      rank -= remove_prefix;
    }

    // Number of leaves that exist before the leaves of the current block
    const size_type prefix_leaves = block_index - child;
    for (size_t block = 0; block < child * leaf_size; block++) {
      const uint8_t byte =
          decompress_map_[compressed_leaves_[prefix_leaves * leaf_size +
                                             block]];
      rank += std::popcount(byte);
    }
    for (size_t block = 0; block < block_offset; block++) {
      const uint8_t byte =
          decompress_map_[compressed_leaves_[block_index * leaf_size + block]];
      rank += std::popcount(byte);
    }

    // Masks to remove bits from the last byte,
    // that aren't part of the ran query
    static constexpr std::array<uint8_t, 8> MASKS = {
        0b0000'0000,
        0b0000'0001,
        0b0000'0011,
        0b0000'0111,
        0b0000'1111,
        0b0001'1111,
        0b0011'1111,
        0b0111'1111,
    };
    rank += std::popcount<uint8_t>(
        decompress_map_[compressed_leaves_[block_index * leaf_size +
                                           block_offset]] &
        MASKS[bit_index % 8]);
    return rank;
  }

  /// @brief Counts the number of 0-bits up to (and excluding) an index.
  size_t rank0(const size_type bit_index) const {
    return bit_index - rank1(bit_index);
  }

  int64_t print_space_usage() {
    int64_t space_usage = sizeof(tau_) + sizeof(max_leaf_length_) + sizeof(s_) +
                          sizeof(leaf_size);
    for (auto bv : block_tree_types_) {
      space_usage += bv->size() / 8;
    }
    for (auto rs : block_tree_types_rs_) {
      space_usage += rs->space_usage();
    }
    for (const auto iv : block_tree_pointers_) {
      space_usage += (int64_t)sdsl::size_in_bytes(*iv);
    }
    for (const auto iv : block_tree_offsets_) {
      space_usage += (int64_t)sdsl::size_in_bytes(*iv);
    }
    if (rank_support) {
      for (auto c : chars_) {
        int64_t sum = 0;
        for (auto lvl : pointer_c_ranks_[chars_index_[c]]) {
          sum += sdsl::size_in_bytes(lvl);
        }
        for (auto lvl : c_ranks_[chars_index_[c]]) {
          sum += sdsl::size_in_bytes(lvl);
        }
        space_usage += sum;
      }
    }

    for (auto v : block_size_lvl_) {
      space_usage += sizeof(v);
    }
    for (auto v : block_per_lvl_) {
      space_usage += sizeof(v);
    }
    // space_usage += leaves_.size() * sizeof(uint8_t);
    space_usage += sdsl::size_in_bytes(compressed_leaves_);
    space_usage += compress_map_.size();

    return space_usage;
  };

  int32_t add_bit_rank_support() {
    rank_support = true;

    // Resize rank information vectors
    one_ranks_.resize(height(), sdsl::int_vector<0>());
    for (uint64_t level = 0; level < height(); level++) {
      one_ranks_[level].resize(block_tree_types_[level]->size());
    }
    pointer_prefix_one_counts_.resize(height(), sdsl::int_vector<0>());
    for (uint64_t level = 0; level < height(); level++) {
      pointer_prefix_one_counts_[level].resize(
          block_tree_pointers_[level]->size());
    }

    for (size_t block = 0; block < block_tree_types_[0]->size(); block++) {
      bit_rank_block(0, block);
    }

    for (size_t block = 1; block < block_tree_types_[0]->size(); block++) {
      one_ranks_[0][block] += one_ranks_[0][block - 1];
    }

    for (size_t level = 1; level < height(); level++) {
      size_type counter = tau_;
      size_t acc = 0;
      for (size_t block = 0; block < one_ranks_[level].size(); block++) {
        const size_type ones_in_block = one_ranks_[level][block];
        acc += ones_in_block;
        one_ranks_[level][block] = acc;
        --counter;
        if (counter == 0) {
          acc = 0;
          counter = tau_;
        }
      }
    }
    for (auto& prefix_one_counts : pointer_prefix_one_counts_) {
      sdsl::util::bit_compress(prefix_one_counts);
    }
    for (auto& ranks : one_ranks_) {
      sdsl::util::bit_compress(ranks);
    }
    return 0;
  }

  int32_t add_rank_support() {
    rank_support = true;
    c_ranks_.resize(chars_.size(), std::vector<sdsl::int_vector<0>>());
    pointer_c_ranks_.resize(chars_.size(), std::vector<sdsl::int_vector<0>>());
    for (uint64_t i = 0; i < c_ranks_.size(); i++) {
      c_ranks_[i].resize(height(), sdsl::int_vector<0>());
      for (uint64_t j = 0; j < c_ranks_[i].size(); j++) {
        c_ranks_[i][j].resize(block_tree_types_[j]->size());
      }
    }
    for (uint64_t i = 0; i < pointer_c_ranks_.size(); i++) {
      pointer_c_ranks_[i].resize(block_tree_pointers_.size(),
                                 sdsl::int_vector<0>());
      for (uint64_t j = 0; j < pointer_c_ranks_[i].size(); j++) {
        pointer_c_ranks_[i][j].resize(block_tree_pointers_[j]->size());
      }
    }
    for (auto c : chars_) {
      for (uint64_t i = 0; i < block_tree_types_[0]->size(); i++) {
        rank_block(c, 0, i);
      }
      size_type max = 0;
      for (uint64_t i = 1; i < block_tree_types_[0]->size(); i++) {
        c_ranks_[chars_index_[c]][0][i] += c_ranks_[chars_index_[c]][0][i - 1];
        if (c_ranks_[chars_index_[c]][0][i] > static_cast<uint64_t>(max)) {
          max = c_ranks_[chars_index_[c]][0][i];
        }
      }
      for (uint64_t i = 1; i < height(); i++) {
        size_type counter = tau_;
        size_type acc = 0;
        for (uint64_t j = 0; j < block_tree_types_[i]->size(); j++) {
          size_type temp = c_ranks_[chars_index_[c]][i][j];
          c_ranks_[chars_index_[c]][i][j] += acc;
          acc += temp;
          counter--;
          if (counter == 0) {
            acc = 0;
            counter = tau_;
          }
        }
      }
      for (uint64_t i = 0; i < pointer_c_ranks_[chars_index_[c]].size(); i++) {
        sdsl::util::bit_compress(pointer_c_ranks_[chars_index_[c]][i]);
      }
      for (uint64_t i = 0; i < c_ranks_[chars_index_[c]].size(); i++) {
        sdsl::util::bit_compress(c_ranks_[chars_index_[c]][i]);
      }
    }
    return 0;
  }

  int32_t add_rank_support_omp(int32_t threads) {
    rank_support = true;
    c_ranks_.resize(chars_.size(), std::vector<sdsl::int_vector<0>>());
    pointer_c_ranks_.resize(chars_.size(), std::vector<sdsl::int_vector<0>>());
    for (uint64_t i = 0; i < c_ranks_.size(); i++) {
      c_ranks_[i].resize(height(), sdsl::int_vector<0>());
      for (uint64_t j = 0; j < c_ranks_[i].size(); j++) {
        c_ranks_[i][j].resize(block_tree_types_[j]->size());
      }
    }
    for (uint64_t i = 0; i < pointer_c_ranks_.size(); i++) {
      pointer_c_ranks_[i].resize(block_tree_pointers_.size(),
                                 sdsl::int_vector<0>());
      for (uint64_t j = 0; j < pointer_c_ranks_[i].size(); j++) {
        pointer_c_ranks_[i][j].resize(block_tree_pointers_[j]->size());
      }
    }
    omp_set_num_threads(threads);

#pragma omp parallel for default(none)
    for (auto c : chars_) {
      for (uint64_t i = 0; i < block_tree_types_[0]->size(); i++) {
        rank_block(c, 0, i);
      }
      size_type max = 0;
      for (uint64_t i = 1; i < block_tree_types_[0]->size(); i++) {
        c_ranks_[chars_index_[c]][0][i] += c_ranks_[chars_index_[c]][0][i - 1];
        if (c_ranks_[chars_index_[c]][0][i] > static_cast<uint64_t>(max)) {
          max = c_ranks_[chars_index_[c]][0][i];
        }
      }
      for (uint64_t i = 1; i < height(); i++) {
        size_type counter = tau_;
        size_type acc = 0;
        for (uint64_t j = 0; j < block_tree_types_[i]->size(); j++) {
          size_type temp = c_ranks_[chars_index_[c]][i][j];
          c_ranks_[chars_index_[c]][i][j] += acc;
          acc += temp;
          counter--;
          if (counter == 0) {
            acc = 0;
            counter = tau_;
          }
        }
      }
      for (uint64_t i = 0; i < pointer_c_ranks_[chars_index_[c]].size(); i++) {
        sdsl::util::bit_compress(pointer_c_ranks_[chars_index_[c]][i]);
      }
      for (uint64_t i = 0; i < c_ranks_[chars_index_[c]].size(); i++) {
        sdsl::util::bit_compress(c_ranks_[chars_index_[c]][i]);
      }
    }
    return 0;
  }

protected:
  void compress_leaves() {
    // Holds a 1 on every char that exists
    compress_map_.resize(256, 0);
    decompress_map_.resize(256, 0);
    for (size_t i = 0; i < this->leaves_.size(); ++i) {
      compress_map_[this->leaves_[i]] = 1;
    }
    for (size_t c = 0, cur_val = 0; c < this->compress_map_.size(); ++c) {
      const size_t tmp = compress_map_[c];
      compress_map_[c] = cur_val;
      decompress_map_[cur_val] = c;
      cur_val += tmp;
    }

    compressed_leaves_.resize(this->leaves_.size());
    for (size_t i = 0; i < this->leaves_.size(); ++i) {
      compressed_leaves_[i] = compress_map_[this->leaves_[i]];
    }
    sdsl::util::bit_compress(this->compressed_leaves_);
    leaves_.resize(0);
    leaves_.shrink_to_fit();
  }
  /// @brief Calculate the number of leading zeros for a 32-bit integer.
  /// This value is capped at 31.
  static size_type leading_zeros(const int32_t val) {
    return __builtin_clz(static_cast<unsigned int>(val) | 1);
  }

  /// @brief Calculate the number of leading zeros for a 64-bit integer.
  /// This value is capped at 64.
  static size_type leading_zeros(const int64_t val) {
    return __builtin_clzll(static_cast<unsigned long long>(val) | 1);
  }

  ///
  /// @brief Determine the padding and minimum height and the size of the blocks
  /// on the top level of a block tree with s top-level blocks and an arity of
  /// tau with leaves also of size tau.
  ///
  /// The height is the number of levels in the tree.
  /// The padding is the number of characters that the top-level exceeds the
  /// text length. For example, if the result was that the top level consists of
  /// s = 5 blocks of size 30 and the text size being 80, then the padding would
  /// be (5 * 30) - 80 = 70.
  ///
  /// @param[out] padding The number of characters in the last block (of the
  ///   first level of the tree) that are empty.
  /// @param[in] text_length The number of characters in the input string.
  /// @param[out] height The number of levels in the tree.
  /// @param[out] blk_size The size of blocks on the first level of the tree.
  ///
  void calculate_padding(int64_t& padding,
                         int64_t text_length,
                         int64_t& height,
                         int64_t& blk_size) {
    // This is the number of characters occupied by a tree with s*tau^h levels
    // and leaves of size tau. At the start, we only have a tree with the first
    // level with s leaf blocks which each have size tau. If we insert another
    // level, the number of leaf blocks (and therefore the number of occupied
    // characters) increases by a factor of tau.
    int64_t tmp_padding = this->s_ * this->tau_;
    int64_t h = 1;
    // Size of the blocks on the current level (starting at the leaf level)
    blk_size = tau_;
    // While the tree does not cover the entire text, add a level
    while (tmp_padding < text_length) {
      tmp_padding *= this->tau_;
      blk_size *= this->tau_;
      h++;
    }
    // once the tree has enough levels to cover the entire text, we set the
    // tree's values
    height = h;
    // The padding is the number of excess characters that the block tree covers
    // over the length of the text.
    padding = tmp_padding - text_length;
  }

  size_type bit_rank_block(size_type level, size_type block_index) {
    const auto& is_internal = *block_tree_types_[level];
    const auto& is_internal_rank = *block_tree_types_rs_[level];
    if (static_cast<uint64_t>(block_index) >= is_internal.size()) {
      return 0;
    }

    size_type num_ones = 0;
    if (is_internal[block_index]) {
      const size_type internal_index = is_internal_rank.rank1(block_index);
      if (static_cast<uint64_t>(level) < height() - 1) {
        // If we are not on the last level recursively call
        for (size_type k = 0; k < tau_; ++k) {
          num_ones += bit_rank_block(level + 1, internal_index * tau_ + k);
        }
      } else {
        // If we are on the last level
        for (size_type k = 0; k < tau_; ++k) {
          num_ones += bit_rank_leaf(internal_index * tau_ + k, leaf_size);
        }
      }
    } else {
      // TODO: Handle the case where blocks are not internal
      const size_type back_block_index = is_internal_rank.rank0(block_index);
      const size_type ptr = (*block_tree_pointers_[level])[back_block_index];
      const size_type off = (*block_tree_offsets_[level])[back_block_index];
      size_type num_ones_parts = 0;
      num_ones += one_ranks_[level][ptr];
      if (off > 0) {
        num_ones_parts = part_bit_rank_block(level, ptr, off);
        const size_type num_ones_2nd_part =
            part_bit_rank_block(level, ptr + 1, off);
        num_ones -= num_ones_parts;
        num_ones += num_ones_2nd_part;
      }
      pointer_prefix_one_counts_[level][back_block_index] = num_ones_parts;
    }
    one_ranks_[level][block_index] = num_ones;
    return num_ones;
  }

  ///
  /// @brief Generates rank information for a block and all its children
  /// recursively.
  ///
  /// @param c The character to generate rank information for.
  /// @param level The level index the block is on.
  /// @param block_index The index of the block on this level
  size_type rank_block(uint8_t c, size_type level, size_type block_index) {
    if (static_cast<uint64_t>(block_index) >=
        block_tree_types_[level]->size()) {
      return 0;
    }
    size_type rank_c = 0;
    if ((*block_tree_types_[level])[block_index] == 1) {
      if (static_cast<uint64_t>(level) != height() - 1) {
        size_type rank_blk = block_tree_types_rs_[level]->rank1(block_index);
        for (size_type k = 0; k < tau_; k++) {
          rank_c += rank_block(c, level + 1, rank_blk * tau_ + k);
        }
      } else {
        size_type rank_blk = block_tree_types_rs_[level]->rank1(block_index);
        for (size_type k = 0; k < tau_; k++) {
          rank_c += rank_leaf(c, rank_blk * tau_ + k, leaf_size);
        }
      }
    } else {
      size_type rank_0 = block_tree_types_rs_[level]->rank0(block_index);
      size_type ptr = (*block_tree_pointers_[level])[rank_0];
      size_type off = (*block_tree_offsets_[level])[rank_0];
      size_type rank_g = 0;
      rank_c += c_ranks_[chars_index_[c]][level][ptr];
      if (off != 0) {
        rank_g = part_rank_block(c, level, ptr, off);
        size_type rank_2nd = part_rank_block(c, level, ptr + 1, off);
        rank_c -= rank_g;
        rank_c += rank_2nd;
      }
      pointer_c_ranks_[chars_index_[c]][level][rank_0] = rank_g;
    }
    c_ranks_[chars_index_[c]][level][block_index] = rank_c;
    return rank_c;
  }

  size_type part_bit_rank_block(const size_type level,
                                const size_type block_index,
                                const size_type chars_to_process) {
    // FIXME: Seems to be kinda broken. Doesn't seem to report all bits it needs
    const auto& is_internal = *block_tree_types_[level];
    const auto& is_internal_rank = *block_tree_types_rs_[level];
    if (static_cast<uint64_t>(block_index) >= is_internal.size()) {
      return 0;
    }

    size_type num_ones = 0;
    if (is_internal[block_index]) {
      const size_type internal_index = is_internal_rank.rank1(block_index);
      size_type k = 0;
      size_type processed_chars = 0;
      if (static_cast<size_t>(level) < height() - 1) {
        const size_type child_size = block_size_lvl_[level + 1];
        // We're not on the last level
        // iterate over the children as long as we don't exceed the limit
        for (k = 0;
             k < tau_ && processed_chars + child_size <= chars_to_process;
             ++k) {
          num_ones += one_ranks_[level + 1][internal_index * tau_ + k];
          processed_chars += child_size;
        }

        // If we still need to process more chars and they end inside the next
        // child, rank that part of the next child
        if (processed_chars != chars_to_process) {
          num_ones += part_bit_rank_block(level + 1,
                                          internal_index * tau_ + k,
                                          chars_to_process - processed_chars);
        }
      } else {
        // We're on the last level
        for (k = 0; k < tau_ && processed_chars + leaf_size <= chars_to_process;
             k++) {
          num_ones += bit_rank_leaf(internal_index * tau_ + k, leaf_size);
          processed_chars += leaf_size;
        }

        if (processed_chars != chars_to_process) {
          num_ones += bit_rank_leaf(internal_index * tau_ + k,
                                    chars_to_process % leaf_size);
        }
      }
    } else {
      const size_type back_block_index = is_internal_rank.rank0(block_index);
      const size_type ptr = (*block_tree_pointers_[level])[back_block_index];
      const size_type off = (*block_tree_offsets_[level])[back_block_index];

      // If we need to process chars beyond this block, we need to
      if (chars_to_process + off >= block_size_lvl_[level]) {
        // Ones in the entire block this block points to
        num_ones += one_ranks_[level][ptr];
        // Ones that overflow into the next block
        num_ones += part_bit_rank_block(level,
                                        ptr + 1,
                                        chars_to_process + off -
                                            block_size_lvl_[level]);
        // Num ones in the pointed-to block *before* the pointed-to area
        num_ones -= pointer_prefix_one_counts_[level][back_block_index];
      } else {
        // Number of ones up to the cutoff point
        num_ones += part_bit_rank_block(level, ptr, chars_to_process + off);
        // Num ones in the pointed-to block *before* the pointed-to area
        num_ones -= pointer_prefix_one_counts_[level][back_block_index];
      }
    }
    return num_ones;
  }

  size_type part_rank_block(uint8_t c,
                            size_type level,
                            size_type block_index,
                            size_type g) {
    if (static_cast<uint64_t>(block_index) >=
        block_tree_types_[level]->size()) {
      return 0;
    }
    size_type rank_c = 0;
    if ((*block_tree_types_[level])[block_index] == 1) {
      if (static_cast<uint64_t>(level) != height() - 1) {
        size_type rank_blk = block_tree_types_rs_[level]->rank1(block_index);
        size_type k = 0;
        size_type k_sum = 0;
        for (k = 0; k < tau_ && k_sum + block_size_lvl_[level + 1] <= g; k++) {
          rank_c += c_ranks_[chars_index_[c]][level + 1][rank_blk * tau_ + k];
          k_sum += block_size_lvl_[level + 1];
        }

        if (k_sum != g) {
          rank_c +=
              part_rank_block(c, level + 1, rank_blk * tau_ + k, g - k_sum);
        }
      } else {
        size_type rank_blk = block_tree_types_rs_[level]->rank1(block_index);
        size_type k = 0;
        size_type k_sum = 0;
        for (k = 0; k < tau_ && k_sum + leaf_size <= g; k++) {
          rank_c += rank_leaf(c, rank_blk * tau_ + k, leaf_size);
          k_sum += leaf_size;
        }

        if (k_sum != g) {
          rank_c += rank_leaf(c, rank_blk * tau_ + k, g % leaf_size);
        }
      }
    } else {
      size_type rank_0 = block_tree_types_rs_[level]->rank0(block_index);
      size_type ptr = (*block_tree_pointers_[level])[rank_0];
      size_type off = (*block_tree_offsets_[level])[rank_0];
      if (g + off >= block_size_lvl_[level]) {
        rank_c += c_ranks_[chars_index_[c]][level][ptr] -
                  pointer_c_ranks_[chars_index_[c]][level][rank_0] +
                  part_rank_block(c,
                                  level,
                                  ptr + 1,
                                  g + off - block_size_lvl_[level]);
      } else {
        rank_c += part_rank_block(c, level, ptr, g + off) -
                  pointer_c_ranks_[chars_index_[c]][level][rank_0];
      }
    }
    return rank_c;
  }

  ///
  /// @brief Count ones in leaf block.
  ///
  /// @param leaf_index The index of the leaf block.
  /// @param max_char_index The maximum character index (exclusive) to
  /// consider. This is used for when this block is at the end of the string.
  /// @return The number of ones in this block.
  ///
  size_type bit_rank_leaf(size_type leaf_index, size_type max_char_index) {
    if (static_cast<uint64_t>(leaf_index * leaf_size) >=
        compressed_leaves_.size()) {
      return 0;
    }

    size_type result = 0;
    for (size_type i = 0; i < max_char_index; ++i) {
      const uint8_t byte =
          decompress_map_[compressed_leaves_[leaf_index * leaf_size + i]];
      result += std::popcount(byte);
    }
    return result;
  }

  size_type rank_leaf(uint8_t c, size_type leaf_index, size_type i) {
    if (static_cast<uint64_t>(leaf_index * leaf_size) >=
        compressed_leaves_.size()) {
      return 0;
    }
    //        size_type x = leaves_.size() - leaf_index * this->tau_;
    //        i = std::min(i, x);
    size_type result = 0;
    for (size_type ind = 0; ind < i; ind++) {
      if (compressed_leaves_[leaf_index * leaf_size + ind] ==
          compress_map_[c]) {
        result++;
      }
    }
    return result;
  }

  size_type
  find_next_smallest_index_binary_search(size_type i,
                                         std::vector<int64_t>& pVector) {
    int64_t l = 0;
    int64_t r = pVector.size();
    while (l < r) {
      int64_t m = std::floor((l + r) / 2);
      if (i < pVector[m]) {
        r = m;
      } else {
        l = m + 1;
      }
    }
    return r - 1;
  };
  int64_t
  find_next_smallest_index_linear_scan(size_type i,
                                       std::vector<size_type>& pVector) {
    int64_t b = 0;
    while (b < pVector.size() && i >= pVector[b]) {
      b++;
    }
    return b - 1;
  };
  size_type find_next_smallest_index_block_tree(size_type index) {
    size_type block_size = this->block_size_lvl_[0];
    size_type blk_pointer = index / block_size;
    size_type off = index % block_size;
    size_type child = 0;
    for (size_type i = 0; i < this->height(); i++) {
      if ((*this->block_tree_types_[i])[blk_pointer] == 0) {
        return -1;
      }
      if (off > 0 && (*this->block_tree_types_[i])[blk_pointer + 1] == 0) {
        return -1;
      }
      size_type rank_blk = this->block_tree_types_rs_[i]->rank1(blk_pointer);
      blk_pointer = rank_blk * this->tau_;
      block_size /= this->tau_;
      child = off / block_size;
      off = off % block_size;
      blk_pointer += child;
    }
    return blk_pointer;
  };
};

} // namespace pasta

/******************************************************************************/
