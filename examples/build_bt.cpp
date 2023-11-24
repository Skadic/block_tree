/*******************************************************************************
 * This file is part of pasta::block_tree
 *
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

#include "pasta/block_tree/utils/MersenneHash.hpp"

#include <bitset>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <pasta/block_tree/bit_block_tree.hpp>
#include <pasta/block_tree/block_tree.hpp>
#include <pasta/block_tree/construction/bit_block_tree_sharded.hpp>
#include <syncstream>

#define PAR_SHARDED_SYNC_SMALL
#ifdef FP
#  include <pasta/block_tree/construction/block_tree_fp.hpp>
std::unique_ptr<pasta::BlockTreeFP<uint8_t, int32_t>>
make_bt(std::vector<uint8_t>& text,
        const size_t arity,
        const size_t leaf_length,
        const size_t,
        const size_t) {
  ;
  return std::unique_ptr<pasta::BlockTreeFP<uint8_t, int32_t>>(
      pasta::make_block_tree_fp<uint8_t, int32_t>(text, arity, leaf_length));
}
#  define ALGO_NAME "fp"
#elif defined FP2
#  include <pasta/block_tree/construction/block_tree_fp2_seq.hpp>
std::unique_ptr<pasta::BlockTreeFP2<uint8_t, int32_t>>
make_bt(std::vector<uint8_t>& text,
        const size_t arity,
        const size_t leaf_length,
        const size_t,
        const size_t) {
  ;
  return std::make_unique<pasta::BlockTreeFP2<uint8_t, int32_t>>(text,
                                                                 arity,
                                                                 1,
                                                                 leaf_length);
}
#  define ALGO_NAME "fp2"
#elif defined LPF
#  include <pasta/block_tree/construction/block_tree_lpf.hpp>
std::unique_ptr<pasta::BlockTreeLPF<uint8_t, int32_t>>
make_bt(std::vector<uint8_t>& text,
        const size_t arity,
        const size_t leaf_length,
        const size_t threads,
        const size_t) {
  ;
  return std::unique_ptr<pasta::BlockTreeLPF<uint8_t, int32_t>>(
      pasta::make_block_tree_lpf_parallel<uint8_t, int32_t>(text,
                                                            arity,
                                                            leaf_length,
                                                            true,
                                                            threads));
}
#  define ALGO_NAME "lpf"
#elif defined PAR_SHARDED
#  include <pasta/block_tree/construction/block_tree_fp_par_sharded.hpp>
std::unique_ptr<pasta::BlockTreeFPParSharded<uint8_t, int32_t>>
make_bt(std::vector<uint8_t>& text,
        const size_t arity,
        const size_t leaf_length,
        const size_t threads,
        const size_t) {
  ;
  return std::make_unique<pasta::BlockTreeFPParSharded<uint8_t, int32_t>>(
      text,
      arity,
      1,
      leaf_length,
      threads);
}
#  define ALGO_NAME "shard"
#elif defined PAR_SHARDED_SYNC
#  include <pasta/block_tree/construction/block_tree_fp_par_sync_sharded.hpp>
std::unique_ptr<pasta::BlockTreeFPParShardedSync<uint8_t, int32_t>>
make_bt(std::vector<uint8_t>& text,
        const size_t arity,
        const size_t leaf_length,
        const size_t threads,
        const size_t queue_size) {
  ;
  return std::make_unique<pasta::BlockTreeFPParShardedSync<uint8_t, int32_t>>(
      text,
      arity,
      1,
      leaf_length,
      threads,
      queue_size);
}
#  define ALGO_NAME "shard_sync"
#elif defined PAR_SHARDED_SYNC_SMALL
#  include <pasta/block_tree/construction/block_tree_fp_par_sync_sharded_small.hpp>
std::unique_ptr<pasta::BlockTreeFPParShardedSyncSmall<uint8_t, int32_t>>
make_bt(std::vector<uint8_t>& text,
        const size_t arity,
        const size_t leaf_length,
        const size_t threads,
        const size_t queue_size) {
  return std::make_unique<
      pasta::BlockTreeFPParShardedSyncSmall<uint8_t, int32_t>>(text,
                                                               arity,
                                                               1,
                                                               leaf_length,
                                                               threads,
                                                               queue_size);
}
#  define ALGO_NAME "shard_sync_small"
#elif defined PAR_PHMAP
#  include <pasta/block_tree/construction/block_tree_fp_par_phmap.hpp>
std::unique_ptr<pasta::BlockTreeFPParPH<uint8_t, int32_t>>
make_bt(std::vector<uint8_t>& text,
        const size_t arity,
        const size_t leaf_length,
        const size_t threads,
        const size_t) {
  ;
  return std::make_unique<pasta::BlockTreeFPParPH<uint8_t, int32_t>>(
      text,
      arity,
      1,
      leaf_length,
      threads);
}
#  define ALGO_NAME "par_map"
#elif defined PAR_PARLAY
#  include <pasta/block_tree/construction/block_tree_fp_par_parlay.hpp>
std::unique_ptr<pasta::BlockTreeFPParParlay<uint8_t, int32_t>>
make_bt(std::vector<uint8_t>& text,
        const size_t arity,
        const size_t leaf_length,
        const size_t threads,
        const size_t) {
  ;
  return std::make_unique<pasta::BlockTreeFPParParlay<uint8_t, int32_t>>(
      text,
      arity,
      1,
      leaf_length,
      threads);
}
#  define ALGO_NAME "par_parlay"
#endif

#if defined PAR_SHARDED_SYNC || defined PAR_SHARDED_SYNC_SMALL
#  define USES_QUEUE true
#  define IS_PARALLEL true
#else
#  define USES_QUEUE false
#endif

#ifndef IS_PARALLEL
#  if defined PAR_SHARDED_SYNC_SMALL || defined PAR_SHARDED_SYNC ||            \
      defined PAR_SHARDED || defined PAR_PHMAP || defined LPF ||               \
      defined PAR_PHMAP
#    define IS_PARALLEL true
#  else
#    define IS_PARALLEL false
#  endif
#endif

#define BIT_BT

#include <sstream>
#include <string>
#include <tlx/cmdline_parser.hpp>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using Duration = Clock::duration;

int main(int argc, char** argv) {
  using namespace pasta;

  tlx::CmdlineParser cp;
  cp.set_description(
      "Build a block tree for a given input text or bit vector.");

  std::string file;
  cp.add_param_string("file",
                      file,
                      "The path to the file which to build a block tree from");

  size_t arity = 0;
  cp.add_param_size_t("arity", arity, "The arity of the block tree");
  size_t leaf_length = 0;
  cp.add_param_size_t(
      "leaf",
      leaf_length,
      "The maximum number of characters saved verbatim per leaf block.");

  size_t threads = 1;
  cp.add_size_t('t',
                "threads",
                threads,
                "The number of threads to use for parallel algorithms (ignored "
                "for sequential algorithms)");
  size_t queue_size = 1024;
  cp.add_size_t(
      'q',
      "queue_size",
      queue_size,
      "The size of each thread's queue used for sharded hash map algorithms");

  bool make_bv = false;
  cp.add_bool('b',
              "bitvec",
              make_bv,
              "Whether to interpret the input as a bitvector. If \"-o\" is not "
              "used, each byte of the input file will be interpreted as 8 bits "
              "of the bit vector respectively. In each byte, the least "
              "significant bit is index 0.");

  std::string one_chars = "";
  cp.add_string(
      'o',
      "one_chars",
      one_chars,
      "A string to be used with the \"-b\" flag. If used, each character of "
      "the input represents one bit of the bit vector. It will be a 1 if this "
      "parameter string contains the respective character, 0 otherwise.");

  if (!cp.process(argc, argv)) {
    return 1;
  }

#ifdef BT_DBG
  std::cout << "building block tree with parameters:"
            << "\narity: " << arity << "\nmax leaf length: " << leaf_length
            << "\nsaving to " << out_path << "\nusing " << threads << " threads"
            << std::endl;
#endif

  pasta::BitVector bv;
  std::vector<uint8_t> text;
  {
    std::string input;
    std::ifstream t(argv[1]);
    std::stringstream buffer;
    buffer << t.rdbuf();
    input = buffer.str();
    if (make_bv) {
      if (one_chars.empty()) {
        // Interpret each character as 8 bits
        new (&bv) pasta::BitVector(input.size() * 8);
        std::span<std::byte> bytes = std::as_writable_bytes(bv.data());
        for (size_t i = 0; i < input.size(); ++i) {
          bytes[i] = std::byte{static_cast<uint8_t>(input[i])};
        }
      } else {
        // Interpret each character as a bit
        new (&bv) pasta::BitVector(input.size());
        std::array<bool, 256> is_one{};
        for (char c : one_chars) {
          is_one[static_cast<uint8_t>(c)] = true;
        }
        for (size_t i = 0; i < input.size(); ++i) {
          bv[i] = is_one[static_cast<uint8_t>(input[i])];
        }
      }
    } else {
      text = std::vector<uint8_t>(input.begin(), input.end());
    }
  }

  std::cout << "RESULT algo=" << ALGO_NAME
            << " file=" << std::filesystem::path(file).filename().string()
            << " threads=" << threads << " arity=" << arity
            << " leaf_length=" << leaf_length;
  if (make_bv) {
    std::cout << " bv_size=" << bv.size();
  } else {
    std::cout << " file_size=" << text.size();
  }
  TimePoint now = Clock::now();

  if (make_bv) {
    // Make bit vector block tree
    auto bt = std::make_unique<BitBlockTreeSharded<int32_t>>(bv,
                                                             arity,
                                                             1,
                                                             leaf_length,
                                                             threads,
                                                             queue_size);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       Clock::now() - now)
                       .count();
    std::cout << " time=" << elapsed << " space=" << bt->print_space_usage();
    std::cout << std::endl;

#if defined BT_INSTRUMENT && defined BT_DBG
    pasta::print_hash_data();
#endif

#pragma omp parallel for
    for (size_t i = 0; i < bv.size(); ++i) {
      const bool c = bt->access(i);
      if (c != bv[i]) {
        std::osyncstream(std::cerr)
            << "Error at position " << i << "\nExpected: " << std::boolalpha
            << bv[i] << "\nActual: " << c << std::noboolalpha << std::endl;
        exit(1);
      }
    }

    bt->add_bit_rank_support();
    size_t num_ones = 0;
    for (size_t i = 0; i < bv.size(); i++) {
      const size_t rank = bt->rank1(i);
      if (num_ones != rank) {
        std::osyncstream(std::cerr)
            << "Error at position " << i << "\nExpected: " << num_ones
            << "\nActual: " << rank << std::endl;
        throw std::runtime_error("oof");
      }

      num_ones += bv[i];
    }

  } else {
    // Make text block tree
    auto bt = make_bt(text, arity, leaf_length, threads, queue_size);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       Clock::now() - now)
                       .count();

    std::cout << " time=" << elapsed << " space=" << bt->print_space_usage();
    std::cout << std::endl;

#if defined BT_INSTRUMENT && defined BT_DBG
    pasta::print_hash_data();
#endif

#pragma omp parallel for
    for (size_t i = 0; i < text.size(); ++i) {
      const auto c = bt->access(i);
      if (c != text[i]) {
        std::osyncstream(std::cerr)
            << "Error at position " << i
            << "\nExpected: " << static_cast<char>(text[i])
            << "\nActual: " << static_cast<char>(c) << std::endl;
        exit(1);
      }
    }
  }

  return 0;
}

/******************************************************************************/
