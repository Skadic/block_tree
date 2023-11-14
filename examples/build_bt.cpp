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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <pasta/block_tree/block_tree.hpp>
#include <syncstream>

#define PAR_SHARDED_SYNC
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
#elif defined PAR_PHF
#  include <pasta/block_tree/construction/block_tree_fp_par_phf.hpp>
std::unique_ptr<pasta::BlockTreeFPParPHF<uint8_t, int32_t>>
make_bt(std::vector<uint8_t>& text,
        const size_t arity,
        const size_t leaf_length,
        const size_t threads,
        const size_t) {
  ;
  return std::make_unique<pasta::BlockTreeFPParPHF<uint8_t, int32_t>>(
      text,
      arity,
      1,
      leaf_length,
      threads);
}
#  define ALGO_NAME "par_phf"
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

#include <sstream>
#include <string>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using Duration = Clock::duration;

int main(int argc, char** argv) {
  using namespace pasta;
  if (argc < 2) {
    std::cerr << "Please input file" << std::endl;
    exit(1);
  }

  if (!std::filesystem::exists(argv[1])) {
    std::cerr << "File " << argv[1] << " does not exist" << std::endl;
    exit(1);
  }

  if (argc < 3) {
    std::cerr << "Please input tree arity (tau)" << std::endl;
    exit(1);
  }

  const size_t arity = atoi(argv[2]);

  if (argc < 4) {
    std::cerr << "Please input max leaf length" << std::endl;
    exit(1);
  }

  const size_t leaf_length = atoi(argv[3]);

#if IS_PARALLEL
  if (argc < 5) {
    std::cerr << "Please input number of threads (ignored if single threaded "
                 "algorithm)"
              << std::endl;
    exit(1);
  }

  const size_t threads = atoi(argv[4]);
#else
  const size_t threads = 1;
#endif

#if USES_QUEUE
  if (argc < 6) {
    std::cerr << "Please input queue size" << std::endl;
    exit(1);
  }
  const size_t queue_size = atoi(argv[5]);
#else
  const size_t queue_size = 0;
#endif

  std::stringstream ss;
  ss << argv[1] << "_arit" << arity << "_leaf" << leaf_length << "_new.bt";
  std::string out_path = ss.str();

#ifdef BT_DBG
  std::cout << "building block tree with parameters:"
            << "\narity: " << arity << "\nmax leaf length: " << leaf_length
            << "\nsaving to " << out_path << "\nusing " << threads << " threads"
            << std::endl;
#endif

  std::vector<uint8_t> text;
  {
    std::string input;
    std::ifstream t(argv[1]);
    std::stringstream buffer;
    buffer << t.rdbuf();
    input = buffer.str();
    text = std::vector<uint8_t>(input.begin(), input.end());
  }

  std::cout << "RESULT algo=" << ALGO_NAME
            << " file=" << std::filesystem::path(argv[1]).filename().string()
            << " threads=" << threads << " arity=" << arity
            << " leaf_length=" << leaf_length;
  TimePoint now = Clock::now();
  auto bt = make_bt(text, arity, leaf_length, threads, queue_size);
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - now)
          .count();

  std::cout << " time=" << elapsed << " space=" << bt->print_space_usage();

  std::cout << std::endl;

#ifdef BT_INSTRUMENT
  std::cout << "comparisons: " << mersenne_hash_comparisons
            << ", equals: " << mersenne_hash_equals
            << ", collisions: " << mersenne_hash_collisions
            << ", percent equals: "
            << 100 * mersenne_hash_equals / ((double)mersenne_hash_comparisons)
            << ", percent collisions: "
            << 100 * mersenne_hash_collisions /
                   ((double)mersenne_hash_comparisons)
            << std::endl;
#endif

  // std::ofstream ot(out_path);
  //  bt->serialize(ot);
#pragma omp parallel for
  for (size_t i = 0; i < text.size(); ++i) {
    const auto c = bt->access(i);
    if (c != text[i]) {
      std::osyncstream(std::cerr)
          << "Error at position " << i << "\nExpected: " << (char)text[i]
          << "\nActual: " << (char)c << std::endl;
      exit(1);
    }
  }
  // ot.close();

  return 0;
}

/******************************************************************************/
