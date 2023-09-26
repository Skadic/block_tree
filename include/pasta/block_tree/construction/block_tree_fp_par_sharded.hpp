/*******************************************************************************
 * This file is part of pasta::block_tree
 *
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

#include "data-structures/hash_table_mods.hpp"
#include "pasta/bit_vector/bit_vector.hpp"
#include "pasta/block_tree/block_tree.hpp"
#include "pasta/block_tree/utils/MersenneHash.hpp"
#include "pasta/block_tree/utils/MersenneRabinKarp.hpp"
#include "pasta/block_tree/utils/mpsc_queue/jiffy.hpp"
#include "pasta/block_tree/utils/mpsc_queue/queue.hpp"
#include "pasta/block_tree/utils/mpsc_queue/stupid_queue.hpp"
#include "pasta/block_tree/utils/sharded_map.hpp"

#include <atomic>
#include <concepts>
#include <list>
#include <memory>
#include <mutex>
#include <omp.h>
#include <phmap.h>
#include <robin_hood.h>
#include <sdsl/int_vector.hpp>
#include <sdsl/util.hpp>

#define BT_NUM_THREADS 8
#define BT_FILL_THRESHOLD 0.5
#define BT_QUEUE_CAPACITY 1024

__extension__ typedef unsigned __int128 uint128_t;

namespace pasta {

/// @brief A parallel block tree construction algorithm using Rabin-Karp hashes
///   and a sharded hash map.
/// @tparam input_type The type of the characters in the input string
/// @tparam size_type The type used for indices etc. (must be a signed integer)
/// @tparam queue_type The type of queue to use for communication
///   in the sharded hash map.
template <std::integral input_type,
          std::signed_integral size_type,
          template <typename> typename queue_type = StupidQueue>
class BlockTreeFPParSharded : public BlockTree<input_type, size_type> {
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = Clock::time_point;

  /// @brief A marker for a block that has no earlier occurrence
  constexpr static size_type NO_EARLIER_OCC = -1;
  /// @brief A marker for a block that has been pruned
  constexpr static size_type PRUNED = -2;

  /// @brief Base of the polynomial used for the Rabin-Karp hasher
  constexpr static size_type SIGMA = 256;
  /// @brief A mersenne prime used for the Rabin-Karp hasher
  constexpr static uint128_t PRIME = 2305843009213693951ULL;
  /// @brief The exponent of the mersenne prime used for the Rabin-Karp hasher
  constexpr static uint8_t PRIME_EXPONENT = 61;

  /// @brief A bit vector
  using BitVector = pasta::BitVector;
  /// @brief A rank data structure for a bit vector
  using Rank = pasta::RankSelect<pasta::OptimizedFor::ONE_QUERIES>;

  /// @brief A concurrent queue for communication between threads in the
  ///   sharded hash map.
  template <typename elem_type>
    requires MpscQueue<queue_type<elem_type>, elem_type>
  using Queue = queue_type<elem_type>;

  /// @brief A sequential hash map used as backing for the sharded hash map.
  template <typename key_type, typename value_type>
  using SeqHashMap =
      robin_hood::unordered_node_map<key_type, value_type, std::hash<key_type>>;

  /// @brief A rabin karp hasher preconfigured for the current template
  ///   parameters
  using RabinKarp = MersenneRabinKarp<input_type, size_type, PRIME_EXPONENT>;
  /// @brief A rabin karp hash for the preconfigured rabin karp hasher
  using RabinKarpHash = MersenneHash<input_type>;

  /// @brief A hash map with rabin karp hashes as keys
  template <typename value_type,
            UpdateFunction<RabinKarpHash, value_type> update_fn_type>
  using RabinKarpMap =
      ShardedMap<RabinKarpHash, value_type, SeqHashMap, Queue, update_fn_type>;

public:
  size_t bp_hash_pairs_ns = 0;
  size_t bp_scan_pairs_ns = 0;
  size_t bp_markings_ns = 0;
  size_t bp_bitvec_ns = 0;

  size_t b_hash_blocks_ns = 0;
  size_t b_scan_blocks_ns = 0;
  size_t b_update_blocks_ns = 0;

private:
  /// @brief Contains data about a block tree level under construction
  struct LevelData {
    /// @brief Contains a 1 for each internal block (= block with children)
    ///   and a 0 for each block that has a back pointer
    std::unique_ptr<BitVector> is_internal;
    /// @brief Rank data structure for is_internal
    std::unique_ptr<Rank> is_internal_rank;
    /// @brief The block from which a back block is copying
    std::unique_ptr<std::vector<size_type>> pointers;
    /// @brief The offset into the block from which the back block is copying
    std::unique_ptr<std::vector<size_type>> offsets;
    /// @brief The number of back blocks pointing to the block
    std::unique_ptr<std::vector<size_type>> counters;
    /// @brief Block start indices
    std::unique_ptr<std::vector<size_type>> block_starts;
    /// @brief The block size on this level
    size_type block_size;
    /// @brief The index of the current level. First level is 0, second level is
    ///   1 etc.
    size_type level_index;
    /// @brief The number of blocks on the current level
    size_type num_blocks;

    inline LevelData(size_type level_index_,
                     size_type block_size_,
                     size_type num_blocks_)
        : is_internal(nullptr),
          is_internal_rank(nullptr),
          pointers(new std::vector<size_type>()),
          offsets(new std::vector<size_type>()),
          counters(new std::vector<size_type>()),
          block_starts(new std::vector<size_type>()),
          block_size(block_size_),
          level_index(level_index_),
          num_blocks(num_blocks_) {}

    /// @brief Checks whether a block is adjacent in the text
    ///   to its successor on this level
    [[nodiscard]] inline bool next_is_adjacent(size_t i) const {
      return (*block_starts)[i] + static_cast<size_type>(block_size) ==
             (*block_starts)[i + 1];
    }
  };

  /// @brief Contains data about the occurrences of a hashed block pair
  struct PairOccurrences {
    /// @brief The first block in the text in which the content appears
    size_type first_occ_block;
    /// @brief A list of block indices in which the content of the hashed block
    /// pair appears
    ///
    /// We're using an std::list here instead of an std::vector, since the
    /// reallocation upon insertion lead to issues during parallel access, when
    /// another thread tries to access the vector during reallocation.
    std::list<size_type> occurrences;

    /// @brief Initialize the occurrences of a hashed block pair.
    ///
    /// Note, that this only sets the first occurrence to the given block index,
    /// but does not add it to the occurrences list.
    /// @param first_occ_block_ The block index of the pair's first block.
    inline explicit PairOccurrences(size_type first_occ_block_)
        : first_occ_block(first_occ_block_),
          occurrences() {}

    /// @brief Add a block index to the occurrences.
    /// @param block_index The block index to add to the occurrences.
    inline void add_block_pair(size_type block_index) {
      occurrences.push_back(block_index);
    }

    /// @brief If the given block index is an earlier occurrence, update it
    /// @param block_index The block index of an occurrence
    inline void update(size_type block_index) {
      first_occ_block = std::min<size_type>(first_occ_block, block_index);
    }
  };

  /// @brief Contains data about the occurrences of a hashed block
  struct BlockOccurrences {
    /// @brief Represents the first occurrence of a block
    struct FirstOccurrence {
      /// @brief Block index of the first occurrence of the block's content
      size_type block;
      /// @brief The offset into the block at which that first occurrence occurs
      size_type offset;

      inline FirstOccurrence(size_type first_occ_block_,
                             size_type first_occ_offset_)
          : block(first_occ_block_),
            offset(first_occ_offset_) {}
    };

    // @brief The block index and offset of the first occurrence of this block's
    //   content
    std::atomic<FirstOccurrence> first_occ;

    /// @brief A list of block indices in which the content of the hashed block
    ///   occurs
    std::list<size_type> occurrences;

    /// @brief Initialize the occurrences of a hashed block.
    ///
    /// Note, that this only sets the first occurrence to the given block index,
    /// but does not add it to the occurrences list.
    /// @param first_occ_block_ The block index of the block's first occurrence.
    explicit BlockOccurrences(size_type first_occ_block_)
        : first_occ({first_occ_block_, 0}),
          occurrences() {}

    BlockOccurrences(const BlockOccurrences& other)
        : first_occ(other.first_occ.load()),
          occurrences(other.occurrences) {}

    BlockOccurrences(BlockOccurrences&& other) noexcept
        : first_occ(other.first_occ.load()),
          occurrences(std::move(other.occurrences)) {}

    /// @brief Add a block index to the occurrences.
    /// @param block_index The block index to add to the occurrences.
    inline void add_block(size_type block_index) {
      occurrences.push_back(block_index);
    }

    /// @brief If the given block index and offset are an earlier occurrence,
    ///   update them
    /// @param block_index The block index of an occurrence
    /// @param block_index The offset of that occurrence
    inline void update(size_type block_index, size_type block_offset) {
      FirstOccurrence prev_first_occ = this->first_occ.load();
      FirstOccurrence set(block_index, block_offset);
      while (block_index < prev_first_occ.block &&
             !first_occ.compare_exchange_weak(prev_first_occ, set)) {
      }
    }
  };

  /// @brief An update function for the sharded hash map that updates the
  ///   occurrences of a hashed block pair
  struct UpdatePairOccurrences {
    /// @brief The block index to add to the occurrences
    using InputValue = size_type;
    /// @brief Update the occurrences of a hashed block pair by adding the new
    ///   block index and updating the first occurrence if needed
    /// @param occurrences A reference to the occurrences in the map
    /// @param input_value The new block index to add to the occurrences
    inline static void update(RabinKarpHash&,
                              PairOccurrences& occurrences,
                              InputValue&& input_value) {
      occurrences.add_block_pair(input_value);
      occurrences.update(input_value);
    }

    /// @brief Initialize the occurrences of a hashed block pair
    /// @param input_value The block index of the pair's first block
    /// @return The initialized occurrences only containing the given block pair
    inline static PairOccurrences init(RabinKarpHash&,
                                       InputValue&& input_value) {
      PairOccurrences occurrences(input_value);
      occurrences.add_block_pair(input_value);
      occurrences.update(input_value);
      return occurrences;
    }
  };

  /// @brief An update function for the sharded hash map that updates the
  ///   occurrences of a hashed block
  struct UpdateBlockOccurrences {
    /// @brief A pair of the block index
    ///   and offset of the first occurrence of a block
    using InputValue = std::pair<size_type, size_type>;

    /// @brief Update the occurrences of a hashed block by adding the new
    ///   block index and offset and updating the first occurrence if needed
    /// @param occurrences A reference to the occurrences in the map
    /// @param input_value The new block index and offset to add to the
    ///   occurrences
    inline static void update(RabinKarpHash&,
                              BlockOccurrences& occurrences,
                              InputValue&& input_value) {
      occurrences.add_block(input_value.first);
      occurrences.update(input_value.first, input_value.second);
    }

    /// @brief Initialize the occurrences of a hashed block.
    /// @param input_value A pair of the block index and offset of one of the
    ///   block's occurrences
    /// @return The initialized occurrences only containing the given block
    inline static BlockOccurrences init(RabinKarpHash&,
                                        InputValue&& input_value) {
      BlockOccurrences occurrences(input_value.first);
      occurrences.add_block(input_value.first);
      occurrences.update(input_value.first, input_value.second);
      return occurrences;
    }
  };

  /// @brief A map containing hashed block pairs mapped to their occurrences
  using BlockPairMap = RabinKarpMap<PairOccurrences, UpdatePairOccurrences>;
  /// @brief A map containing hashed blocks mapped to their occurrences
  using BlockMap = RabinKarpMap<BlockOccurrences, UpdateBlockOccurrences>;

  /// @brief Constructs the block tree.
  /// @param text The input text.
  void construct(const std::vector<input_type>& text) {
    const size_type text_len = text.size();
    /// The number of characters a block tree with s top-level blocks and arity
    /// of strictly tau would exceed over the text size
    int64_t padding;
    /// The height of the tree
    int64_t tree_height;
    /// The size of the largest blocks (i.e. the top level blocks)
    int64_t top_block_size;

    this->calculate_padding(padding, text_len, tree_height, top_block_size);

    const bool is_padded = padding > 0;

    std::vector<LevelData> levels;

    // Prepare the top level
    levels.emplace_back(0, top_block_size, text_len / top_block_size);
    LevelData& top_level = levels.back();
    top_level.block_starts->reserve(ceil_div(text_len, top_level.block_size));
    for (size_type i = 0; i < text_len; i += top_level.block_size) {
      top_level.block_starts->push_back(i);
    }
    top_level.block_size = top_block_size;
    top_level.num_blocks = top_level.block_starts->size();

    size_t pairs_ns = 0;
    size_t blocks_ns = 0;
    size_t generate_ns = 0;

    // Construct the pre-pruned tree level by level
    for (size_t level = 0; level < static_cast<size_t>(tree_height); level++) {
      std::cout << "level " << level << std::endl;
      LevelData& current = levels.back();

      TimePoint now = Clock::now();
      scan_block_pairs(text, current, is_padded);
      pairs_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                      Clock::now() - now)
                      .count();
      now = Clock::now();
      scan_blocks(text, current, is_padded);
      blocks_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                       Clock::now() - now)
                       .count();
      now = Clock::now();

      // Generate the next level (if we're not at the last level)
      if (level < static_cast<size_t>(tree_height) - 1) {
        levels.push_back(std::move(generate_next_level(text, current)));
      }
      generate_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                         Clock::now() - now)
                         .count();
    }
    TimePoint now = Clock::now();

    std::cout << "pairs: " << (pairs_ns / 1'000'000)
              << "ms,\n\thash pairs: " << (bp_hash_pairs_ns / 1'000'000)
              << "ms,\n\tscan pairs: " << (bp_scan_pairs_ns / 1'000'000)
              << "ms,\n\tmarkings: " << (bp_markings_ns / 1'000'000)
              << "ms,\n\tbitvec: " << (bp_bitvec_ns / 1'000'000)
              << "ms,\nblocks: " << (blocks_ns / 1'000'000)
              << "ms,\n\thash blocks: " << (b_hash_blocks_ns / 1'000'000)
              << "ms,\n\tscan blocks: " << (b_scan_blocks_ns / 1'000'000)
              << "ms,\n\tupdate blocks: " << (b_update_blocks_ns / 1'000'000)
              << "ms,\ngenerate_next: " << (generate_ns / 1'000'000) << "ms,"
              << std::endl;
    prune(levels);
    size_t prune_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - now)
            .count();
    now = Clock::now();

    std::cout << "prune: " << (prune_ns / 1'000'000) << "ms," << std::endl;
    make_tree(text, levels, padding);
    size_t make_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - now)
            .count();
    std::cout << "make: " << (make_ns / 1'000'000) << "ms" << std::endl;
  }

  /// @brief Returns the ceiling of x / y for x > 0;
  ///
  /// https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
  inline static size_t ceil_div(std::integral auto x, std::integral auto y) {
    return 1 + ((x - 1) / y);
  }

  /// @brief Scan through the blocks pairwise in order to identify which blocks
  /// should be replaced with back blocks.
  ///
  /// @param text The input string.
  /// @param level The data for the current level.
  /// @param is_padded `true` iff the last block on this level *does not* end at
  ///   the exact end of the text.
  ///
  /// @return The block start indices for the next level of the tree
  void scan_block_pairs(const std::vector<input_type>& text,
                        LevelData& level,
                        const bool is_padded) {
    if (level.num_blocks < 4) {
      level.is_internal = std::make_unique<BitVector>(level.num_blocks, true);
      level.is_internal_rank = std::make_unique<Rank>(*level.is_internal);
      return;
    }

    // A map containing hashed block pairs mapped to their indices of the
    // pairs' first block respectively
    BlockPairMap map(BT_FILL_THRESHOLD, BT_NUM_THREADS, BT_QUEUE_CAPACITY);

    TimePoint now = Clock::now();
    std::atomic_size_t num_threads_finished = 0;

#pragma omp parallel default(none) num_threads(BT_NUM_THREADS)                 \
    shared(level, map, text, now, is_padded, std::cout, num_threads_finished)
    {
      const size_t thread_id = omp_get_thread_num();
      typename BlockPairMap::Shard shard = map.get_shard(thread_id);
      const size_t num_threads = omp_get_num_threads();
      const size_t block_size = level.block_size;
      const size_t pair_size = 2 * block_size;
      const size_t num_block_pairs = level.num_blocks - 1 - is_padded;
      const auto& block_starts = *level.block_starts;

      // Hash every window and determine for all block pairs whether they have
      // previous occurrences.
      size_t segment_size =
          std::max<size_t>(1, ceil_div(num_block_pairs, num_threads));

      // Start and end index of the current thread's segment
      const auto start = thread_id * segment_size;
      const auto end =
          std::min<size_t>(num_block_pairs, (thread_id + 1) * segment_size);

      for (size_t i = start; i < end; ++i) {
        // If the next block is not adjacent, we cannot hash the pair starting
        // at the current block
        if (!level.next_is_adjacent(i)) {
          continue;
        }
        // Move the hasher to the current block pair
        RabinKarp rk(text, SIGMA, block_starts[i], pair_size, PRIME);
        RabinKarpHash hash = rk.current_hash();
        // Try to find the hash in the map, insert a new entry if it doesn't
        // exist, and add the current block to the entry
        shard.insert(hash, i);
        if (shard.should_handle_queue()) {
          shard.handle_queue();
        }
      }
      num_threads_finished.fetch_add(1);
      // Threads might be done before the others with the loop. So essentially,
      // we need all threads to wait for the others to finish the loop and
      // handle thread events that might come in from the other threads
      do {
        shard.handle_queue();
      } while (num_threads_finished.load() < num_threads);
#pragma omp barrier
#pragma omp single
      {
        bp_hash_pairs_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                 now)
                .count();
        now = Clock::now();
      }

      if (start < static_cast<size_t>(num_block_pairs)) {
        RabinKarp rk(text, SIGMA, block_starts[start], pair_size, PRIME);
        for (size_t i = start; i < end; ++i) {
          if (!level.next_is_adjacent(i) | !level.next_is_adjacent(i + 1)) {
            continue;
          }
          scan_windows_in_block_pair(rk, map, block_size, i);
        }
      }
    }

    bp_scan_pairs_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - now)
            .count();

    level.is_internal = std::make_unique<BitVector>(level.num_blocks);
    fill_is_internal(*level.is_internal, map);
    level.is_internal_rank = std::make_unique<Rank>(*level.is_internal);
  }

  /// @brief Fills the bit vector `is_internal` based on the values in the given
  ///     map.
  /// @param is_internal An unfilled bit vector with a bit for each block on
  ///     this level.
  /// @param map A map, mapping hashed block pairs to their first occurrence's
  ///     block index.
  void fill_is_internal(BitVector& is_internal, BlockPairMap& map) {
    const size_type num_blocks = is_internal.size();
    TimePoint now = Clock::now();
    // Set up the packed array holding the markings for each block.
    // Each mark is a 2-bit number.
    // The MSB is 1 iff the block and its successor have a prior occurrence.
    // The LSB is 1 iff the block and its predecessor have a prior occurrence.
    sdsl::int_vector<2> markings(num_blocks, 0);
    map.for_each(
        [&markings](const RabinKarpHash&, const PairOccurrences& pair_occs) {
          for (const size_type occ : pair_occs.occurrences) {
            if (pair_occs.first_occ_block < occ) {
              markings[occ] = markings[occ] | 0b10;
              markings[occ + 1] = markings[occ + 1] | 0b01;
            }
          }
        });
    bp_markings_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - now)
            .count();
    now = Clock::now();

    // Generate the bit vector indicating which blocks are internal

    is_internal[0] = true;
    is_internal[num_blocks - 1] = markings[num_blocks - 1] != 0b01;
    for (size_type i = 0; i < num_blocks - 1; ++i) {
      const bool block_is_internal = markings[i] != 0b11;
      is_internal[i] = block_is_internal;
    }
    bp_bitvec_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - now)
            .count();
  }

  /// @brief Scan through the windows starting in a block and mark
  ///   them accordingly if they represent the earliest occurrence of some block
  ///   hash.
  ///
  /// The supplied `RabinKarp` hasher must be at the start of the block.
  /// @param rk A Rabin-Karp hasher whose state is at the start of the block.
  /// @param map The map containing the hashes of block pairs mapped to their
  ///   block indexes at which they occur.
  /// @param num_iterations The number of contiguous windows to hash.
  /// @param current_block_index The index of the block being currently hashed.
  static inline void
  scan_windows_in_block_pair(RabinKarp& rk,
                             BlockPairMap& map,
                             const size_t num_iterations,
                             const size_type current_block_index) {
    for (size_t offset = 0; offset < num_iterations; ++offset, rk.next()) {
      RabinKarpHash current_hash = rk.current_hash();
      // Find the hash of the current window among the hashed block pairs.
      auto found = map.find(current_hash);
      if (found == map.end()) {
        continue;
      }
      PairOccurrences& occurrences = found->second;
      occurrences.update(current_block_index);
    }
  }

  /// @brief Determine the positions for each block's earliest occurrence if
  /// there is any.
  ///
  /// @param s The input text
  /// @param level_data The data for the current level
  /// @param is_padded true, iff the last block of the level extends past the
  ///   end of the text
  void scan_blocks(const std::vector<input_type>& text,
                   LevelData& level_data,
                   const bool is_padded) {
    const size_t num_blocks = level_data.num_blocks;

    level_data.pointers =
        std::make_unique<std::vector<size_type>>(num_blocks, NO_EARLIER_OCC);
    level_data.offsets =
        std::make_unique<std::vector<size_type>>(num_blocks, 0);
    level_data.counters =
        std::make_unique<std::vector<size_type>>(num_blocks, 0);

    if (num_blocks <= 2) {
      return;
    }

    // A map hashing blocks and saving where they occur.
    BlockMap links(BT_FILL_THRESHOLD, BT_NUM_THREADS, BT_QUEUE_CAPACITY);

    TimePoint now = Clock::now();

    // The number of threads finished with hashing blocks
    std::atomic_size_t num_threads_finished = 0;
#pragma omp parallel default(none) num_threads(BT_NUM_THREADS)                 \
    shared(level_data, text, links, now, is_padded, num_threads_finished)
    {
      const size_t num_threads = omp_get_num_threads();
      const size_t thread_id = omp_get_thread_num();
      typename BlockMap::Shard shard = links.get_shard(thread_id);
      const size_t block_size = level_data.block_size;
      const std::vector<size_type>& block_starts = *level_data.block_starts;
      // Number of total iterations the for loop should do
      const size_t num_total_iterations = level_data.num_blocks - is_padded - 1;
      // The number of iterations each thread should do
      const size_t segment_size = ceil_div(num_total_iterations, num_threads);
      // The start and end index of the current thread's segment
      const size_t start = thread_id * segment_size;
      const size_t end = std::min<size_t>(num_total_iterations,
                                          (thread_id + 1) * segment_size);

      // Hash each block and store their hashes in the map
      // FIXME This causes issues when run in parallel
      //  In make_tree, we get an error when deallocating vectors in LevelData
      //  Seems like in this case, the algorithm fails to identify some earlier
      //  occurrences for non-internal blocks, leading to writes to offsets[-1]
      //  etc. later on.
      for (size_t i = start; i < end; ++i) {
        const RabinKarp rk(text, SIGMA, block_starts[i], block_size, PRIME);
        RabinKarpHash hash = rk.current_hash();
        shard.insert(hash, {i, 0});
        // The thread checks whether it should handle the inserts in its queue
        if (shard.should_handle_queue()) {
          shard.handle_queue();
        }
      }
      num_threads_finished.fetch_add(1);
      // Threads might be done with the loop before others. So essentially,
      // we need all threads to wait for the others to finish the loop and
      // handle thread events that might come in from the other threads
      do {
        shard.handle_queue();
      } while (num_threads_finished.load() < num_threads);
#pragma omp single
      {
        b_hash_blocks_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                 now)
                .count();
        now = Clock::now();
        // links.print_map_loads();
      }

      // Hash every window and find the first occurrences for every block.
      if (start < block_starts.size() - is_padded) {
        RabinKarp rk(text, SIGMA, block_starts[start], block_size, PRIME);
        for (size_t i = start; i < end; ++i) {
          if (!level_data.next_is_adjacent(i)) {
            continue;
          }
          if (static_cast<int64_t>(rk.init_) != block_starts[i]) {
            rk.restart(block_starts[i]);
          }
          scan_windows_in_block(rk, links, level_data, i);
        }
      }
    }
    b_scan_blocks_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - now)
            .count();
    now = Clock::now();

    // By this point, the map should contain the first occurrences of every
    // respective block's content. We then fill the pointers and offsets with
    // this data and increment counters accordingly
    links.for_each(
        [&level_data](const RabinKarpHash&, const BlockOccurrences& occs) {
          auto first_occ = occs.first_occ.load();
          for (const size_type occ : occs.occurrences) {
            if (occ == first_occ.block ||
                (first_occ.offset > 0 && occ == first_occ.block + 1)) {
              continue;
            }

            (*level_data.pointers)[occ] = first_occ.block;
            (*level_data.offsets)[occ] = first_occ.offset;
            const bool is_back_block = !(*level_data.is_internal)[occ];
            (*level_data.counters)[first_occ.block] += 1;
            (*level_data.counters)[first_occ.block + 1] +=
                is_back_block && (first_occ.offset > 0);
          }
        });

    b_update_blocks_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - now)
            .count();
  }

  /// @brief Scans through block-sized windows starting inside one block and
  ///     tries to find blocks with matching hashes in the map. Such blocks will
  ///     have their earliest occurrence update.
  /// @param rk A Rabin-Karp hasher whose current state is at a block start.
  /// @param links A map whose keys are hashed blocks and the values
  ///   are all block indices of blocks matching the hash in ascending order.
  /// @param level_data The data for the current level.
  /// @param current_block_index The index of the block which the
  ///   Rabin-Karp hasher is situated in.
  static void scan_windows_in_block(RabinKarp& rk,
                                    BlockMap& links,
                                    LevelData& level_data,
                                    const size_type current_block_index) {
    for (size_type offset = 0; offset < level_data.block_size;
         ++offset, rk.next()) {
      const RabinKarpHash hash = rk.current_hash();
      // Find all blocks in the multimap that match our hash
      auto found = links.find(hash);
      if (found == links.end()) {
        continue;
      }
      found->second.update(current_block_index, offset);
    }
  }

  /// @brief Generate the block size, number of block and block start indices
  /// for the next level.
  ///
  /// This depends on the current level's block size, number of blocks and
  /// is_internal bit vector being filled.
  ///
  /// @param text The input text.
  /// @param level The level data of the current level.
  /// @return The level data of the next level.
  [[nodiscard]] LevelData
  generate_next_level(const std::vector<input_type>& text,
                      const LevelData& level) const {
    const size_t block_size = level.block_size;
    const size_t num_blocks = level.num_blocks;
    const auto& is_internal = *level.is_internal;
    const size_t next_block_size = block_size / this->tau_;

    std::vector<size_type> new_block_starts;
    new_block_starts.reserve(num_blocks * this->tau_);
    for (size_t i = 0; i < num_blocks; ++i) {
      if (!is_internal[i]) {
        continue;
      }

      // We generate up to tau new blocks for each internal block,
      // excluding blocks that start past the end of the text
      const auto parent_block_start = (*level.block_starts)[i];
      for (size_t j = 0, current_block_start = parent_block_start;
           j < static_cast<size_t>(this->tau_) &&
           current_block_start < text.size();
           ++j, current_block_start += next_block_size) {
        new_block_starts.push_back(current_block_start);
      }
    }

    LevelData next_level(level.level_index + 1,
                         next_block_size,
                         new_block_starts.size());
    next_level.block_starts =
        std::make_unique<std::vector<size_type>>(std::move(new_block_starts));
    return next_level;
  }

  ///
  /// @brief Takes a vector of levels and fills the block tree fields with them.
  ///
  /// @param[in] levels A vector containing data for each level, with the first
  /// entry corresponding to the topmost level.
  ///
  void make_tree(const std::vector<input_type>& text,
                 std::vector<LevelData>& levels,
                 int64_t padding) {
    const bool is_padded = padding > 0;

    // Count the current number of internal blocks per level
    std::vector<size_type> new_num_internal(levels.size(), 0);
    for (size_t level = 0; level < levels.size(); level++) {
      for (size_t block = 0; block < levels[level].is_internal->size();
           block++) {
        if ((*levels[level].is_internal)[block]) {
          new_num_internal[level]++;
        }
      }
    }

    // Create first level
    bool found_back_block = levels[0].is_internal->size() >
                                static_cast<size_t>(new_num_internal[0]) ||
                            !this->CUT_FIRST_LEVELS;
    LevelData& top_level = levels.front();
    if (found_back_block) {
      const size_t n = top_level.num_blocks;
      const size_t num_internal = new_num_internal[0];
      auto pointers = new sdsl::int_vector<>(n - num_internal, 0);
      auto offsets = new sdsl::int_vector<>(n - num_internal, 0);
      size_t num_back_blocks = 0;
      for (size_t i = 0; i < n; i++) {
        // if a back block is found, add its pointer and offset
        if (!(*top_level.is_internal)[i]) {
          (*pointers)[num_back_blocks] = (*top_level.pointers)[i];
          (*offsets)[num_back_blocks] = (*top_level.offsets)[i];
          num_back_blocks++;
        }
      }
      sdsl::util::bit_compress(*pointers);
      sdsl::util::bit_compress(*offsets);
      this->block_tree_types_.push_back(top_level.is_internal.release());
      this->block_tree_types_rs_.push_back(
          new Rank(*this->block_tree_types_.back()));
      this->block_tree_pointers_.push_back(pointers);
      this->block_tree_offsets_.push_back(offsets);
      this->block_size_lvl_.push_back(top_level.block_size);
    }
    top_level.pointers.reset();
    top_level.offsets.reset();
    top_level.counters.reset();

    // Add level data to the tree
    for (size_t level_index = 1; level_index < levels.size(); level_index++) {
      LevelData& level = levels[level_index];
      LevelData& previous_level = levels[level_index - 1];
      found_back_block |= static_cast<size_t>(new_num_internal[level_index]) <
                          levels[level_index].is_internal->size();
      if (!found_back_block) {
        level.is_internal.reset();
        level.is_internal_rank.reset();
        level.pointers.reset();
        level.offsets.reset();
        level.counters.reset();
        previous_level.block_starts.reset();
        continue;
      }

      make_tree_level(levels,
                      new_num_internal,
                      level_index,
                      is_padded,
                      text.size());

      // We don't need these anymore
      if (level_index < levels.size() - 1) {
        level.is_internal.reset();
      }
      level.is_internal_rank.reset();
      level.pointers.reset();
      level.offsets.reset();
      level.counters.reset();
      previous_level.block_starts.reset();
    }

    this->leaf_size = levels.back().block_size / this->tau_;
    // Construct the leaf string
    int64_t leaf_count = 0;
    auto& last_is_internal = *levels.back().is_internal;
    std::vector<size_type>& last_block_starts = *levels.back().block_starts;
    for (size_t block = 0; block < last_is_internal.size(); block++) {
      if (!last_is_internal[block]) {
        continue;
      }
      const size_type block_start = last_block_starts[block];
      // For every leaf on the last level, we have tau leaf blocks
      leaf_count += this->tau_;
      // Iterate through all characters in this child and
      // add them to the leaf string
      for (size_t b = 0; b < static_cast<size_t>(this->leaf_size * this->tau_);
           b++) {
        if (static_cast<size_t>(block_start + b) < text.size()) {
          this->leaves_.push_back(text[block_start + b]);
        }
      }
    }
    this->amount_of_leaves = leaf_count;
    this->compress_leaves();
  }

  /// @brief Generates a level and adds the relevant data to the block tree.
  ///
  /// @param levels The vector of levels of the tree.
  /// @param level_index The index of the level to generate. This must be
  ///   strictly greater than 0.
  /// @param is_padded Whether there is padding in the last block of the tree
  void make_tree_level(std::vector<LevelData>& levels,
                       const std::vector<size_type>& new_num_internal,
                       const size_t level_index,
                       const bool is_padded,
                       const size_t text_len) {
    LevelData& previous_level = levels[level_index - 1];
    LevelData& level = levels[level_index];

    size_type new_size =
        (new_num_internal[level_index - 1] - is_padded) * this->tau_;
    // Determine the number of children the last block generated
    if (is_padded) {
      const size_type last_block_parent_start =
          previous_level.block_starts->back();
      const size_type block_size = level.block_size;
      new_size += ceil_div(text_len - last_block_parent_start, block_size);
    }
    previous_level.block_starts.reset();
    const size_type num_internal = new_num_internal[level_index];

    // Allocate new vectors for the tree
    auto* is_internal = new BitVector(new_size);
    auto* pointers = new sdsl::int_vector<>(new_size - num_internal, 0);
    auto* offsets = new sdsl::int_vector<>(new_size - num_internal, 0);

    // Number of non-pruned blocks before the current block
    size_type num_non_pruned = 0;
    // Number of back blocks before the current block
    size_type num_back_blocks = 0;
    // Number of pruned blocks before the current block
    size_type num_pruned = 0;

    // We will reuse the allocated memory of the pointers vector to store
    // the number of pruned blocks before the block.
    // The invariant is that all values up to i are overwritten while all
    // values starting after i will still be valid pointers
    // This contains the number of pruned blocks before the block i
    std::vector<size_type>& prefix_pruned_blocks = *level.pointers;
    for (size_type i = 0; i < level.num_blocks; i++) {
      const size_type ptr = (*level.pointers)[i];
      prefix_pruned_blocks[i] = num_pruned;

      // If the current block is not pruned, add it to the new tree
      if (ptr == PRUNED) {
        num_pruned++;
        continue;
      }

      // Add it to the is_internal bit vector
      const bool block_is_internal = (*level.is_internal)[i];
      (*is_internal)[num_non_pruned] = block_is_internal;
      num_non_pruned++;

      if (block_is_internal) {
        continue;
      }

      // If it is a back block, add its pointer and offset
      const size_type offset = (*level.offsets)[i];

      (*pointers)[num_back_blocks] = ptr - prefix_pruned_blocks[ptr];
      (*offsets)[num_back_blocks] = offset;
      num_back_blocks++;
    }

    sdsl::util::bit_compress(*pointers);
    sdsl::util::bit_compress(*offsets);
    this->block_tree_types_.push_back(is_internal);
    this->block_tree_types_rs_.push_back(new Rank(*is_internal));
    this->block_tree_pointers_.push_back(pointers);
    this->block_tree_offsets_.push_back(offsets);
    this->block_size_lvl_.push_back(level.block_size);
  }

  /// @brief Prunes the tree of unnecessary nodes.
  /// @param levels The levels of the tre represented as a vector of levels.
  void prune(std::vector<LevelData>& levels) {
    // We need to traverse the block tree in post order,
    // handling children from right to left
    for (int block_index = levels[0].num_blocks - 1; block_index >= 0;
         --block_index) {
      prune_block(levels, 0, block_index);
    }
  }

  /// @brief Prunes a block and its descendants of unnecessary internal nodes.
  /// @param levels The WIP levels of the tree.
  /// @param level_index The level of the block to prune.
  /// @param block_index The index of the block to prune.
  /// @return Whether this block is/stays internal after the pruning process
  bool prune_block(std::vector<LevelData>& levels,
                   const size_t level_index,
                   const size_t block_index) {
    LevelData& level = levels[level_index];
    BitVector& is_internal = *level.is_internal;

    // If the current block is a back block already, there is nothing to prune
    if (!is_internal[block_index]) {
      return false;
    }

    const size_type first_child =
        level.is_internal_rank->rank1(block_index) * this->tau_;

    bool has_internal_children = false;

    // On the last level, all blocks just have leaves as children,
    // none of which can be pointed to. So only recurse, if we are not on the
    // last level.
    if (level_index < levels.size() - 1) {
      const size_type last_child =
          std::min<size_type>(first_child + this->tau_ - 1,
                              levels[level_index + 1].is_internal->size() - 1);
      // Iterate through children in reverse
      for (size_type child = last_child; child >= first_child; --child) {
        has_internal_children |= prune_block(levels, level_index + 1, child);
      }
    }

    // If any of the children is internal, this block stays internal as well
    if (has_internal_children) {
      return true;
    }

    const size_type pointer = (*level.pointers)[block_index];
    const size_type offset = (*level.offsets)[block_index];
    const size_type counter = (*level.counters)[block_index];
    // If there is no earlier occurrence or there are blocks pointing to this,
    // then this must stay internal
    if (pointer == NO_EARLIER_OCC || counter > 0) {
      return true;
    }

    // Now we know that there is an earlier occurrence,
    // and nothing is pointing here.
    // We will make this block here into a back block...
    is_internal[block_index] = false;
    (*level.counters)[pointer] += 1;
    (*level.counters)[pointer + 1] += offset > 0;

    if (level_index == levels.size() - 1) {
      return false;
    }

    // ...and mark the children as pruned
    LevelData& child_level = levels[level_index + 1];
    const size_type last_child =
        std::min<size_type>(first_child + this->tau_ - 1,
                            child_level.is_internal->size() - 1);
    for (size_type child = last_child; child >= first_child; --child) {
      const size_type child_pointer = (*child_level.pointers)[child];
      const size_type child_offset = (*child_level.offsets)[child];
      if (!(*child_level.is_internal)[child] && child_pointer < 0) {
        std::cout << "non-internal node missing pointer" << std::endl;
        std::cout << level_index << ", " << block_index << std::endl;
      } else if (child_pointer == PRUNED && child_pointer < 0) {
        std::cout << "pruned node missing pointer" << std::endl;
      }
      assert(!(*child_level.is_internal)[child] || child_pointer == PRUNED);
      assert(child_pointer >= 0);
      // Decrement the counter of where the child points
      (*child_level.counters)[child_pointer] -= 1;
      (*child_level.counters)[child_pointer + 1] -= child_offset > 0;
      // Mark the child as pruned
      (*child_level.pointers)[child] = PRUNED;
    }

    return false;
  }

public:
  BlockTreeFPParSharded(const std::vector<input_type>& text,
                        const size_t arity,
                        const size_t root_arity,
                        const size_t max_leaf_length,
                        const size_t threads) {
    const auto old = omp_get_max_threads();
    const auto old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(threads));
    this->tau_ = arity;
    this->s_ = root_arity;
    this->max_leaf_length_ = max_leaf_length;
    this->map_unique_chars(text);
    construct(text);
    omp_set_dynamic(old_dynamic);
    omp_set_num_threads(old);
  }

  ~BlockTreeFPParSharded() {
    for (auto& rank : this->block_tree_types_rs_) {
      delete rank;
    }
    for (auto& bv : this->block_tree_types_) {
      delete bv;
    }
    for (auto& ptrs : this->block_tree_pointers_) {
      delete ptrs;
    }
    for (auto& offsets : this->block_tree_offsets_) {
      delete offsets;
    }
  }

  /// @brief Validates that a back-pointer actually points to the same text
  /// content.
  /// @param text The input text.
  /// @param level_index The index of the current level.
  /// @param block_index The block index.
  /// @param block_start The start index of the block's content in the text.
  /// @param source_start The start index of the source block's content in the
  /// text.
  /// @param source_pointer The block index of the source block.
  /// @param source_offset The offset from which the block copies out of the
  /// source block.
  /// @param block_size The block size.
  /// @return `true`, iff the pointer is valid. false otherwise
  bool debug_validate_pointer(const std::vector<input_type>& text,
                              const size_type level_index,
                              const size_type block_index,
                              const size_type block_start,
                              const size_type source_start,
                              const size_type source_pointer,
                              const size_type source_offset,
                              const size_type block_size) const {
    if (source_start + block_size > block_start) {
      std::cerr << "source overlapping block on level " << level_index
                << ":\n\tBlock Start: " << block_start
                << "\n\tSource Start: " << source_start
                << "\n\tBlock Size: " << block_size
                << "\n\tBlock: " << block_index
                << "\n\tSource Block: " << source_pointer
                << "\n\tSource Offset: " << source_offset << std::endl;
      return false;
    }
    for (size_type i = 0; i < block_size; i++) {
      if (text[block_start + i] != text[source_start + i]) {
        std::cerr << "source block mismatch on level " << level_index << ": "
                  << "\n\tBlock Start: " << block_start
                  << "\n\tSource Start: " << source_start
                  << "\n\tBlock Size: " << block_size
                  << "\n\tBlock: " << block_index
                  << "\n\tSource Block: " << source_pointer
                  << "\n\tSource Offset: " << source_offset << std::endl;
        return false;
      }
    }
    return true;
  }
}; // namespace pasta

} // namespace pasta

#undef BT_NUM_THREADS
