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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <pasta/block_tree/construction/block_tree_fp.hpp>
#include <pasta/block_tree/construction/block_tree_fp2_seq.hpp>
#include <sstream>
#include <string>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;
using Duration = Clock::duration;

int main(int argc, char** argv) {
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

  size_t arity = atoi(argv[2]);

  if (argc < 4) {
    std::cerr << "Please input root arity (s)" << std::endl;
    exit(1);
  }

  size_t root_arity = atoi(argv[3]);

  if (argc < 5) {
    std::cerr << "Please input max leaf length" << std::endl;
    exit(1);
  }

  size_t leaf_length = atoi(argv[4]);

  std::stringstream ss;
  ss << argv[1] << "_arit" << arity << "_root" << root_arity << "_leaf"
     << leaf_length << "_new.bt";
  std::string out_path = ss.str();

  std::cout << "building block tree with parameters:"
            << "\narity: " << arity << "\nroot arity: " << root_arity
            << "\nmax leaf length: " << leaf_length << "\nsaving to "
            << out_path << std::endl;

  std::string input;
  std::ifstream t(argv[1]);
  std::stringstream buffer;
  buffer << t.rdbuf();
  input = buffer.str();

  std::vector<uint8_t> text(input.begin(), input.end());

  TimePoint now = Clock::now();

  auto bt =
      std::make_unique<pasta::BlockTreeFP2<uint8_t, int32_t>>(text,
                                                              arity,
                                                              root_arity,
                                                              leaf_length);

  // auto bt =
  //     pasta::make_block_tree_fp<uint8_t, int32_t>(text, arity, leaf_length);

  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - now)
          .count();

  std::cout << "bt size: " << bt->print_space_usage() / 1000 << "kb\n"
            << "Time: " << elapsed << "ms" << std::endl;
  // std::ofstream ot(out_path);
  // bt->serialize(ot);
  // ot.close();

  return 0;
}

/******************************************************************************/
