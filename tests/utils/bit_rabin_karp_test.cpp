
#include <cstddef>
#include <istream>
#include <random>
#define asdasdkasld

#include <bitset>
#include <gtest/gtest.h>
#include <pasta/block_tree/utils/MersenneRabinKarp.hpp>

class BitRabinKarpTest : public ::testing::Test {
protected:
  pasta::BitVector bv;

  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<uint8_t> dist(1);

    constexpr size_t string_length = 100000;
    bv.resize(string_length * 2 + 3);
    for (size_t i = 0; i < string_length; ++i) {
      bv[i] = dist(gen);
    }
    // Offset it by some amount to check that even misaligned bit sequences
    // correctly match
    bv[string_length] = false;
    bv[string_length + 1] = false;
    bv[string_length + 2] = false;
    for (size_t i = 0; i < string_length; ++i) {
      bv[string_length + 3 + i] = static_cast<bool>(bv[i]);
    }
  }

public:
  bool
  compare_ranges(const size_t s1, const size_t s2, const size_t len) const {
    for (size_t i = 0; i < len; ++i) {
      if (get_bit(s1 + i) == get_bit(s2 + i)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool get_bit(const size_t bit_index) const {
    return bv[bit_index];
  }
};

TEST_F(BitRabinKarpTest, test_eq) {
  constexpr std::array<size_t, 4> sizes = {13, 24, 59, 1220};

  for (const size_t size : sizes) {
    const size_t half_len = bv.size() / 2;
    pasta::MersenneRabinKarp<bool, int32_t> rk1(bv, 0, size, (1ULL << 61) - 1);
    pasta::MersenneRabinKarp<bool, int32_t> rk2(bv,
                                                half_len + 3,
                                                size,
                                                (1ULL << 61) - 1);
    for (size_t i = 0; i < half_len - size; ++i) {
      const auto h1 = rk1.current_hash();
      const auto h2 = rk2.current_hash();
      if (h1 != h2) {
        std::cerr << "h1: ";
        for (const auto byte : h1.overlapping_range()) {
          std::cerr << std::bitset<8>{std::to_integer<uint8_t>(byte)} << ", ";
        }
        std::cerr << "\n";
        std::cerr << "h2: ";
        for (const auto byte : h2.overlapping_range()) {
          std::cerr << std::bitset<8>{std::to_integer<uint8_t>(byte)} << ", ";
        }
        std::cerr << "\n";
      }
      ASSERT_TRUE(h1 == h2);
      rk1.next();
      rk2.next();
    }
  }
}

TEST_F(BitRabinKarpTest, test_rnd) {
  constexpr std::array<size_t, 4> sizes = {13, 24, 59, 1220};

  for (const size_t size : sizes) {
    const size_t half_len = bv.size() / 2;
    pasta::MersenneRabinKarp<bool, int32_t> rk1(bv, 0, size, (1ULL << 61) - 1);
    pasta::MersenneRabinKarp<bool, int32_t> rk2(bv,
                                                half_len,
                                                size,
                                                (1ULL << 61) - 1);
    size_t offset = 0;
    for (size_t i = 0; i < half_len - size; ++i) {
      const auto h1 = rk1.current_hash();
      const auto h2 = rk2.current_hash();
      const bool success =
          (h1 == h2) == compare_ranges(offset, half_len + offset, size);
      if (!success) {
        std::cerr << "h1: ";
        for (const auto byte : h1.overlapping_range()) {
          std::cerr << std::bitset<8>{std::to_integer<uint8_t>(byte)} << ", ";
        }
        std::cerr << " offset: " << h1.start_ % 8 << "\n";
        std::cerr << "h2: ";
        for (const auto byte : h2.overlapping_range()) {
          std::cerr << std::bitset<8>{std::to_integer<uint8_t>(byte)} << ", ";
        }
        std::cerr << " offset: " << h2.start_ % 8 << "\n";
      }
      ASSERT_TRUE((h1 == h2) == compare_ranges(offset, half_len + offset, size))
          << "error at offset " << offset << " for half_len " << half_len
          << " and window size " << size;
      rk1.next();
      rk2.next();
      offset++;
    }
  }
}
