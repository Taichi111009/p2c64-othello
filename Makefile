CXX      := g++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Wpedantic -Iinclude \
            -fno-tree-vectorize -mno-avx -mno-avx2 -mno-sse4.2

BUILD_DIR := build
TEST_BIN  := $(BUILD_DIR)/test_symmetry_and_safety
BENCH_BIN := $(BUILD_DIR)/bench_scalar_eval

HEADERS := include/p2c64_patterns.hpp include/p2c64_eval.hpp

.PHONY: all test bench audit clean

all: $(TEST_BIN) $(BENCH_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_BIN): tests/test_symmetry_and_safety.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BENCH_BIN): bench/bench_scalar_eval.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

audit: $(BENCH_BIN)
	chmod +x scripts/audit_asm.sh
	./scripts/audit_asm.sh $(BENCH_BIN)

clean:
	rm -rf $(BUILD_DIR)
