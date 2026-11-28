CXX          := g++
CXXFLAGS     := -O3 -std=c++17 -Wall -Wextra -Wpedantic -Iinclude \
                -fno-tree-vectorize -mno-avx -mno-avx2 -mno-sse4.2
BRIDGE_FLAGS := -O3 -std=c++17 -Wall -Wextra -Wpedantic -Iinclude \
                -fno-tree-vectorize -mno-avx -mno-avx2 -mno-sse4.2 -shared -fPIC -fopenmp

BUILD_DIR    := build
TEST_BIN     := $(BUILD_DIR)/test_symmetry_and_safety
BENCH_BIN    := $(BUILD_DIR)/bench_scalar_eval
PREP_BIN     := $(BUILD_DIR)/preprocess_egaroucid
BRIDGE_SO    := $(BUILD_DIR)/libp2c64_batch.so
MATCH_BIN    := $(BUILD_DIR)/match_engine
MPC_BIN      := $(BUILD_DIR)/measure_mpc_sigma
RELABEL_BIN  := $(BUILD_DIR)/relabel_dataset

HEADERS      := include/p2c64_patterns.hpp include/p2c64_eval.hpp \
                include/p2c64_data.hpp include/p2c64_io.hpp include/p2c64_search.hpp

.PHONY: all tools test bench audit clean

all: $(TEST_BIN) $(BENCH_BIN) tools

tools: $(PREP_BIN) $(BRIDGE_SO) $(MATCH_BIN) $(MPC_BIN) $(RELABEL_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_BIN): tests/test_symmetry_and_safety.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BENCH_BIN): bench/bench_scalar_eval.cpp include/p2c64_patterns.hpp include/p2c64_eval.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(PREP_BIN): tools/preprocess_egaroucid.cpp include/p2c64_patterns.hpp include/p2c64_data.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BRIDGE_SO): tools/p2c64_batch_bridge.cpp include/p2c64_patterns.hpp include/p2c64_eval.hpp include/p2c64_io.hpp | $(BUILD_DIR)
	$(CXX) $(BRIDGE_FLAGS) $< -o $@

$(MATCH_BIN): tools/match_engine.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(MPC_BIN): tools/measure_mpc_sigma.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(RELABEL_BIN): tools/relabel_dataset.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

audit: $(BENCH_BIN)
	chmod +x scripts/audit_asm.sh
	./scripts/audit_asm.sh $(BENCH_BIN)

clean:
	rm -rf $(BUILD_DIR)
