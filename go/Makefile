# Makefile — build all C/C++ binaries for the Go AlphaZero project
# Run from the go/ directory.

TORCH := $(shell .venv/bin/python -c "import torch,os; print(os.path.dirname(torch.__file__))")

CC  = gcc
CXX = g++

CFLAGS   = -O2 -Wall -I.
CXXFLAGS = -O2 -std=c++17 -Wall -fno-pie -no-pie -I. \
           -I$(TORCH)/include \
           -I$(TORCH)/include/torch/csrc/api/include
LDFLAGS  = -L$(TORCH)/lib -Wl,-rpath,$(TORCH)/lib \
           -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10

# Shared source files
ENGINE_C = go_engine.c
MCTS_CPP = mcts.cpp
NN_CPP   = nn_inference.cpp
NPY_C    = npy_writer.c

BINARIES = test_go_engine test_mcts test_nn selfplay_cpp gatekeeper

.PHONY: all tests clean

all: $(BINARIES)

# Phase 1 — Go engine unit tests (pure C)
test_go_engine: $(ENGINE_C) tests/test_go_engine.c
	$(CC) $(CFLAGS) -o $@ $^

# Phase 2 — MCTS unit tests (C++, no LibTorch)
test_mcts: $(ENGINE_C) $(MCTS_CPP) tests/test_mcts.cpp
	$(CXX) -O2 -std=c++17 -Wall -I. -o $@ $^

# Phase 3 — NN inference unit tests (C++ + LibTorch)
test_nn: $(ENGINE_C) $(MCTS_CPP) $(NN_CPP) tests/test_nn.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# Phase 4 — Self-play data generator (C++ + LibTorch)
selfplay_cpp: $(ENGINE_C) $(MCTS_CPP) $(NN_CPP) $(NPY_C) selfplay_cpp.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# Model gating binary (C++ + LibTorch)
gatekeeper: $(ENGINE_C) $(MCTS_CPP) $(NN_CPP) gatekeeper.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# Run all test binaries
tests: test_go_engine test_mcts test_nn
	@echo "=== test_go_engine ==="
	./test_go_engine
	@echo "=== test_mcts ==="
	./test_mcts
	@echo "=== test_nn ==="
	./test_nn

clean:
	rm -f $(BINARIES)
