# Makefile — build the C++ self-play backend for chess AlphaZero.
# Run from the chess/ directory.
#
#   make all      build chess_selfplay
#   make tests    build & run encoding + MCTS unit tests (no LibTorch)
#   make test_nn  build the LibTorch inference smoke test
#   make clean

# Resolve the Python that has torch installed (project .venv lives one level up).
PYTHON := $(shell test -x ../.venv/bin/python && echo ../.venv/bin/python || \
                  (test -x .venv/bin/python && echo .venv/bin/python || echo python3))
TORCH  := $(shell $(PYTHON) -c "import torch,os; print(os.path.dirname(torch.__file__))")
UNAME  := $(shell uname -s)

CC  = gcc
CXX = g++

CFLAGS = -O2 -Wall -I.

# Pure C++17 flags (no LibTorch) for the engine/MCTS/encoding and unit tests.
PURE_CXXFLAGS = -O2 -std=c++17 -Wall -I.

ifeq ($(UNAME), Darwin)
# macOS: no PIE / --no-as-needed flags (those are GNU ld only).
CXXFLAGS = -O2 -std=c++17 -Wall -I. \
           -I$(TORCH)/include \
           -I$(TORCH)/include/torch/csrc/api/include
LDFLAGS  = -L$(TORCH)/lib -Wl,-rpath,$(TORCH)/lib \
           -ltorch -ltorch_cpu -lc10
else
# Linux
CXXFLAGS = -O2 -std=c++17 -Wall -fno-pie -no-pie -I. \
           -I$(TORCH)/include \
           -I$(TORCH)/include/torch/csrc/api/include
LDFLAGS  = -L$(TORCH)/lib -Wl,-rpath,$(TORCH)/lib \
           -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10
endif

ENC_CPP  = chess_encoding.cpp
MCTS_CPP = mcts.cpp
NN_CPP   = nn_inference.cpp
NPY_C    = npy_writer.c

.PHONY: all tests clean

all: chess_selfplay

# Self-play data generator (C++ + LibTorch)
chess_selfplay: $(ENC_CPP) $(MCTS_CPP) $(NN_CPP) $(NPY_C) chess_selfplay.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# Encoding unit tests (pure C++, cross-validated against Python reference)
test_encoding: $(ENC_CPP) tests/test_encoding.cpp
	$(CXX) $(PURE_CXXFLAGS) -o $@ $^

# MCTS unit tests (pure C++, stub NN)
test_mcts: $(ENC_CPP) $(MCTS_CPP) tests/test_mcts.cpp
	$(CXX) $(PURE_CXXFLAGS) -o $@ $^

# NN inference smoke test (C++ + LibTorch)
test_nn: $(ENC_CPP) $(MCTS_CPP) $(NN_CPP) tests/test_nn.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

# Build + run the pure-C++ tests.  test_encoding needs a reference file; it is
# generated from Python if missing.
tests: test_encoding test_mcts
	@test -f tests/encoding_ref.txt || \
	    $(PYTHON) tests/gen_encoding_reference.py tests/encoding_ref.txt
	@echo "=== test_encoding ==="
	./test_encoding tests/encoding_ref.txt
	@echo "=== test_mcts ==="
	./test_mcts

clean:
	rm -f chess_selfplay test_encoding test_mcts test_nn
