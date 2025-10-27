cmake_minimum_required(VERSION 3.15)
project(mcts_cpp VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Create the Python module
pybind11_add_module(mcts_cpp
    src/bindings.cpp
    src/game/tictactoe.cpp
    src/mcts/node.cpp
    src/mcts/search.cpp
    src/nn/predictor.cpp
    src/nn/python_callback_predictor.cpp
    src/selfplay/game.cpp
    src/selfplay/batch_generator.cpp
)

# Include directories
target_include_directories(mcts_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set output directory
set_target_properties(mcts_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optional: LibTorch support (will be added in Phase 3)
# Uncomment when ready to add neural network inference
# find_package(Torch REQUIRED)
# target_link_libraries(mcts_cpp PRIVATE "${TORCH_LIBRARIES}")
# target_compile_definitions(mcts_cpp PRIVATE USE_TORCH)
