#pragma once

#include "predictor.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace mcts {

/**
 * NNPredictor that calls back to a Python neural network model.
 *
 * This allows C++ MCTS to use PyTorch models without LibTorch.
 * The tree search is fast (C++), but NN inference uses Python.
 */
class PythonCallbackPredictor : public NNPredictor {
public:
    /**
     * Create predictor with Python callable.
     *
     * @param py_predictor Python object with predict(state) -> (policy, value) method
     */
    explicit PythonCallbackPredictor(py::object py_predictor);

    ~PythonCallbackPredictor() override = default;

    /**
     * Get prediction from Python model.
     *
     * @param state Board state (27 floats: 3x3x3 planes)
     * @return Prediction with policy (9 floats) and value (1 float)
     */
    Prediction predict(const std::array<float, 27>& state) override;

private:
    py::object py_predictor_;  // Python model object
};

} // namespace mcts
