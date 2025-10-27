#include "nn/python_callback_predictor.hpp"
#include <stdexcept>

namespace mcts {

PythonCallbackPredictor::PythonCallbackPredictor(py::object py_predictor)
    : py_predictor_(std::move(py_predictor)) {

    // Verify the Python object has a predict method
    if (!py::hasattr(py_predictor_, "predict")) {
        throw std::runtime_error("Python predictor must have a 'predict(state)' method");
    }
}

NNPredictor::Prediction PythonCallbackPredictor::predict(const std::array<float, 27>& state) {
    // Convert C++ array to NumPy array for Python
    py::array_t<float> state_array({27}, state.data());

    // Call Python predict method
    py::object result = py_predictor_.attr("predict")(state_array);

    // Extract policy and value from Python tuple (policy, value)
    py::tuple py_result = result.cast<py::tuple>();
    if (py_result.size() != 2) {
        throw std::runtime_error("Python predict() must return (policy, value) tuple");
    }

    // Extract policy (9 floats)
    py::array_t<float> policy_array = py_result[0].cast<py::array_t<float>>();
    if (policy_array.size() != 9) {
        throw std::runtime_error("Policy must have 9 elements");
    }

    // Extract value (1 float)
    float value = py_result[1].cast<float>();

    // Build prediction
    NNPredictor::Prediction pred;
    auto policy_ptr = policy_array.data();
    for (size_t i = 0; i < 9; ++i) {
        pred.policy[i] = policy_ptr[i];
    }
    pred.value = value;

    return pred;
}

} // namespace mcts
