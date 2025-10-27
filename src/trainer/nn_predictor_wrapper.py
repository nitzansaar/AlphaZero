"""
Python wrapper for neural network to be used by C++ PythonCallbackPredictor.

This allows C++ MCTS to use PyTorch models without LibTorch.
The tree search is fast (C++), but NN inference uses Python.
"""

import torch
import numpy as np
from typing import Tuple


class NNPredictorWrapper:
    """
    Wrapper for PyTorch model that provides predict(state) interface for C++.

    The C++ PythonCallbackPredictor calls this wrapper's predict() method
    during MCTS tree search.
    """

    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        Create predictor wrapper.

        Args:
            model: PyTorch model with (policy_logits, value) output
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict policy and value for a board state.

        This method is called by C++ during MCTS search.

        Args:
            state: NumPy array of shape (27,) - 3x3x3 canonical board state

        Returns:
            (policy, value) tuple:
                - policy: NumPy array of shape (9,) with move probabilities
                - value: float in [-1, 1] representing position evaluation
        """
        # Convert to PyTorch tensor
        # state is (27,) -> reshape to (1, 3, 3, 3) for model
        state_tensor = torch.from_numpy(state).reshape(1, 3, 3, 3).to(self.device)

        # Get model prediction
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)

        # Convert policy logits to probabilities
        policy_probs = torch.softmax(policy_logits, dim=1)

        # Extract and convert to NumPy
        policy = policy_probs.squeeze(0).cpu().numpy()  # (9,)
        value_scalar = value.item()  # float

        return policy, value_scalar

    def __repr__(self):
        return f"NNPredictorWrapper(device={self.device})"
