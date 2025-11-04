from config import Config as cfg

import torch
import numpy as np
from model import NeuralNetwork
from game import board_to_canonical_3d

device = "cuda" if torch.cuda.is_available() else "cpu"

class ValuePolicyNetwork:
    def __init__(self,path=None):
        self.model = NeuralNetwork().to(device)
        if path:
            try:
                self.model.load_state_dict(torch.load(path, map_location=device))
                print(f"Loaded model from {path}")
            except RuntimeError as e:
                print(f"Warning: Could not load model from {path} (architecture mismatch)")
                print("Using randomly initialized model instead")
         
        self.model.eval()
    
    def get_vp(self, state, player=1):
        """
        Get value and policy predictions for a board state.
        
        Args:
            state: Flat array of 9 values (from game.state)
            player: Current player (1 or -1) for canonical representation
        
        Returns:
            value: Position evaluation (float)
            policy: Move probabilities (array of 9 values)
        """
        # Convert to canonical 3-plane representation
        canonical_state = board_to_canonical_3d(state, player)
        
        # Convert to tensor and add batch dimension: (3, 3, 3) -> (1, 3, 3, 3)
        state_tensor = torch.from_numpy(canonical_state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            value, policy = self.model(state_tensor)
        
        value = value.cpu().numpy().flatten()[0]
        policy = torch.nn.functional.softmax(policy, dim=1)
        policy = policy.cpu().numpy().flatten()
        
        return value, policy




