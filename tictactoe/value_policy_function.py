from config import Config as cfg

import torch
import numpy as np
from model import NeuralNetwork
from game import board_to_canonical_3d

device = "cuda" if torch.cuda.is_available() else "cpu"

# RTX 5090 Optimizations for inference
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

class ValuePolicyNetwork:
    def __init__(self, path=None, use_compile=True):
        self.original_model = NeuralNetwork().to(device)
        
        if path:
            try:
                loaded_state = torch.load(path, map_location=device)
                # Handle models saved with _orig_mod prefix (from torch.compile)
                if any(key.startswith('_orig_mod.') for key in loaded_state.keys()):
                    # Strip _orig_mod prefix
                    new_state_dict = {}
                    for key, value in loaded_state.items():
                        if key.startswith('_orig_mod.'):
                            new_key = key[len('_orig_mod.'):]
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    loaded_state = new_state_dict
                
                self.original_model.load_state_dict(loaded_state)
                print(f"Loaded model from {path}")
            except RuntimeError as e:
                print(f"Warning: Could not load model from {path} (architecture mismatch)")
                print(f"Error details: {e}")
                print("Using randomly initialized model instead")

        self.original_model.eval()

        # Compile model for faster inference on RTX 5090
        # Do this AFTER loading so we load the uncompiled model
        if use_compile and device == "cuda" and hasattr(torch, 'compile'):
            print("Compiling inference model with torch.compile...")
            self.model = torch.compile(self.original_model, mode="reduce-overhead")
            print("Inference model compilation complete!")
        else:
            self.model = self.original_model
    
    def get_vp(self, state, player=1):
        """
        Get value and policy predictions for a board state.

        Args:
            state: Flat array of 81 values (from game.state)
            player: Current player (1 or -1) for canonical representation

        Returns:
            value: Position evaluation (float)
            policy: Move probabilities (array of 81 values)
        """
        # Convert to canonical 3-plane representation
        canonical_state = board_to_canonical_3d(state, player)

        # Convert to tensor and add batch dimension: (3, 9, 9) -> (1, 3, 9, 9)
        state_tensor = torch.from_numpy(canonical_state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            value, policy = self.model(state_tensor)
        
        value = value.cpu().numpy().flatten()[0]
        policy = torch.nn.functional.softmax(policy, dim=1)
        policy = policy.cpu().numpy().flatten()
        
        return value, policy




