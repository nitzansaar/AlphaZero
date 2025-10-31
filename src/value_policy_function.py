from config import Config as cfg

import torch
from model import NeuralNetwork
device = "cuda" if torch.cuda.is_available() else "cpu"

class ValuePolicyNetwork:
    def __init__(self,path=None):
        self.model = NeuralNetwork().to(device)
        if path:
            self.model.load_state_dict(torch.load(path))
         
        self.model.eval()
    def get_vp(self,state):
        state = state.reshape(1,cfg.ACTION_SIZE)
        state = torch.tensor(state,dtype=torch.float).to(device)
        
        with torch.no_grad():
            value,policy = self.model(state)
        value = value.cpu().numpy().flatten()[0]
        policy = torch.nn.functional.softmax(policy)
        policy = policy.cpu().numpy().flatten()
        
        return value,policy




