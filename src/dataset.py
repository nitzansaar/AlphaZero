import torch
import numpy as np
import pickle
from copy import copy
from config import Config as cfg

class TicTacToeDataset:
    def __init__(self,dataset):
        self.data = dataset
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        datapoint = self.data[index]
        state = datapoint[0]
        v = datapoint[3]
        p = datapoint[1]
        return torch.tensor(state,dtype=torch.float),torch.tensor(v,dtype=torch.float),torch.tensor(p,dtype=torch.float)

class TrainingDataset:
    def __init__(self):
        self.training_dataset = [] # list of tuples (state,value,policy)
    
    def calculate_values(self,dataset,winner): # assign value to each position in the dataset based on the winner
        for ind,step in enumerate(dataset):
            step_ = copy(step)
            step_player = step_[2]
            if winner == 0: # draw
                value = 0
            else:
                if winner==step_player:
                    value = 1
                else:
                    value =-1
            step_.append(value)
            dataset[ind] = step_
        return dataset
    def add_game_to_training_dataset(self,dataset,winner): # add the completed game data to the training dataset
        data = self.calculate_values(dataset,winner)
        self.training_dataset.extend(data)
        self.training_dataset = self.training_dataset[-1*cfg.DATASET_QUEUE_SIZE:]    
    
    def save(self,path): # save the training dataset to a pickle file
#         pickle.dump(self.training_dataset,path)
        with open(path, 'wb') as handle:
            pickle.dump(self.training_dataset,handle)
    def load(self,path): # load the training dataset from a pickle file
#         self.training_dataset = pickle.load(path)
        with open(path, 'rb') as handle:
            self.training_dataset = pickle.load(handle)
    def retreive_test_train_data(self): 
        data = self.training_dataset
        num_samples = len(data)
        train_idx = np.random.choice(np.arange(num_samples),int(cfg.TRAIN_TEST_SPLIT*num_samples),replace=False)
        train_idx_set = set(train_idx)
        val_idx = [t for t in range(num_samples) if t not in train_idx_set]
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        return TicTacToeDataset(train_data),TicTacToeDataset(val_data)
    