import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import NeuralNetwork
from dataset import TrainingDataset, TicTacToeDataset
from config import Config as cfg
from game import TicTacToe
from glob import glob
import pandas as pd
from value_policy_function import ValuePolicyNetwork
from mcts import MonteCarloTreeSearch
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Trainer:
    def __init__(self,modelpath=None):
        os.makedirs(cfg.SAVE_MODEL_PATH,exist_ok=True)
        os.makedirs(cfg.LOGDIR,exist_ok=True)
        self.model = NeuralNetwork().to(device)
        self.latest_file_number = -1
        
        # Always start fresh - save initial model
        savepath = os.path.join(cfg.SAVE_MODEL_PATH,cfg.BEST_MODEL.format(self.latest_file_number))
        torch.save(self.model.state_dict(), savepath)
        print("Starting fresh. Initialized model saved to:", savepath)
            
        self.train_data, self.eval_data = self.load_data()
        
        

        
    def load_data(self):
        ds = TrainingDataset()
        save_path = os.path.join(cfg.SAVE_PICKLES,cfg.DATASET_PATH)
        ds.load(save_path)
        # Return all data as training data, empty eval dataset
        all_data = TicTacToeDataset(ds.training_dataset)
        empty_eval = TicTacToeDataset([])
        return all_data, empty_eval

    def train(self):
        train_dataloader = DataLoader(self.train_data,\
                        batch_size=cfg.BATCH_SIZE,\
                        shuffle=True)
        value_criterion = nn.MSELoss().to(device)
        policy_criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(self.model.parameters())
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',\
                factor=0.5, patience=5, threshold=0.0001, threshold_mode='rel',\
                     cooldown=0, min_lr=0, eps=1e-08)
        best_loss = 1000
        history = []
        for epoch in range(cfg.EPOCHS):
            self.model.train()
            train_loss = 0
            for i, (X, v, p) in enumerate(train_dataloader): # iterate through the batch
                X = X.to(device) # board state
                v = v.to(device) # value target 
                p = p.to(device) # policy target

                # forward pass & loss calculation
                yv, yp = self.model(X)
                vloss = value_criterion(yv, v) # value loss
                aloss = policy_criterion(yp, p) # policy loss

                loss = vloss + aloss
                train_loss += loss.item() # accumulate the loss

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss = train_loss / len(train_dataloader)

            # Save model based on training loss
            lr_scheduler.step(train_loss)
            if train_loss < best_loss:
                best_loss = train_loss
                savepath = os.path.join(cfg.SAVE_MODEL_PATH,cfg.BEST_MODEL.format(self.latest_file_number+1))
                torch.save(self.model.state_dict(), savepath)
                print("Saving Model.....BL",savepath)
            print(f"Epoch {epoch}:: Train Loss: {train_loss};")
            history.append([epoch,train_loss])
        
        history = pd.DataFrame(history,columns=["Epoch","Tr_Loss"])
        logpath = os.path.join(cfg.LOGDIR,"{}_history.csv".format(self.latest_file_number+1))
        history.to_csv(logpath,index=None)
        print(history)

        # evalutaion step

if __name__=="__main__":
    trainer = Trainer()
    trainer.train()