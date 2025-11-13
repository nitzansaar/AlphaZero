import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from model import NeuralNetwork
from dataset import TrainingDataset, TicTacToeDataset
from config import Config as cfg
from glob import glob
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# RTX 5090 Optimizations
if device == "cuda":
    # Enable TensorFloat-32 for faster matrix multiplications on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cuDNN autotuner for optimal convolution algorithms
    torch.backends.cudnn.benchmark = True
    print("GPU Optimizations Enabled:")
    print(f"  - TF32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"  - cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class Trainer:
    def __init__(self, modelpath=None, use_compile=True):
        os.makedirs(cfg.SAVE_MODEL_PATH, exist_ok = True)
        os.makedirs(cfg.LOGDIR,exist_ok = True)
        self.model = NeuralNetwork().to(device)

        # Compile model for faster execution (PyTorch 2.x feature for RTX 5090)
        if use_compile and device == "cuda" and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile for optimized execution...")
            self.model = torch.compile(self.model, mode="max-autotune")
            print("Model compilation complete!")

        self.modelpath = modelpath
        self.latest_file_number = -1
        if modelpath:
            try:
                self.model.load_state_dict(torch.load(modelpath, map_location=device)) # load the model from the modelpath
                print(f"Loaded model from {modelpath}")
            except RuntimeError as e:
                print(f"Warning: Could not load model from {modelpath}")
                print(f"Error: {e}")
                print("Starting with new randomly initialized model")
        else:
            all_models = glob(cfg.SAVE_MODEL_PATH + "/*.pt")
            if len(all_models) > 0: # if there are any models in the save model path
                files = [int(os.path.basename(f).split("_")[0]) for f in all_models if os.path.basename(f).split("_")[0].isdigit()]
                if files:
                    self.latest_file_number = max(files) # get the latest file number
                    latest_file = os.path.join(cfg.SAVE_MODEL_PATH,cfg.BEST_MODEL.format(self.latest_file_number))
                    print("Attempting to load latest model: {}".format(latest_file))
                    try:
                        self.model.load_state_dict(torch.load(latest_file, map_location=device))
                        print("Successfully loaded model from {}".format(latest_file))
                    except RuntimeError as e:
                        print("Warning: Could not load model (architecture mismatch)")
                        print("This is expected if the model was trained with old architecture.")
                        print("Starting with new randomly initialized model")
                        self.latest_file_number = -1  # Start fresh
            else:
                savepath = os.path.join(cfg.SAVE_MODEL_PATH,cfg.BEST_MODEL.format(self.latest_file_number))
                torch.save(self.model.state_dict(), savepath)
                print("init.....Saving Model.....BL",savepath)
        
        

        
    def load_data(self):
        ds = TrainingDataset()
        save_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)
        ds.load(save_path)
        # return all data as training data
        all_data = TicTacToeDataset(ds.training_dataset)
        # empty_eval = TicTacToeDataset([])
        return all_data

    def train(self, use_mixed_precision=True):
        self.train_data = self.load_data()

        # Optimize DataLoader for RTX 5090
        train_dataloader = DataLoader(
            self.train_data,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster data transfer to GPU
            persistent_workers=True  # Keep workers alive between epochs
        )

        value_criterion = nn.MSELoss().to(device)
        policy_criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(self.model.parameters())
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',\
                factor=0.5, patience=5, threshold=0.0001, threshold_mode='rel',\
                     cooldown=0, min_lr=0, eps=1e-08)

        # Mixed precision training for RTX 5090 (faster training, less memory)
        scaler = GradScaler() if (use_mixed_precision and device == "cuda") else None
        if scaler:
            print("Mixed precision training enabled (FP16/FP32)")

        best_loss = 1000
        history = []
        for epoch in range(cfg.EPOCHS):
            self.model.train()
            train_loss = 0
            for i, (X, v, p) in enumerate(train_dataloader): # iterate through the batch
                X = X.to(device, non_blocking=True) # board state (non_blocking for async transfer)
                v = v.to(device, non_blocking=True) # value target
                p = p.to(device, non_blocking=True) # policy target

                # Mixed precision forward pass & loss calculation
                if scaler:
                    with autocast():
                        yv, yp = self.model(X)
                        vloss = value_criterion(yv, v) # value loss
                        aloss = policy_criterion(yp, p) # policy loss
                        loss = vloss + aloss

                    train_loss += loss.item() # accumulate the loss

                    # Mixed precision backpropagation
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision training
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
            # save the model based on the training loss
            if train_loss < best_loss:
                best_loss = train_loss
                savepath = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(self.latest_file_number + 1))
                torch.save(self.model.state_dict(), savepath)
                print("Saving Model.....BL", savepath)
            print(f"Epoch {epoch}:: Train Loss: {train_loss};")
            history.append([epoch, train_loss])
        
        history = pd.DataFrame(history,columns=["Epoch","Tr_Loss"])
        logpath = os.path.join(cfg.LOGDIR, "{}_history.csv".format(self.latest_file_number + 1))
        history.to_csv(logpath, index=None)
        print(history)

        # evalutaion step

if __name__=="__main__":
    trainer = Trainer()
    trainer.train()