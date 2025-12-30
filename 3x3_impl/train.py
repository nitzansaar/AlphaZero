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

class Trainer:
    def __init__(self, modelpath=None, use_compile=True):
        os.makedirs(cfg.SAVE_MODEL_PATH, exist_ok = True)
        os.makedirs(cfg.LOGDIR,exist_ok = True)
        self.original_model = NeuralNetwork().to(device)  # Keep original for saving/loading
        
        # Helper function to strip _orig_mod prefix from state dict
        def strip_orig_mod(state_dict):
            """Remove _orig_mod. prefix from compiled model state dict keys"""
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            return new_state_dict

        self.modelpath = modelpath # use the existing model 
        self.latest_file_number = -1
        if modelpath:
            try:
                loaded_state = torch.load(modelpath, map_location=device)
                loaded_state = strip_orig_mod(loaded_state)  # Handle compiled models
                self.original_model.load_state_dict(loaded_state)
                print(f"Model successfully loaded from {modelpath}")
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
                        loaded_state = torch.load(latest_file, map_location=device)
                        loaded_state = strip_orig_mod(loaded_state)  # Handle compiled models
                        self.original_model.load_state_dict(loaded_state)
                        print("Model successfully loaded from {}".format(latest_file))
                    except RuntimeError as e:
                        print("Warning: Could not load model (architecture mismatch)")
                        print("This is expected if the model was trained with old architecture.")
                        print("Starting with new randomly initialized model")
                        self.latest_file_number = -1  # Start fresh
            else:
                savepath = os.path.join(cfg.SAVE_MODEL_PATH,cfg.BEST_MODEL.format(self.latest_file_number))
                torch.save(self.original_model.state_dict(), savepath)
                print("init.....Saving Model.....BL",savepath)

        # Compile model for faster execution (PyTorch 2.x feature for RTX 5090)
        # Do this AFTER loading so we save/load the uncompiled model
        if use_compile and device == "cuda" and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile for optimized execution...")
            self.model = torch.compile(self.original_model, mode="max-autotune")
            print("Model compilation complete!")
        else:
            self.model = self.original_model
        
        

        
    def load_data(self):
        ds = TrainingDataset()
        save_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)
        ds.load(save_path)
        # return all data as training data with augmentation enabled
        all_data = TicTacToeDataset(ds.training_dataset, use_augmentation=True)
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

        # AlphaGo Zero uses MSE for value and cross-entropy for policy
        # Policy loss: KL divergence between predicted policy and MCTS visit distribution
        value_criterion = nn.MSELoss().to(device)
        
        # Custom policy loss: cross-entropy with soft targets (MCTS visit distribution)
        def policy_loss_fn(pred_logits, target_probs):
            """Compute cross-entropy loss with soft targets (probability distribution)"""
            log_probs = torch.nn.functional.log_softmax(pred_logits, dim=1)
            # Cross-entropy: -sum(target_probs * log(pred_probs))
            loss = -torch.sum(target_probs * log_probs, dim=1).mean()
            return loss
        
        policy_criterion = policy_loss_fn
        
        # Use Adam optimizer with weight decay (L2 regularization) like AlphaGo Zero
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY
        )
        
        # Learning rate schedule: decay by factor of 0.1 at specific epochs
        # AlphaGo Zero uses step decay, but we'll use ReduceLROnPlateau for adaptive learning
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5, 
            patience=10,  # Increased patience
            threshold=0.0001, 
            threshold_mode='rel',
            cooldown=5, 
            min_lr=1e-6,  # Minimum learning rate
            eps=1e-08
        )

        # Mixed precision training for RTX 5090 (faster training, less memory)
        scaler = GradScaler() if (use_mixed_precision and device == "cuda") else None
        if scaler:
            print("Mixed precision training enabled (FP16/FP32)")

        best_loss = 1000
        history = []
        for epoch in range(cfg.EPOCHS):
            self.model.train()
            train_loss = 0
            train_vloss = 0
            train_aloss = 0
            for i, (X, v, p) in enumerate(train_dataloader): # iterate through the batch
                X = X.to(device, non_blocking=True) # board state (non_blocking for async transfer)
                v = v.to(device, non_blocking=True) # value target
                p = p.to(device, non_blocking=True) # policy target

                # Mixed precision forward pass & loss calculation
                if scaler:
                    with autocast():
                        yv, yp = self.model(X)
                        vloss = value_criterion(yv, v) # value loss
                        # Policy loss: cross-entropy with soft targets (MCTS visit distribution)
                        aloss = policy_criterion(yp, p) # policy loss
                        # Weighted combination like AlphaGo Zero
                        loss = cfg.VALUE_LOSS_WEIGHT * vloss + cfg.POLICY_LOSS_WEIGHT * aloss

                    train_loss += loss.item() # accumulate the loss
                    train_vloss += vloss.item()
                    train_aloss += aloss.item()

                    # Mixed precision backpropagation
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision training
                    yv, yp = self.model(X)
                    vloss = value_criterion(yv, v) # value loss
                    # Policy loss: cross-entropy with soft targets (MCTS visit distribution)
                    aloss = policy_criterion(yp, p) # policy loss
                    # Weighted combination like AlphaGo Zero
                    loss = cfg.VALUE_LOSS_WEIGHT * vloss + cfg.POLICY_LOSS_WEIGHT * aloss
                    train_loss += loss.item() # accumulate the loss
                    train_vloss += vloss.item()
                    train_aloss += aloss.item()

                    # backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            train_loss = train_loss / len(train_dataloader)
            train_vloss = train_vloss / len(train_dataloader)
            train_aloss = train_aloss / len(train_dataloader)

            # Save model based on training loss
            lr_scheduler.step(train_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # save the model based on the training loss
            if train_loss < best_loss:
                best_loss = train_loss
                current_iteration = self.latest_file_number + 1
                savepath = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(current_iteration))
                # Save the original uncompiled model, not the compiled one
                torch.save(self.original_model.state_dict(), savepath)
                print("Saving Model.....BL", savepath)
                # Store iteration number for evaluation script
                self.current_iteration = current_iteration
            print(f"Epoch {epoch}:: Total Loss: {train_loss:.6f}; Value Loss: {train_vloss:.6f}; Policy Loss: {train_aloss:.6f}; LR: {current_lr:.2e}")
            history.append([epoch, train_loss, train_vloss, train_aloss])
        
        history = pd.DataFrame(history,columns=["Epoch","Tr_Loss","Value_Loss","Policy_Loss"])
        current_iteration = self.latest_file_number + 1
        logpath = os.path.join(cfg.LOGDIR, "{}_history.csv".format(current_iteration))
        history.to_csv(logpath, index=None)
        print(history)
        
        # Store iteration number in a file for evaluation script
        iter_file = os.path.join(cfg.LOGDIR, "current_iteration.txt")
        with open(iter_file, 'w') as f:
            f.write(str(current_iteration))

        # evalutaion step doesnt exist yet

if __name__=="__main__":
    trainer = Trainer()
    trainer.train()