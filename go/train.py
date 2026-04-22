import os
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from model import NeuralNetwork
from dataset import TrainingDataset, GoDataset
from config import Config as cfg
from shuffle import apply_shuffle, MIN_ROWS
from glob import glob
import pandas as pd
import argparse
from profiler import Timer

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
                        raise RuntimeError(
                            f"Cannot load latest checkpoint '{latest_file}': {e}\n"
                            "Refusing to overwrite existing checkpoints with a freshly "
                            "initialised model.  Delete or rename the corrupt checkpoint "
                            "to start a new run from scratch."
                        ) from e
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
        return ds.training_dataset

    def train(self, use_mixed_precision=True):
        timer = Timer()

        with timer.track("load_data"):
            raw_dataset = self.load_data()

        # KataGo-style shuffle: enforce minimum row count and apply temporal taper
        # so that newer self-play data drives the gradient signal more than old data.
        shuffled_rows, sample_weights = apply_shuffle(raw_dataset)
        if shuffled_rows is None:
            print(f"Insufficient data: {len(raw_dataset)} rows < {MIN_ROWS} minimum. "
                  "Skipping training this cycle.")
            return

        n_rows = len(shuffled_rows)
        print(f"Shuffle: using {n_rows} rows (of {len(raw_dataset)} total); "
              f"weight range [{sample_weights[0]:.2f}, {sample_weights[-1]:.2f}]")

        self.train_data = GoDataset(shuffled_rows, use_augmentation=True)

        # WeightedRandomSampler applies the temporal taper at sample time.
        # replacement=True is required for WeightedRandomSampler.
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=n_rows,
            replacement=True,
        )

        train_dataloader = DataLoader(
            self.train_data,
            batch_size=cfg.BATCH_SIZE,
            sampler=sampler,   # replaces shuffle=True
            num_workers=8,
            pin_memory=True,
            persistent_workers=False,
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

        # Compute LR for this iteration using step-decay schedule
        current_iter = self.latest_file_number + 1
        lr = cfg.LEARNING_RATE
        for decay_iter in cfg.LR_DECAY_ITERS:
            if current_iter >= decay_iter:
                lr *= cfg.LR_DECAY_FACTOR
        print(f"Iteration {current_iter}: using LR={lr:.2e}")

        # Policy head gets a higher LR (POLICY_LR_MULTIPLIER) to accelerate
        # policy learning, which lags behind value learning in self-play.
        policy_modules = [
            self.original_model.policy_conv,
            self.original_model.policy_bn,
            self.original_model.policy_fc,
        ]
        policy_params = []
        for m in policy_modules:
            policy_params.extend(m.parameters())
        policy_param_ids = {id(p) for p in policy_params}
        backbone_params = [p for p in self.original_model.parameters()
                           if id(p) not in policy_param_ids]

        policy_lr = lr * cfg.POLICY_LR_MULTIPLIER
        print(f"  Backbone/value LR: {lr:.2e}  |  Policy head LR: {policy_lr:.2e}")

        optimizer = torch.optim.SGD(
            [
                {'params': backbone_params},
                {'params': policy_params, 'lr': policy_lr},
            ],
            lr=lr,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY,
            nesterov=True,
        )

        # Mixed precision training (AMP); falls back to no-op on CPU.
        scaler = GradScaler(device)

        history = []

        self.model.train()
        data_iter = iter(train_dataloader)
        for step in range(cfg.TRAIN_STEPS):
            try:
                X, v, p = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                X, v, p = next(data_iter)

            with timer.track("data_transfer"):
                X = X.to(device, non_blocking=True)
                v = v.to(device, non_blocking=True)
                p = p.to(device, non_blocking=True)

            with timer.track("forward_pass"):
                with autocast(device):
                    yv, yp = self.model(X)
                    vloss_raw = value_criterion(yv.squeeze(-1), v)
                    aloss_raw = policy_criterion(yp, p)
                    # Scale value loss to match policy cross-entropy magnitude
                    # (~0.5 raw vs ~5.0 raw) so both heads drive the backbone equally.
                    loss = vloss_raw * cfg.VALUE_LOSS_WEIGHT + aloss_raw * cfg.POLICY_LOSS_WEIGHT

            with timer.track("backward_pass"):
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

            with timer.track("optimizer_step"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            train_loss = loss.item()
            print(f"Step {step}: Total Loss: {train_loss:.6f}; "
                  f"Value Loss (raw): {vloss_raw.item():.6f}; "
                  f"Policy Loss (raw): {aloss_raw.item():.6f}")
            history.append([step, train_loss, vloss_raw.item(), aloss_raw.item()])

        # Always save the model after completing all steps
        current_iteration = self.latest_file_number + 1
        savepath = os.path.join(cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(current_iteration))
        with timer.track("model_save"):
            torch.save(self.original_model.state_dict(), savepath)
        print(f"Saved model: {savepath}")
        self.current_iteration = current_iteration

        timer.print_summary("Training Timing")
        timing_path = os.path.join(cfg.LOGDIR, "train_timing.json")
        timer.save(timing_path)
        print(f"Timing data saved to {timing_path}")

        history = pd.DataFrame(history, columns=["Step", "Tr_Loss", "Value_Loss", "Policy_Loss"])
        logpath = os.path.join(cfg.LOGDIR, "{}_history.csv".format(current_iteration))
        history.to_csv(logpath, index=None)
        print(history)

        # Store iteration number in a file for evaluation script
        iter_file = os.path.join(cfg.LOGDIR, "current_iteration.txt")
        with open(iter_file, 'w') as f:
            f.write(str(current_iteration))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Go neural network')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    args = parser.parse_args()

    trainer = Trainer(use_compile=not args.no_compile)
    trainer.train()