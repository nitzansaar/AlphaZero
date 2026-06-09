import os
from glob import glob

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from torch.amp import autocast, GradScaler
    _AMP_NEW = True
except ImportError:  # older torch
    from torch.cuda.amp import autocast, GradScaler
    _AMP_NEW = False

from config import Config as cfg
from model import NeuralNetwork
from dataset import TrainingDataset, ChessDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _strip_orig_mod(state_dict):
    new_state = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_state[key[len("_orig_mod."):]] = value
        else:
            new_state[key] = value
    return new_state


class Trainer:
    def __init__(self, modelpath=None, use_compile=True):
        os.makedirs(cfg.SAVE_MODEL_PATH, exist_ok=True)
        os.makedirs(cfg.LOGDIR, exist_ok=True)
        self.original_model = NeuralNetwork().to(device)
        self.modelpath = modelpath
        self.latest_file_number = -1

        if modelpath:
            self._try_load(modelpath)
        else:
            all_models = glob(os.path.join(cfg.SAVE_MODEL_PATH, "*.pt"))
            nums = [
                int(os.path.basename(f).split("_")[0])
                for f in all_models
                if os.path.basename(f).split("_")[0].lstrip("-").isdigit()
            ]
            if nums:
                self.latest_file_number = max(nums)
                latest_file = os.path.join(
                    cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(self.latest_file_number)
                )
                print(f"Attempting to load latest model: {latest_file}")
                if not self._try_load(latest_file):
                    self.latest_file_number = -1
            else:
                savepath = os.path.join(
                    cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(self.latest_file_number)
                )
                torch.save(self.original_model.state_dict(), savepath)
                print(f"init.....Saving Model.....BL {savepath}")

        if use_compile and device == "cuda" and hasattr(torch, "compile"):
            print("Compiling model with torch.compile for optimized execution...")
            self.model = torch.compile(self.original_model, mode="max-autotune")
            print("Model compilation complete!")
        else:
            self.model = self.original_model

    def _try_load(self, path):
        try:
            loaded_state = torch.load(path, map_location=device)
            loaded_state = _strip_orig_mod(loaded_state)
            self.original_model.load_state_dict(loaded_state)
            print(f"Model successfully loaded from {path}")
            return True
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Warning: Could not load model from {path}")
            print(f"Error: {e}")
            print("Starting with new randomly initialized model")
            return False

    def load_data(self):
        ds = TrainingDataset()
        save_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)
        ds.load(save_path)
        return ChessDataset(ds.training_dataset)

    def train(self):
        self.train_data = self.load_data()

        train_dataloader = DataLoader(
            self.train_data,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=(device == "cuda"),
            persistent_workers=True,
        )

        value_criterion = nn.MSELoss().to(device)

        def policy_loss_fn(pred_logits, target_probs):
            log_probs = torch.nn.functional.log_softmax(pred_logits, dim=1)
            return -torch.sum(target_probs * log_probs, dim=1).mean()

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg.LEARNING_RATE,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
            threshold=0.0001, threshold_mode="rel", cooldown=5, min_lr=1e-6, eps=1e-08,
        )

        scaler = GradScaler() if not _AMP_NEW else GradScaler(device)
        use_amp = device == "cuda"
        print(f"Mixed precision training: {use_amp}")

        best_loss = float("inf")
        history = []
        for epoch in range(cfg.EPOCHS):
            self.model.train()
            train_loss = train_vloss = train_aloss = 0.0
            for X, v, p in train_dataloader:
                X = X.to(device, non_blocking=True)
                v = v.to(device, non_blocking=True)
                p = p.to(device, non_blocking=True)

                if use_amp:
                    ctx = autocast("cuda") if _AMP_NEW else autocast()
                else:
                    ctx = _nullcontext()
                with ctx:
                    yv, yp = self.model(X)
                    vloss = value_criterion(yv, v)
                    aloss = policy_loss_fn(yp, p)
                    loss = cfg.VALUE_LOSS_WEIGHT * vloss + cfg.POLICY_LOSS_WEIGHT * aloss

                train_loss += loss.item()
                train_vloss += vloss.item()
                train_aloss += aloss.item()

                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            n_batches = max(1, len(train_dataloader))
            train_loss /= n_batches
            train_vloss /= n_batches
            train_aloss /= n_batches

            lr_scheduler.step(train_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            if train_loss < best_loss:
                best_loss = train_loss
                current_iteration = self.latest_file_number + 1
                savepath = os.path.join(
                    cfg.SAVE_MODEL_PATH, cfg.BEST_MODEL.format(current_iteration)
                )
                # Atomic save: write to a temp file then rename, so a failed
                # write (e.g. disk full) can't truncate/corrupt an existing good
                # checkpoint.  torch.save truncates its target before writing, so
                # writing in place would destroy the previous model on failure.
                tmppath = savepath + ".tmp"
                try:
                    torch.save(self.original_model.state_dict(), tmppath)
                    os.replace(tmppath, savepath)
                except Exception:
                    if os.path.exists(tmppath):
                        os.remove(tmppath)
                    raise
                print(f"Saving Model.....BL {savepath}")
                self.current_iteration = current_iteration

            print(
                f"Epoch {epoch}:: Total Loss: {train_loss:.6f}; "
                f"Value Loss: {train_vloss:.6f}; Policy Loss: {train_aloss:.6f}; "
                f"LR: {current_lr:.2e}"
            )
            history.append([epoch, train_loss, train_vloss, train_aloss])

        history = pd.DataFrame(history, columns=["Epoch", "Tr_Loss", "Value_Loss", "Policy_Loss"])
        current_iteration = self.latest_file_number + 1
        logpath = os.path.join(cfg.LOGDIR, f"{current_iteration}_history.csv")
        history.to_csv(logpath, index=None)
        print(history)

        iter_file = os.path.join(cfg.LOGDIR, "current_iteration.txt")
        with open(iter_file, "w") as f:
            f.write(str(current_iteration))


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
