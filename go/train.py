import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from model import NeuralNetwork
from dataset import TrainingDataset, GoDataset
from config import Config as cfg
from shuffle import apply_shuffle, MIN_ROWS
from checkpoint_metadata import (
    checkpoint_iter,
    iteration_model_path,
    iteration_training_state_path,
    latest_checkpoint_path,
    load_model_metadata,
    save_model_metadata,
)
import pandas as pd
import argparse
from profiler import Timer

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} device")

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _checkpoint_iter(path):
    """Return checkpoint iteration number, or None for non-model .pt files."""
    return checkpoint_iter(path)


def _atomic_torch_save(state_dict, path):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_save_object(obj, path):
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_text_save(text, path):
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _global_step_samples(global_minibatch_step):
    return int(global_minibatch_step) * int(cfg.BATCH_SIZE)


def _latest_selfplay_avg_moves():
    timing_path = os.path.join(cfg.LOGDIR, "selfplay_timing.json")
    try:
        with open(timing_path) as f:
            metrics = json.load(f).get("metrics", {})
    except (FileNotFoundError, OSError, ValueError):
        return None
    avg_moves = metrics.get("avg_selfplay_game_moves")
    if avg_moves is None:
        return None
    return float(avg_moves)


def _save_model_metadata(model_path, iteration, global_minibatch_step, total_num_data_rows):
    return save_model_metadata(
        model_path,
        iteration=iteration,
        global_minibatch_step=global_minibatch_step,
        global_step_samples=_global_step_samples(global_minibatch_step),
        total_num_data_rows=total_num_data_rows,
        train_steps_per_iteration=cfg.TRAIN_STEPS,
        batch_size=cfg.BATCH_SIZE,
        avg_selfplay_game_moves=_latest_selfplay_avg_moves(),
    )


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
        self.training_state = {}
        def save_initial_model():
            self.latest_file_number = 0
            savepath = iteration_model_path(cfg.SAVE_MODEL_PATH, self.latest_file_number)
            _atomic_torch_save(self.original_model.state_dict(), savepath)
            metadata_path, _ = _save_model_metadata(
                savepath,
                iteration=self.latest_file_number,
                global_minibatch_step=0,
                total_num_data_rows=0,
            )
            print("init.....Saving Model.....BL",savepath)
            print(f"init.....Saving Model metadata.....BL {metadata_path}")

        if modelpath:
            parsed_iter = _checkpoint_iter(modelpath)
            if parsed_iter is not None:
                self.latest_file_number = parsed_iter
                self.training_state = self._load_training_state(parsed_iter)
            loaded_state = torch.load(modelpath, map_location=device)
            loaded_state = strip_orig_mod(loaded_state)  # Handle compiled models
            self.original_model.load_state_dict(loaded_state)
            print(f"Model successfully loaded from {modelpath}")
        else:
            latest_iter, latest_file = latest_checkpoint_path(cfg.SAVE_MODEL_PATH)
            if latest_file is not None:
                self.latest_file_number = latest_iter
                self.training_state = self._load_training_state(latest_iter)
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
                save_initial_model()

        # Compile model for faster execution (PyTorch 2.x feature for RTX 5090)
        # Do this AFTER loading so we save/load the uncompiled model
        if use_compile and device == "cuda" and hasattr(torch, 'compile'):
            print("Compiling model with torch.compile for optimized execution...")
            self.model = torch.compile(self.original_model, mode="max-autotune")
            print("Model compilation complete!")
        else:
            self.model = self.original_model

    def _load_training_state(self, iteration):
        state_path = iteration_training_state_path(cfg.SAVE_MODEL_PATH, iteration)
        if os.path.exists(state_path):
            return torch.load(state_path, map_location=device, weights_only=False)
        metadata = load_model_metadata(iteration_model_path(cfg.SAVE_MODEL_PATH, iteration))
        if metadata:
            return {
                "global_minibatch_step": int(metadata.get("global_minibatch_step", 0)),
            }
        return {}

    def _save_training_state(self, iteration, optimizer, scaler, scheduler, global_minibatch_step):
        state_path = iteration_training_state_path(cfg.SAVE_MODEL_PATH, iteration)
        _atomic_save_object({
            "iteration": int(iteration),
            "global_minibatch_step": int(global_minibatch_step),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, state_path)
        print(f"Saved training state: {state_path}")
    def load_data(self):
        ds = TrainingDataset()
        save_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)
        if not os.path.exists(save_path) and save_path.endswith(".npz"):
            legacy_path = save_path[:-4] + ".pkl"
            if os.path.exists(legacy_path):
                save_path = legacy_path
        if not os.path.exists(save_path):
            return []
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
            self.current_iteration = max(self.latest_file_number, 0)
            iter_file = os.path.join(cfg.LOGDIR, "current_iteration.txt")
            _atomic_text_save(str(self.current_iteration), iter_file)
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
            num_workers=cfg.DATA_LOADER_WORKERS,
            pin_memory=True,
            persistent_workers=cfg.DATA_LOADER_WORKERS > 0,
        )

        value_criterion = nn.MSELoss().to(device)

        def policy_loss_fn(pred_logits, target_probs):
            log_probs = torch.nn.functional.log_softmax(pred_logits, dim=1)
            loss = -torch.sum(target_probs * log_probs, dim=1).mean()
            return loss

        policy_criterion = policy_loss_fn

        current_iter = self.latest_file_number + 1
        global_step = int(self.training_state.get(
            "global_minibatch_step",
            max(self.latest_file_number, 0) * cfg.TRAIN_STEPS,
        ))
        print(f"Iteration {current_iter}: global minibatch step={global_step}, LR={cfg.LEARNING_RATE:.2e}")

        optimizer = torch.optim.Adam(
            self.original_model.parameters(),
            lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY,
        )
        if "optimizer" in self.training_state:
            try:
                optimizer.load_state_dict(self.training_state["optimizer"])
                for group in optimizer.param_groups:
                    group["lr"] = cfg.LEARNING_RATE
            except Exception:
                print("Could not load optimizer state (incompatible format); starting fresh.")

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=5,
            min_lr=1e-6,
        )
        if "scheduler" in self.training_state:
            try:
                lr_scheduler.load_state_dict(self.training_state["scheduler"])
            except Exception:
                print("Could not load scheduler state; starting fresh.")

        amp_enabled = use_mixed_precision and device == "cuda"
        scaler = GradScaler("cuda" if device == "cuda" else "cpu", enabled=amp_enabled)
        if "scaler" in self.training_state:
            scaler.load_state_dict(self.training_state["scaler"])

        best_loss = float('inf')
        best_state_dict = None
        history = []

        total_steps = 0
        for epoch in range(cfg.EPOCHS):
            self.model.train()
            train_loss = 0
            train_vloss = 0
            train_aloss = 0

            for X, v, p in train_dataloader:
                with timer.track("data_transfer"):
                    X = X.to(device, non_blocking=True)
                    v = v.to(device, non_blocking=True)
                    p = p.to(device, non_blocking=True)

                with timer.track("forward_pass"):
                    with autocast(device, enabled=amp_enabled):
                        yv, yp = self.model(X)
                        vloss_raw = value_criterion(yv.squeeze(-1), v)
                        aloss_raw = policy_criterion(yp, p)
                        loss = vloss_raw * cfg.VALUE_LOSS_WEIGHT + aloss_raw * cfg.POLICY_LOSS_WEIGHT

                train_loss += loss.item()
                train_vloss += vloss_raw.item()
                train_aloss += aloss_raw.item()

                with timer.track("backward_pass"):
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()

                with timer.track("optimizer_step"):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.original_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                total_steps += 1

            train_loss /= len(train_dataloader)
            train_vloss /= len(train_dataloader)
            train_aloss /= len(train_dataloader)

            lr_scheduler.step(train_loss)
            current_lr = optimizer.param_groups[0]['lr']

            if train_loss < best_loss:
                best_loss = train_loss
                best_state_dict = {k: val.cpu().clone() for k, val in self.original_model.state_dict().items()}

            print(f"Epoch {epoch}: Total Loss: {train_loss:.6f}; "
                  f"Value Loss: {train_vloss:.6f}; "
                  f"Policy Loss: {train_aloss:.6f}; "
                  f"LR: {current_lr:.2e}")
            history.append([epoch, train_loss, train_vloss, train_aloss])

        current_iteration = self.latest_file_number + 1
        global_minibatch_step_after = global_step + total_steps
        total_num_data_rows = len(raw_dataset)
        savepath = iteration_model_path(cfg.SAVE_MODEL_PATH, current_iteration)
        state_to_save = best_state_dict if best_state_dict is not None else self.original_model.state_dict()
        with timer.track("model_save"):
            _atomic_torch_save(state_to_save, savepath)
            self._save_training_state(
                current_iteration,
                optimizer,
                scaler,
                lr_scheduler,
                global_minibatch_step_after,
            )
            metadata_path, metadata = _save_model_metadata(
                savepath,
                iteration=current_iteration,
                global_minibatch_step=global_minibatch_step_after,
                total_num_data_rows=total_num_data_rows,
            )
        print(f"Saved model (best loss={best_loss:.6f}): {savepath}")
        print(f"Saved model metadata: {metadata_path}")
        print(f"  global_step_samples={metadata['global_step_samples']}  "
              f"total_num_data_rows={metadata['total_num_data_rows']}")
        self.current_iteration = current_iteration

        timer.print_summary("Training Timing")
        timing_path = os.path.join(cfg.LOGDIR, "train_timing.json")
        timer.save(timing_path)
        print(f"Timing data saved to {timing_path}")

        history = pd.DataFrame(history, columns=["Epoch", "Tr_Loss", "Value_Loss", "Policy_Loss"])
        logpath = os.path.join(cfg.LOGDIR, "{}_history.csv".format(current_iteration))
        history.to_csv(logpath, index=None)
        print(history)

        # Store iteration number in a file for evaluation script
        iter_file = os.path.join(cfg.LOGDIR, "current_iteration.txt")
        _atomic_text_save(str(current_iteration), iter_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Go neural network')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    args = parser.parse_args()

    trainer = Trainer(use_compile=not args.no_compile)
    trainer.train()
