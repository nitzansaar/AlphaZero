import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
from contextlib import nullcontext
from model import NeuralNetwork
from dataset import TrainingDataset, GoDataset
from config import Config as cfg
from glob import glob
import pandas as pd
import argparse

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
        all_data = GoDataset(ds.training_dataset, use_augmentation=True)
        # empty_eval = TicTacToeDataset([])
        return all_data

    def train(self, use_mixed_precision=True, enable_profiling=False, profile_epochs=2):
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

        # Timing statistics
        self.timing_stats = {
            'data_loading': [],
            'forward_pass': [],
            'loss_computation': [],
            'backward_pass': [],
            'optimizer_step': [],
            'epoch_total': [],
        }

        if enable_profiling:
            print(f"\nProfiling enabled for first {profile_epochs} epochs")
            print("=" * 60)

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
        scaler = GradScaler('cuda')

        best_loss = 1000
        history = []

        # Setup PyTorch profiler if enabled
        profiler_context = None
        if enable_profiling:
            os.makedirs(os.path.join(cfg.LOGDIR, "profiler"), exist_ok=True)
            profiler_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=tensorboard_trace_handler(os.path.join(cfg.LOGDIR, "profiler")),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            profiler_context.__enter__()

        for epoch in range(cfg.EPOCHS):
            epoch_start = time.perf_counter()
            self.model.train()
            train_loss = 0
            train_vloss = 0
            train_aloss = 0

            # Per-epoch timing accumulators
            epoch_data_time = 0
            epoch_forward_time = 0
            epoch_loss_time = 0
            epoch_backward_time = 0
            epoch_optim_time = 0

            data_start = time.perf_counter()
            for i, (X, v, p) in enumerate(train_dataloader): # iterate through the batch
                # Data loading time
                data_end = time.perf_counter()
                epoch_data_time += data_end - data_start

                # Transfer to GPU
                transfer_start = time.perf_counter()
                X = X.to(device, non_blocking=True) # board state
                v = v.to(device, non_blocking=True) # value target
                p = p.to(device, non_blocking=True) # policy target
                if device == "cuda":
                    torch.cuda.synchronize()

                # Forward pass
                forward_start = time.perf_counter()
                with autocast('cuda'):
                    with record_function("forward_pass") if enable_profiling else nullcontext():
                        yv, yp = self.model(X)

                    # Loss computation
                    loss_start = time.perf_counter()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    epoch_forward_time += loss_start - forward_start

                    with record_function("loss_computation") if enable_profiling else nullcontext():
                        vloss = value_criterion(yv.squeeze(-1), v) # value loss
                        aloss = policy_criterion(yp, p) # policy loss
                        loss = cfg.VALUE_LOSS_WEIGHT * vloss + cfg.POLICY_LOSS_WEIGHT * aloss

                backward_start = time.perf_counter()
                if device == "cuda":
                    torch.cuda.synchronize()
                epoch_loss_time += backward_start - loss_start

                train_loss += loss.item() # accumulate the loss
                train_vloss += vloss.item()
                train_aloss += aloss.item()

                # Mixed precision backpropagation with gradient clipping
                with record_function("backward_pass") if enable_profiling else nullcontext():
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()

                optim_start = time.perf_counter()
                if device == "cuda":
                    torch.cuda.synchronize()
                epoch_backward_time += optim_start - backward_start

                # Gradient clipping to prevent exploding gradients (important for ResNets)
                with record_function("optimizer_step") if enable_profiling else nullcontext():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                optim_end = time.perf_counter()
                if device == "cuda":
                    torch.cuda.synchronize()
                epoch_optim_time += optim_end - optim_start

                # Step profiler
                if enable_profiling and profiler_context is not None:
                    profiler_context.step()

                # Start timing next data load
                data_start = time.perf_counter()

            train_loss = train_loss / len(train_dataloader)
            train_vloss = train_vloss / len(train_dataloader)
            train_aloss = train_aloss / len(train_dataloader)

            epoch_end = time.perf_counter()
            epoch_total = epoch_end - epoch_start

            # Store timing stats
            self.timing_stats['data_loading'].append(epoch_data_time)
            self.timing_stats['forward_pass'].append(epoch_forward_time)
            self.timing_stats['loss_computation'].append(epoch_loss_time)
            self.timing_stats['backward_pass'].append(epoch_backward_time)
            self.timing_stats['optimizer_step'].append(epoch_optim_time)
            self.timing_stats['epoch_total'].append(epoch_total)

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

            # Print epoch stats with timing if profiling
            if enable_profiling and epoch < profile_epochs:
                print(f"Epoch {epoch}:: Loss: {train_loss:.6f} | "
                      f"Time: {epoch_total:.2f}s | "
                      f"Data: {epoch_data_time:.2f}s ({epoch_data_time/epoch_total*100:.1f}%) | "
                      f"Fwd: {epoch_forward_time:.2f}s ({epoch_forward_time/epoch_total*100:.1f}%) | "
                      f"Bwd: {epoch_backward_time:.2f}s ({epoch_backward_time/epoch_total*100:.1f}%) | "
                      f"Opt: {epoch_optim_time:.2f}s ({epoch_optim_time/epoch_total*100:.1f}%)")
            else:
                print(f"Epoch {epoch}:: Total Loss: {train_loss:.6f}; Value Loss: {train_vloss:.6f}; Policy Loss: {train_aloss:.6f}; LR: {current_lr:.2e}")

            history.append([epoch, train_loss, train_vloss, train_aloss])
        
        # Close profiler if enabled
        if enable_profiling and profiler_context is not None:
            profiler_context.__exit__(None, None, None)
            print(f"\nProfiler traces saved to: {os.path.join(cfg.LOGDIR, 'profiler')}")
            print("View with: tensorboard --logdir=logs/profiler")

        history = pd.DataFrame(history,columns=["Epoch","Tr_Loss","Value_Loss","Policy_Loss"])
        current_iteration = self.latest_file_number + 1
        logpath = os.path.join(cfg.LOGDIR, "{}_history.csv".format(current_iteration))
        history.to_csv(logpath, index=None)
        print(history)

        # Print timing summary
        if len(self.timing_stats['epoch_total']) > 0:
            self._print_timing_summary()

        # Store iteration number in a file for evaluation script
        iter_file = os.path.join(cfg.LOGDIR, "current_iteration.txt")
        with open(iter_file, 'w') as f:
            f.write(str(current_iteration))

    def _print_timing_summary(self):
        """Print a summary of timing statistics to identify bottlenecks."""
        import numpy as np

        print("\n" + "=" * 70)
        print("TRAINING TIMING SUMMARY")
        print("=" * 70)

        # Calculate averages (skip first epoch for warmup)
        skip = 1 if len(self.timing_stats['epoch_total']) > 1 else 0

        avg_total = np.mean(self.timing_stats['epoch_total'][skip:])
        avg_data = np.mean(self.timing_stats['data_loading'][skip:])
        avg_forward = np.mean(self.timing_stats['forward_pass'][skip:])
        avg_loss = np.mean(self.timing_stats['loss_computation'][skip:])
        avg_backward = np.mean(self.timing_stats['backward_pass'][skip:])
        avg_optim = np.mean(self.timing_stats['optimizer_step'][skip:])

        # Calculate percentages
        components = [
            ('Data Loading', avg_data),
            ('Forward Pass', avg_forward),
            ('Loss Computation', avg_loss),
            ('Backward Pass', avg_backward),
            ('Optimizer Step', avg_optim),
        ]

        # Sort by time (descending)
        components_sorted = sorted(components, key=lambda x: x[1], reverse=True)

        print(f"\nAverage epoch time: {avg_total:.3f}s")
        print(f"Number of epochs: {len(self.timing_stats['epoch_total'])}")
        print(f"Batches per epoch: {len(self.train_data) // cfg.BATCH_SIZE}")
        print(f"Batch size: {cfg.BATCH_SIZE}")
        print(f"Dataset size: {len(self.train_data)}")
        print()
        print("Time breakdown by component (sorted by time):")
        print("-" * 50)

        accounted = 0
        for name, avg_time in components_sorted:
            pct = (avg_time / avg_total) * 100 if avg_total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {name:20s}: {avg_time:7.3f}s ({pct:5.1f}%) {bar}")
            accounted += avg_time

        other_time = avg_total - accounted
        other_pct = (other_time / avg_total) * 100 if avg_total > 0 else 0
        bar = "█" * int(other_pct / 2)
        print(f"  {'Other/Overhead':20s}: {other_time:7.3f}s ({other_pct:5.1f}%) {bar}")

        print("-" * 50)
        print(f"  {'TOTAL':20s}: {avg_total:7.3f}s (100.0%)")

        # Identify bottleneck
        bottleneck_name, bottleneck_time = components_sorted[0]
        bottleneck_pct = (bottleneck_time / avg_total) * 100 if avg_total > 0 else 0

        print()
        print(f"BOTTLENECK: {bottleneck_name} ({bottleneck_pct:.1f}% of epoch time)")

        # Suggestions based on bottleneck
        print()
        print("Optimization suggestions:")
        if bottleneck_name == 'Data Loading':
            print("  - Increase num_workers in DataLoader")
            print("  - Use faster storage (SSD/NVMe)")
            print("  - Pre-load dataset into memory")
            print("  - Reduce augmentation complexity")
        elif bottleneck_name == 'Forward Pass':
            print("  - Use torch.compile() with mode='max-autotune'")
            print("  - Reduce model size/complexity")
            print("  - Increase batch size if GPU memory allows")
        elif bottleneck_name == 'Backward Pass':
            print("  - Use gradient checkpointing for memory-bound cases")
            print("  - Ensure cudnn.benchmark=True")
            print("  - Check for unnecessary gradient computation")
        elif bottleneck_name == 'Optimizer Step':
            print("  - Use fused optimizers (torch.optim.AdamW with fused=True)")
            print("  - Reduce gradient clipping overhead if not needed")

        print("=" * 70)

        # Save timing stats to CSV
        timing_df = pd.DataFrame({
            'epoch': range(len(self.timing_stats['epoch_total'])),
            'total_time': self.timing_stats['epoch_total'],
            'data_loading': self.timing_stats['data_loading'],
            'forward_pass': self.timing_stats['forward_pass'],
            'loss_computation': self.timing_stats['loss_computation'],
            'backward_pass': self.timing_stats['backward_pass'],
            'optimizer_step': self.timing_stats['optimizer_step'],
        })
        timing_path = os.path.join(cfg.LOGDIR, "timing_stats.csv")
        timing_df.to_csv(timing_path, index=False)
        print(f"\nTiming stats saved to: {timing_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Go neural network')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('--profile-epochs', type=int, default=3, help='Number of epochs to profile')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    args = parser.parse_args()

    trainer = Trainer(use_compile=not args.no_compile)
    trainer.train(enable_profiling=args.profile, profile_epochs=args.profile_epochs)