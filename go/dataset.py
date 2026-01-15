import torch
import numpy as np
import pickle
from copy import copy
from config import Config as cfg
import random

class GoDataset:
    def __init__(self, dataset, use_augmentation=False):
        self.data = dataset
        self.use_augmentation = use_augmentation and cfg.USE_AUGMENTATION
        if self.use_augmentation:
            from augmentation import get_augmentations
            self.augmentations = get_augmentations()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datapoint = self.data[index]
        state_flat = datapoint[0]  # Game state (27 values: 25 board + ko + passes)
        player = datapoint[2]      # Player (1 or -1)
        v = datapoint[3]           # Value target
        p = datapoint[1]           # Policy target (26 values: 25 positions + pass)

        # Apply data augmentation with 50% probability
        if self.use_augmentation and random.random() < 0.5:
            from augmentation import augment_data
            transform_type = random.choice(self.augmentations)
            state_flat, p = augment_data(state_flat, p, transform_type)

        # Convert flat state to canonical 3-plane representation
        from game import board_to_canonical_3d
        state_canonical = board_to_canonical_3d(state_flat, player)

        return (torch.tensor(state_canonical, dtype=torch.float),
                torch.tensor(v, dtype=torch.float),
                torch.tensor(p, dtype=torch.float))


# Alias for backward compatibility
TicTacToeDataset = GoDataset


class TrainingDataset:
    def __init__(self):
        self.training_dataset = []

    def calculate_values(self, dataset, winner):
        """Assign value to each position in the dataset based on the winner."""
        for ind, step in enumerate(dataset):
            step_ = copy(step)
            step_player = step_[2]
            if winner == 0:  # draw
                value = 0
            else:
                if winner == step_player:
                    value = 1
                else:
                    value = -1
            step_.append(value)
            dataset[ind] = step_
        return dataset

    def add_game_to_training_dataset(self, dataset, winner):
        """Add the completed game data to the training dataset."""
        data = self.calculate_values(dataset, winner)
        self.training_dataset.extend(data)
        self.training_dataset = self.training_dataset[-1 * cfg.DATASET_QUEUE_SIZE:]

    def save(self, path):
        """Save the training dataset to a pickle file."""
        with open(path, 'wb') as handle:
            pickle.dump(self.training_dataset, handle)

    def load(self, path):
        """Load the training dataset from a pickle file."""
        with open(path, 'rb') as handle:
            self.training_dataset = pickle.load(handle)

    def retreive_test_train_data(self):
        data = self.training_dataset
        num_samples = len(data)
        train_idx = np.random.choice(np.arange(num_samples), int(num_samples), replace=False)
        train_idx_set = set(train_idx)
        val_idx = [t for t in range(num_samples) if t not in train_idx_set]
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        return GoDataset(train_data), GoDataset(val_data)
