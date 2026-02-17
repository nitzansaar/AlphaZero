import torch
import numpy as np
import pickle
from copy import copy
from config import Config as cfg
import random
from game import board_to_canonical_3d
from augmentation import augment_data, get_augmentations

class GoDataset:
    def __init__(self, dataset, use_augmentation=False):
        self.use_augmentation = use_augmentation and cfg.USE_AUGMENTATION
        if self.use_augmentation:
            self.augmentations = get_augmentations()

        # Precompute all canonical board representations and targets
        n = len(dataset)
        # Store raw data for augmented samples
        self.states_flat = [d[0] for d in dataset]
        self.players = [d[2] for d in dataset]

        # Precompute canonical 3D tensors for all samples
        canonical_list = []
        values = np.empty(n, dtype=np.float32)
        policies = np.empty((n, len(dataset[0][1])), dtype=np.float32) if n > 0 else np.empty((0, 0), dtype=np.float32)

        for i, datapoint in enumerate(dataset):
            state_flat = datapoint[0]
            player = datapoint[2]
            values[i] = datapoint[3]
            policies[i] = datapoint[1]
            canonical_list.append(board_to_canonical_3d(state_flat, player))

        if n > 0:
            self.canonical = torch.from_numpy(np.stack(canonical_list))
        else:
            self.canonical = torch.empty(0)
        self.values = torch.from_numpy(values)
        self.policies = torch.from_numpy(policies)

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, index):
        # Apply data augmentation with 50% probability
        if self.use_augmentation and random.random() < 0.5:
            transform_type = random.choice(self.augmentations)
            state_flat, p = augment_data(self.states_flat[index], self.policies[index].numpy(), transform_type)
            state_canonical = board_to_canonical_3d(state_flat, self.players[index])
            return (torch.from_numpy(state_canonical),
                    self.values[index],
                    torch.from_numpy(np.array(p, dtype=np.float32)))

        # Non-augmented: return precomputed tensors directly
        return (self.canonical[index],
                self.values[index],
                self.policies[index])

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
