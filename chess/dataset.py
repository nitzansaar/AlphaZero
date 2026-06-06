"""
Training data storage for chess self-play.

Each training sample is stored compactly as:
    [fen, (policy_indices, policy_probs), player, value]

  - fen:            board position (recompute planes lazily; far smaller than
                    storing a 19x8x8 float tensor per sample)
  - policy target:  sparse representation of the 4672-wide MCTS visit
                    distribution (only ~35 legal entries are nonzero)
  - player:         +1 if White to move at this position, -1 if Black
  - value:          game outcome from this position's side-to-move perspective

No data augmentation: chess has no rotational symmetry (only an L-R mirror is
valid and it entangles castling encoding), so it is intentionally omitted.
"""
import pickle

import chess
import numpy as np
import torch

from config import Config as cfg
import encoding


class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fen, (policy_indices, policy_probs), _player, value = self.data[index]

        board = chess.Board(fen)
        planes = encoding.board_to_planes(board)

        policy = np.zeros(cfg.ACTION_SIZE, dtype=np.float32)
        policy[np.asarray(policy_indices, dtype=np.int64)] = np.asarray(
            policy_probs, dtype=np.float32
        )

        return (
            torch.tensor(planes, dtype=torch.float),
            torch.tensor([value], dtype=torch.float),
            torch.tensor(policy, dtype=torch.float),
        )


class TrainingDataset:
    def __init__(self):
        self.training_dataset = []

    def calculate_values(self, dataset, winner):
        """Assign value to each position from its side-to-move perspective.

        winner is from White's perspective (+1 White, -1 Black, 0 draw); a
        position's value is winner * player.
        """
        for ind, step in enumerate(dataset):
            step_ = list(step)
            step_player = step_[2]
            value = winner * step_player
            step_.append(value)
            dataset[ind] = step_
        return dataset

    def add_game_to_training_dataset(self, dataset, winner):
        data = self.calculate_values(dataset, winner)
        self.training_dataset.extend(data)
        self.training_dataset = self.training_dataset[-cfg.DATASET_QUEUE_SIZE:]

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self.training_dataset, handle)

    def load(self, path):
        with open(path, "rb") as handle:
            self.training_dataset = pickle.load(handle)
