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
import os
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
        item = self.data[index]

        # Two storage formats coexist:
        #   - C++ self-play (dense): [planes(19,8,8), policy(4672), 0, value]
        #     where planes/policy are np.ndarrays already in network form.
        #   - Legacy Python self-play (sparse): [fen, (idx, probs), player, value]
        if isinstance(item[0], np.ndarray):
            planes = item[0].astype(np.float32)
            policy = np.asarray(item[1], dtype=np.float32)
            value = float(item[3])
        else:
            fen, (policy_indices, policy_probs), _player, value = item
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

    def load_from_npy(self, npy_dir):
        """Append dense positions written by chess_selfplay (states/policies/values).

        Stored as [planes(19,8,8), policy(4672), 0, value]; the value is already
        from the position's side-to-move perspective, so the player slot is an
        unused 0 sentinel.
        """
        states = np.load(os.path.join(npy_dir, "states.npy"))
        policies = np.load(os.path.join(npy_dir, "policies.npy"))
        values = np.load(os.path.join(npy_dir, "values.npy"))

        assert states.shape[1:] == (cfg.NUM_INPUT_PLANES, 8, 8), \
            f"unexpected states shape {states.shape}"
        assert policies.shape[1] == cfg.ACTION_SIZE, \
            f"unexpected policies shape {policies.shape}"
        assert len(states) == len(policies) == len(values), "ragged npy lengths"

        for i in range(len(values)):
            self.training_dataset.append(
                [states[i], policies[i], 0, float(values[i])]
            )
        self.training_dataset = self.training_dataset[-cfg.DATASET_QUEUE_SIZE:]

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self.training_dataset, handle)

    def load(self, path):
        with open(path, "rb") as handle:
            self.training_dataset = pickle.load(handle)
