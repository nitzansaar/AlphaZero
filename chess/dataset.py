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
        """Append sparse positions written by chess_selfplay.

        The C++ binary emits a compact representation (a dense (N,4672) policy
        is ~159 MB per 8.5k positions; only ~35 entries per row are nonzero):
            fens.txt            N lines, one FEN per position
            policy_counts.npy   (N,)  nonzero policy entries per position
            policy_indices.npy  (S,)  concatenated action indices
            policy_probs.npy    (S,)  concatenated probabilities
            values.npy          (N,)  outcome from the position's STM view

        Stored as [fen, (indices, probs), player, value] — the legacy sparse
        format, so ChessDataset.__getitem__ recomputes the planes from the FEN.
        The value is already from the position's side-to-move perspective.
        """
        with open(os.path.join(npy_dir, "fens.txt")) as f:
            fens = f.read().splitlines()
        counts = np.load(os.path.join(npy_dir, "policy_counts.npy")).astype(np.int64)
        indices = np.load(os.path.join(npy_dir, "policy_indices.npy")).astype(np.int64)
        probs = np.load(os.path.join(npy_dir, "policy_probs.npy")).astype(np.float32)
        values = np.load(os.path.join(npy_dir, "values.npy"))

        assert len(fens) == len(counts) == len(values), "ragged npy lengths"
        assert int(counts.sum()) == len(indices) == len(probs), \
            "policy_counts does not match indices/probs length"

        off = 0
        for i in range(len(values)):
            c = int(counts[i])
            idx = indices[off:off + c]
            pr = probs[off:off + c]
            off += c
            player = 1 if chess.Board(fens[i]).turn == chess.WHITE else -1
            self.training_dataset.append(
                [fens[i], (idx, pr), player, float(values[i])]
            )
        self.training_dataset = self.training_dataset[-cfg.DATASET_QUEUE_SIZE:]

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self.training_dataset, handle)

    def load(self, path):
        with open(path, "rb") as handle:
            self.training_dataset = pickle.load(handle)
