import json
import os
from glob import glob


MODEL_FILENAME = "model.pt"
METADATA_FILENAME = "training_info.json"
TRAINING_STATE_FILENAME = "training_state.pt"


def iteration_dir(save_model_path, iteration):
    return os.path.join(save_model_path, str(int(iteration)))


def iteration_model_path(save_model_path, iteration):
    return os.path.join(iteration_dir(save_model_path, iteration), MODEL_FILENAME)


def iteration_metadata_path(save_model_path, iteration):
    return os.path.join(iteration_dir(save_model_path, iteration), METADATA_FILENAME)


def iteration_training_state_path(save_model_path, iteration):
    return os.path.join(iteration_dir(save_model_path, iteration), TRAINING_STATE_FILENAME)


def model_metadata_path(model_path):
    if os.path.basename(model_path) == MODEL_FILENAME:
        return os.path.join(os.path.dirname(model_path), METADATA_FILENAME)
    root, _ = os.path.splitext(model_path)
    return f"{root}.json"


def _legacy_checkpoint_iter(path):
    name = os.path.basename(path)
    if name.endswith("_ts.pt") or name == "model_ts.pt":
        return None
    stem = name.split("_")[0]
    try:
        return int(stem)
    except ValueError:
        return None


def checkpoint_iter(path):
    """Return checkpoint iteration number for folder or legacy flat layouts."""
    if os.path.basename(path) == MODEL_FILENAME:
        try:
            return int(os.path.basename(os.path.dirname(path)))
        except ValueError:
            return None
    return _legacy_checkpoint_iter(path)


def discover_checkpoints(save_model_path):
    """Return [(iteration, model_path), ...], preferring folder checkpoints."""
    checkpoints = {}

    for child in glob(os.path.join(save_model_path, "*")):
        if not os.path.isdir(child):
            continue
        try:
            iteration = int(os.path.basename(child))
        except ValueError:
            continue
        model_path = os.path.join(child, MODEL_FILENAME)
        if os.path.isfile(model_path):
            checkpoints[iteration] = model_path

    for model_path in glob(os.path.join(save_model_path, "*.pt")):
        iteration = _legacy_checkpoint_iter(model_path)
        if iteration is not None and iteration not in checkpoints:
            checkpoints[iteration] = model_path

    return sorted(checkpoints.items())


def latest_checkpoint_path(save_model_path):
    checkpoints = discover_checkpoints(save_model_path)
    if not checkpoints:
        return None, None
    return checkpoints[-1]


def build_model_metadata(
    *,
    iteration,
    global_step_samples,
    total_num_data_rows,
    global_minibatch_step=None,
    train_steps_per_iteration=None,
    batch_size=None,
    avg_selfplay_game_moves=None,
):
    metadata = {
        "iteration": int(iteration),
        "global_step_samples": int(global_step_samples),
        "total_num_data_rows": int(total_num_data_rows),
    }
    if global_minibatch_step is not None:
        metadata["global_minibatch_step"] = int(global_minibatch_step)
    if train_steps_per_iteration is not None:
        metadata["train_steps_per_iteration"] = int(train_steps_per_iteration)
    if batch_size is not None:
        metadata["batch_size"] = int(batch_size)
    if avg_selfplay_game_moves is not None:
        metadata["avg_selfplay_game_moves"] = float(avg_selfplay_game_moves)
    return metadata


def save_model_metadata(model_path, **kwargs):
    metadata = build_model_metadata(**kwargs)
    metadata_path = model_metadata_path(model_path)
    metadata_dir = os.path.dirname(metadata_path)
    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)
    tmp_path = f"{metadata_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, metadata_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return metadata_path, metadata


def load_model_metadata(model_path):
    metadata_path = model_metadata_path(model_path)
    try:
        with open(metadata_path) as f:
            return json.load(f)
    except (FileNotFoundError, OSError, ValueError):
        return {}
