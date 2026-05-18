from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
LEROBOT_SRC = ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

SYSTEM_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def load_info(dataset_root: Path) -> dict[str, Any]:
    with (dataset_root / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_dataset_root(dataset: str) -> Path:
    path = Path(dataset)
    if path.exists():
        return path.resolve()
    return (DATASETS_DIR / dataset).resolve()


def infer_output_name(source_name: str, shift: int) -> str:
    shift_label = f"p{shift}" if shift >= 0 else f"m{abs(shift)}"
    base = f"{source_name}-shift-action-state-{shift_label}"
    candidate = base
    idx = 1
    while (DATASETS_DIR / candidate).exists():
        candidate = f"{base}-{idx}"
        idx += 1
    return candidate


def normalize_feature_spec(info: dict[str, Any]) -> dict[str, Any]:
    spec = dict(info)
    if isinstance(spec.get("shape"), list):
        spec["shape"] = tuple(spec["shape"])
    if isinstance(spec.get("names"), list):
        spec["names"] = tuple(spec["names"])
    return spec


def tensor_to_numpy(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    return value


def normalize_value(value: Any, feature: dict[str, Any]) -> Any:
    value = tensor_to_numpy(value)
    dtype = feature.get("dtype")
    if dtype in {"image", "video"}:
        array = np.asarray(value)
        expected_shape = tuple(feature.get("shape") or ())
        if (
            len(expected_shape) == 3
            and array.ndim == 3
            and array.shape != expected_shape
            and array.shape[0] == expected_shape[-1]
            and array.shape[1:] == expected_shape[:2]
        ):
            array = np.moveaxis(array, 0, -1)
        return np.ascontiguousarray(array)
    if dtype == "string":
        return str(value)
    array = np.asarray(value, dtype=np.dtype(dtype))
    expected_shape = tuple(feature.get("shape") or ())
    if expected_shape and array.shape != expected_shape:
        array = array.reshape(expected_shape)
    return array


def get_lerobot_dataset_class() -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset


def validate_features(info: dict[str, Any]) -> None:
    features = info.get("features", {})
    missing = {"action", "observation.state"}.difference(features)
    if missing:
        raise KeyError(f"Source dataset is missing required feature(s): {sorted(missing)}")

    for feature in ("action", "observation.state"):
        dtype = features[feature].get("dtype")
        if not str(dtype).startswith("float"):
            raise ValueError(f"{feature} must be a float feature, got dtype={dtype!r}")


def user_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: normalize_feature_spec(info)
        for name, info in features.items()
        if name not in SYSTEM_FEATURES
    }


def episode_bounds(dataset: Any, episode_index: int) -> tuple[int, int]:
    episode = dataset.meta.episodes[episode_index]
    return int(episode["dataset_from_index"]), int(episode["dataset_to_index"])


def load_episode_frames(dataset: Any, episode_index: int) -> list[dict[str, Any]]:
    start, end = episode_bounds(dataset, episode_index)
    return [dataset[index] for index in range(start, end)]


def frame_task(frame: dict[str, Any]) -> str:
    task = frame.get("task")
    if isinstance(task, str) and task:
        return task
    return "task"


def shifted_episode_length(source_length: int, shift: int) -> int:
    if shift >= 0:
        return source_length + shift
    return max(source_length + shift, 0)


def source_indices(output_frame_index: int, source_length: int, shift: int) -> tuple[int, int]:
    if shift >= 0:
        numeric_index = max(output_frame_index - shift, 0)
        visual_index = min(output_frame_index, source_length - 1)
    else:
        numeric_index = output_frame_index - shift
        visual_index = output_frame_index
    return numeric_index, visual_index


def make_shifted_frame(
    frames: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    output_frame_index: int,
    shift: int,
) -> dict[str, Any]:
    numeric_index, visual_index = source_indices(output_frame_index, len(frames), shift)
    numeric_frame = frames[numeric_index]
    visual_frame = frames[visual_index]

    output_frame: dict[str, Any] = {"task": frame_task(visual_frame)}
    for name, feature in features.items():
        dtype = feature.get("dtype")
        source_frame = visual_frame if dtype in {"image", "video"} else numeric_frame
        if name in source_frame:
            output_frame[name] = normalize_value(source_frame[name], feature)

    return output_frame


def shift_dataset(
    dataset_name_or_path: str,
    shift: int,
    output_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    dataset_root = resolve_dataset_root(dataset_name_or_path)
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    source_info = load_info(dataset_root)
    validate_features(source_info)

    output_name = output_name or infer_output_name(dataset_root.name, shift)
    output_root = DATASETS_DIR / output_name
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root} (use --overwrite)")
        shutil.rmtree(output_root)

    features = user_features(source_info["features"])
    has_videos = any(feature.get("dtype") == "video" for feature in features.values())

    dataset_class = get_lerobot_dataset_class()
    source_dataset = dataset_class(f"local/{dataset_root.name}", root=dataset_root, video_backend="pyav")
    output_dataset = dataset_class.create(
        repo_id=f"local/{output_name}",
        fps=int(source_info.get("fps", source_dataset.fps)),
        root=output_root,
        robot_type=source_info.get("robot_type"),
        use_videos=has_videos,
        features=features,
        video_backend="pyav",
        batch_encoding_size=1,
    )

    total_frames = 0
    total_episodes = 0
    for episode_index in range(source_dataset.num_episodes):
        frames = load_episode_frames(source_dataset, episode_index)
        if not frames:
            print(f"Skipped episode {episode_index}: source episode is empty")
            continue
        output_length = shifted_episode_length(len(frames), shift)
        if output_length == 0:
            print(f"Skipped episode {episode_index}: shift {shift} removes all {len(frames)} frames")
            continue

        for frame_index in range(output_length):
            output_dataset.add_frame(make_shifted_frame(frames, features, frame_index, shift))
        output_dataset.save_episode()
        total_frames += output_length
        total_episodes += 1
        print(
            f"Saved episode {episode_index}: {len(frames)} -> {output_length} frames "
            f"(shift={shift:+d})"
        )

    output_dataset.finalize()
    if total_episodes == 0:
        raise ValueError("No episodes were exported.")

    print(f"Saved {total_frames} frames from {total_episodes} episodes to {output_root}")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new LeRobot dataset with action/state shifted against video frames. "
            "For shift N, action[t] and observation.state[t] are taken from old[t-N]. "
            "Positive shifts pad numeric features at the beginning and pad video features "
            "at the end; negative shifts remove the extra beginning numeric frames and "
            "ending video frames."
        )
    )
    parser.add_argument("shift", type=int, help="Frame shift. Example: 5 gives action[t] = old_action[t-5].")
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name under datasets/ or a dataset path. Defaults to the first dataset in datasets/.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output dataset name under datasets/ (default: <dataset>-shift-action-state-<shift>[-N])",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output dataset if it exists")
    return parser.parse_args()


def default_dataset_name() -> str:
    datasets = sorted(path for path in DATASETS_DIR.iterdir() if (path / "meta" / "info.json").exists())
    if not datasets:
        raise FileNotFoundError(f"No datasets found under {DATASETS_DIR}")
    return datasets[0].name


def main() -> None:
    args = parse_args()
    dataset = args.dataset or default_dataset_name()
    shift_dataset(dataset, args.shift, args.output, args.overwrite)


if __name__ == "__main__":
    main()
