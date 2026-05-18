from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
STATS_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


@dataclass
class FeatureStats:
    values: list[np.ndarray]

    def add(self, array: np.ndarray) -> None:
        if array.size:
            self.values.append(np.asarray(array, dtype=np.float32))

    def compute(self) -> dict[str, np.ndarray]:
        if not self.values:
            raise ValueError("Cannot compute stats from empty values.")
        values = np.concatenate(self.values, axis=0)
        return compute_stats(values)


def load_info(dataset_root: Path) -> dict[str, Any]:
    with (dataset_root / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def infer_output_name(source_name: str) -> str:
    base = f"{source_name}-state-action-shifted"
    candidate = base
    idx = 1
    while (DATASETS_DIR / candidate).exists():
        candidate = f"{base}-{idx}"
        idx += 1
    return candidate


def resolve_dataset_root(dataset: str) -> Path:
    path = Path(dataset)
    if path.exists():
        return path.resolve()
    return (DATASETS_DIR / dataset).resolve()


def stack_series(series: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in series.to_list()])


def arrays_to_series(values: np.ndarray, index: pd.Index) -> pd.Series:
    return pd.Series([np.asarray(row, dtype=np.float32) for row in values], index=index, dtype=object)


def compute_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    return {
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "count": np.array([values.shape[0]], dtype=np.int64),
        "q01": np.quantile(values, 0.01, axis=0),
        "q10": np.quantile(values, 0.10, axis=0),
        "q50": np.quantile(values, 0.50, axis=0),
        "q90": np.quantile(values, 0.90, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }


def stats_for_json(stats: dict[str, np.ndarray]) -> dict[str, Any]:
    return {key: np.asarray(value).tolist() for key, value in stats.items()}


def update_episode_stats(
    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]],
    output_root: Path,
) -> None:
    episodes_root = output_root / "meta" / "episodes"
    if not episodes_root.exists():
        return

    for parquet_path in sorted(episodes_root.glob("**/*.parquet")):
        df = pd.read_parquet(parquet_path)
        if "episode_index" not in df.columns:
            continue

        for row_index, episode_index in df["episode_index"].items():
            stats = episode_stats.get(int(episode_index))
            if stats is None:
                continue
            for feature, feature_stats in stats.items():
                for stat_key, value in feature_stats.items():
                    column = f"stats/{feature}/{stat_key}"
                    if column in df.columns:
                        df.at[row_index, column] = np.asarray(value).tolist()
        df.to_parquet(parquet_path, index=False)


def update_global_stats(
    global_stats: dict[str, FeatureStats],
    output_root: Path,
) -> None:
    stats_path = output_root / "meta" / "stats.json"
    if not stats_path.exists():
        return

    with stats_path.open("r", encoding="utf-8") as f:
        stats_json = json.load(f)

    for feature, accumulator in global_stats.items():
        stats_json[feature] = stats_for_json(accumulator.compute())

    save_json(stats_path, stats_json)


def update_data_files(output_root: Path) -> tuple[int, int]:
    data_root = output_root / "data"
    parquet_paths = sorted(data_root.glob("**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {data_root}")

    global_stats = {
        "action": FeatureStats([]),
        "observation.state": FeatureStats([]),
    }
    episode_stats: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    total_frames = 0
    total_episodes = set()

    for parquet_path in parquet_paths:
        df = pd.read_parquet(parquet_path)
        required_columns = {"episode_index", "observation.state", "action"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise KeyError(f"{parquet_path} is missing columns: {sorted(missing_columns)}")

        for episode_index, group in df.groupby("episode_index", sort=False):
            indices = group.index
            states = stack_series(group["observation.state"])
            shifted_states = states.copy()
            if len(states) > 1:
                shifted_states[:-1] = states[1:]
            shifted_states[-1] = states[-1]
            actions = states.copy()

            df.loc[indices, "action"] = arrays_to_series(actions, indices)
            df.loc[indices, "observation.state"] = arrays_to_series(shifted_states, indices)

            action_stats = compute_stats(actions)
            state_stats = compute_stats(shifted_states)
            episode_stats[int(episode_index)] = {
                "action": action_stats,
                "observation.state": state_stats,
            }
            global_stats["action"].add(actions)
            global_stats["observation.state"].add(shifted_states)
            total_frames += len(indices)
            total_episodes.add(int(episode_index))

        df.to_parquet(parquet_path, index=False)
        print(f"Updated {parquet_path.relative_to(output_root)}")

    update_episode_stats(episode_stats, output_root)
    update_global_stats(global_stats, output_root)
    return len(total_episodes), total_frames


def validate_features(info: dict[str, Any]) -> None:
    features = info.get("features", {})
    if "action" not in features:
        raise KeyError("Source dataset does not have an 'action' feature.")
    if "observation.state" not in features:
        raise KeyError("Source dataset does not have an 'observation.state' feature.")

    action_shape = tuple(features["action"].get("shape", ()))
    state_shape = tuple(features["observation.state"].get("shape", ()))
    if action_shape != state_shape:
        raise ValueError(f"action shape {action_shape} differs from observation.state shape {state_shape}.")


def update_dataset(
    dataset_name_or_path: str,
    output_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    dataset_root = resolve_dataset_root(dataset_name_or_path)
    if not (dataset_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    source_name = dataset_root.name
    output_name = output_name or infer_output_name(source_name)
    output_root = DATASETS_DIR / output_name
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root} (use --overwrite)")
        shutil.rmtree(output_root)

    info = load_info(dataset_root)
    validate_features(info)

    shutil.copytree(dataset_root, output_root)
    total_episodes, total_frames = update_data_files(output_root)
    print(f"Saved {total_frames} frames from {total_episodes} episodes to {output_root}")
    print("Videos were copied as-is; no video re-encoding was performed.")
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a dataset, then update only parquet data so action[t] = observation.state[t] "
            "and observation.state[t] = observation.state[t+1]. The final state in each episode is kept as-is."
        )
    )
    parser.add_argument("dataset", help="Dataset name under datasets/ or a dataset path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output dataset name under datasets/ (default: <dataset>-state-action-shifted[-N])",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output dataset if it exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_dataset(args.dataset, args.output, args.overwrite)


if __name__ == "__main__":
    main()
