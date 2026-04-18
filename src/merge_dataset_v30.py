from dataclasses import dataclass
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

@dataclass
class MergeConfig:
    name_list: list[str]
    merged_name: str
    video_files_size_in_mb: float = 0.001

def main(cfg: MergeConfig) -> None:
    dataset_root = "datasets"
    dataset_path = [Path(dataset_root) / name for name in cfg.name_list]
    repo_ids = [f"local/{name}" for name in cfg.name_list]
    datasets = []
    for repo_id, data_path in zip(repo_ids, dataset_path):
        datasets.append(LeRobotDataset(repo_id, root=data_path))
    output_dir = Path(dataset_root) / cfg.merged_name
    aggregate_datasets(
        repo_ids=[dataset.repo_id for dataset in datasets],
        aggr_repo_id=f"local/{cfg.merged_name}",
        roots=[dataset.root for dataset in datasets],
        aggr_root=output_dir,
        video_files_size_in_mb=cfg.video_files_size_in_mb,
    )

if __name__ == "__main__":
    main(MergeConfig(
        name_list=["left_a50b50", "right_a50b50"],
        merged_name="soundReal-m4-f10-s2-p0"
    ))
