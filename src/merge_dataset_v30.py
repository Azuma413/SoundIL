from dataclasses import dataclass
from pathlib import Path
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

@dataclass
class MergeConfig:
    name_list: list[str]
    merged_name: str

def main(cfg: MergeConfig) -> None:
    dataset_root = "datasets"
    dataset_path = [Path(dataset_root) / name for name in cfg.name_list]
    repo_ids = [f"local/{name}" for name in cfg.name_list]
    datasets = []
    for repo_id, data_path in zip(repo_ids, dataset_path):
        datasets.append(LeRobotDataset(repo_id, root=data_path))
    output_dir = Path(dataset_root) / cfg.merged_name
    merged_dataset = merge_datasets(
        datasets,
        output_repo_id=f"local/{cfg.merged_name}",
        output_dir=output_dir,
    )

if __name__ == "__main__":
    main(MergeConfig(
        name_list=["kitcut_1ep","kitcut_2ep","kitcut_8ep","kitcut_11ep","kitcut_32ep","kitcut_46ep"],
        merged_name="kitcut-dataset"
    ))