from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional

import av

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class MergeConfig:
    name_list: list[str]
    merged_name: str
    video_files_size_in_mb: Optional[float] = None


CODEC_FAMILIES = {
    "av1": "av1",
    "libaom-av1": "av1",
    "libdav1d": "av1",
    "libsvtav1": "av1",
    "h264": "h264",
    "libx264": "h264",
    "hevc": "hevc",
    "h265": "hevc",
    "libx265": "hevc",
}

ENCODER_BY_FAMILY = {
    "av1": "libsvtav1",
    "h264": "h264",
    "hevc": "hevc",
}


def codec_family(codec: str) -> str:
    normalized = codec.lower()
    return CODEC_FAMILIES.get(normalized, normalized)


def expected_video_codec(dataset: LeRobotDataset, video_key: str) -> str:
    feature = dataset.features[video_key]
    codec = feature.get("info", {}).get("video.codec")
    if not codec:
        raise ValueError(f"{dataset.root}: missing video.codec metadata for {video_key}")
    return codec


def actual_video_codec(video_path: Path) -> str:
    with av.open(str(video_path), "r") as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found: {video_path}")
        return container.streams.video[0].codec_context.name


def referenced_video_paths(dataset: LeRobotDataset, video_key: str) -> list[Path]:
    episodes = dataset.meta.episodes
    if episodes is None:
        dataset.meta.load_metadata()
        episodes = dataset.meta.episodes
    if episodes is None:
        raise ValueError(f"{dataset.root}: failed to load episode metadata")

    chunk_column = f"videos/{video_key}/chunk_index"
    file_column = f"videos/{video_key}/file_index"
    chunk_file_pairs = sorted(
        {
            (int(chunk), int(file))
            for chunk, file in zip(episodes[chunk_column], episodes[file_column], strict=False)
        }
    )

    return [
        dataset.root
        / dataset.meta.video_path.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        for chunk_index, file_index in chunk_file_pairs
    ]


def backup_video_path(dataset_root: Path, video_path: Path) -> Path:
    relative_path = video_path.relative_to(dataset_root)
    backup_root = dataset_root / "backups" / "merge_dataset_v30"
    candidate = backup_root / relative_path.with_suffix(relative_path.suffix + ".bak")
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        numbered = candidate.with_name(f"{candidate.name}.{index}")
        if not numbered.exists():
            return numbered
        index += 1


def transcode_video(video_path: Path, expected_codec: str, fps: int, pix_fmt: str) -> None:
    expected_family = codec_family(expected_codec)
    encoder = ENCODER_BY_FAMILY.get(expected_family)
    if encoder is None:
        raise ValueError(f"Unsupported target codec '{expected_codec}' for {video_path}")

    temp_fd, temp_name = tempfile.mkstemp(suffix=".mp4", dir=str(video_path.parent))
    os.close(temp_fd)
    Path(temp_name).unlink(missing_ok=True)

    try:
        with av.open(str(video_path), "r") as input_container:
            input_stream = input_container.streams.video[0]
            width = input_stream.codec_context.width
            height = input_stream.codec_context.height

            with av.open(temp_name, "w", options={"movflags": "faststart"}) as output_container:
                output_stream = output_container.add_stream(
                    encoder,
                    rate=fps,
                    options={"g": "2", "crf": "30"},
                )
                output_stream.width = width
                output_stream.height = height
                output_stream.pix_fmt = pix_fmt

                for frame in input_container.decode(input_stream):
                    converted_frame = frame.reformat(width=width, height=height, format="rgb24")
                    for packet in output_stream.encode(converted_frame):
                        output_container.mux(packet)

                for packet in output_stream.encode():
                    output_container.mux(packet)

        converted_codec = actual_video_codec(Path(temp_name))
        if codec_family(converted_codec) != expected_family:
            raise RuntimeError(
                f"Transcoded {video_path} to {converted_codec}, expected {expected_codec}"
            )

        backup_path = backup_video_path(video_path.parents[3], video_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(video_path), str(backup_path))
        shutil.move(temp_name, str(video_path))
        print(f"[TRANSCODED] {video_path} codec={converted_codec} backup={backup_path}")
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def ensure_video_codecs_match(datasets: list[LeRobotDataset]) -> None:
    for dataset in datasets:
        for video_key in dataset.meta.video_keys:
            expected_codec = expected_video_codec(dataset, video_key)
            expected_family = codec_family(expected_codec)
            feature_info = dataset.features[video_key].get("info", {})
            pix_fmt = feature_info.get("video.pix_fmt", "yuv420p")

            for video_path in referenced_video_paths(dataset, video_key):
                if not video_path.exists():
                    raise FileNotFoundError(f"Referenced video file not found: {video_path}")

                current_codec = actual_video_codec(video_path)
                if codec_family(current_codec) == expected_family:
                    print(f"[OK] {video_path} codec={current_codec}")
                    continue

                print(
                    f"[MISMATCH] {video_path} codec={current_codec}, "
                    f"expected={expected_codec}; transcoding before merge"
                )
                transcode_video(
                    video_path=video_path,
                    expected_codec=expected_codec,
                    fps=dataset.fps,
                    pix_fmt=pix_fmt,
                )


def main(cfg: MergeConfig) -> None:
    dataset_root = "datasets"
    dataset_path = [Path(dataset_root) / name for name in cfg.name_list]
    repo_ids = [f"local/{name}" for name in cfg.name_list]
    datasets = []
    for repo_id, data_path in zip(repo_ids, dataset_path):
        datasets.append(LeRobotDataset(repo_id, root=data_path))

    ensure_video_codecs_match(datasets)

    output_dir = Path(dataset_root) / cfg.merged_name
    aggregate_kwargs = {}
    if cfg.video_files_size_in_mb is not None:
        aggregate_kwargs["video_files_size_in_mb"] = cfg.video_files_size_in_mb

    aggregate_datasets(
        repo_ids=[dataset.repo_id for dataset in datasets],
        aggr_repo_id=f"local/{cfg.merged_name}",
        roots=[dataset.root for dataset in datasets],
        aggr_root=output_dir,
        **aggregate_kwargs,
    )

if __name__ == "__main__":
    main(MergeConfig(
        name_list=["soundReal-m4-f10-s2-p0_0", "soundReal-m4-f10-s2-p0_1"],
        merged_name="RealShake-m4-f10-s2-p0"
    ))
