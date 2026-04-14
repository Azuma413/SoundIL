import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
from lerobot.datasets.video_utils import encode_video_frames


@dataclass
class VideoIssue:
    packet_index: int
    decoded_frames_before_error: int
    error_type: str
    message: str


@dataclass
class VideoCheckResult:
    video_path: Path
    frame_count: int
    expected_frames: int | None
    issues: list[VideoIssue]

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)


def find_dataset_root(dataset_name: str) -> Path:
    candidate = Path(dataset_name)
    if candidate.exists():
        return candidate.resolve()

    datasets_candidate = Path("datasets") / dataset_name
    if datasets_candidate.exists():
        return datasets_candidate.resolve()

    raise FileNotFoundError(f"Dataset not found: {dataset_name}")


def iter_video_files(dataset_root: Path) -> list[Path]:
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        raise FileNotFoundError(f"videos directory not found: {videos_dir}")
    return sorted(videos_dir.glob("**/*.mp4"))


def check_video(video_path: Path) -> VideoCheckResult:
    issues: list[VideoIssue] = []
    frame_count = 0

    with av.open(str(video_path), "r") as container:
        stream = container.streams.video[0]
        expected_frames = int(stream.frames) if stream.frames else None

        for packet_index, packet in enumerate(container.demux(stream)):
            try:
                frames = packet.decode()
            except Exception as exc:
                issues.append(
                    VideoIssue(
                        packet_index=packet_index,
                        decoded_frames_before_error=frame_count,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                continue

            frame_count += len(frames)

    return VideoCheckResult(
        video_path=video_path,
        frame_count=frame_count,
        expected_frames=expected_frames,
        issues=issues,
    )


def _choose_output_codec(source_stream: av.video.stream.VideoStream) -> str:
    codec_name = source_stream.codec_context.name
    if codec_name == "av1":
        return "h264"
    if codec_name in {"h264", "hevc"}:
        return codec_name
    return "h264"


def repair_video(video_path: Path, keep_backup: bool = True) -> tuple[int, int]:
    temp_fd, temp_name = tempfile.mkstemp(suffix=".mp4", dir=str(video_path.parent))
    os.close(temp_fd)
    Path(temp_name).unlink(missing_ok=True)

    replacements = 0
    written_frames = 0

    with av.open(str(video_path), "r") as container:
        input_stream = container.streams.video[0]
        fps = input_stream.average_rate
        if fps is None:
            fps = Fraction(30, 1)

        output_codec = _choose_output_codec(input_stream)

        with tempfile.TemporaryDirectory(dir=str(video_path.parent)) as frames_dir:
            frames_path = Path(frames_dir)
            last_good_image = None

            for packet in container.demux(input_stream):
                try:
                    frames = packet.decode()
                except Exception:
                    if last_good_image is None:
                        continue

                    frame_path = frames_path / f"frame-{written_frames:06d}.png"
                    last_good_image.save(frame_path)
                    replacements += 1
                    written_frames += 1
                    continue

                for frame in frames:
                    image = frame.to_image()
                    frame_path = frames_path / f"frame-{written_frames:06d}.png"
                    image.save(frame_path)
                    last_good_image = image.copy()
                    written_frames += 1

            encode_video_frames(
                frames_path,
                temp_name,
                fps=int(round(float(fps))),
                vcodec=output_codec,
                overwrite=True,
            )

    backup_path = video_path.with_suffix(video_path.suffix + ".bak")
    if keep_backup:
        shutil.move(str(video_path), str(backup_path))
    else:
        video_path.unlink()
    shutil.move(temp_name, str(video_path))

    return replacements, written_frames


def format_result(result: VideoCheckResult, dataset_root: Path) -> str:
    rel_path = result.video_path.relative_to(dataset_root)
    expected = result.expected_frames if result.expected_frames is not None else "unknown"
    if not result.has_issues:
        return f"[OK] {rel_path} frames={result.frame_count}/{expected}"

    parts = [f"[BROKEN] {rel_path} frames={result.frame_count}/{expected} issues={len(result.issues)}"]
    for issue in result.issues:
        parts.append(
            "  "
            f"packet={issue.packet_index} decoded_before_error={issue.decoded_frames_before_error} "
            f"{issue.error_type}: {issue.message}"
        )
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect LeRobot dataset videos and repair invalid frames by duplicating the previous frame."
    )
    parser.add_argument("dataset", help="Dataset path or dataset name under datasets/")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only inspect videos and report issues without modifying files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Replace broken files without keeping .bak backups.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = find_dataset_root(args.dataset)
    video_files = iter_video_files(dataset_root)

    print(f"Inspecting dataset: {dataset_root}")
    print(f"Found {len(video_files)} video files")

    broken_results: list[VideoCheckResult] = []
    for video_path in video_files:
        result = check_video(video_path)
        print(format_result(result, dataset_root))
        if result.has_issues:
            broken_results.append(result)

    if not broken_results:
        print("No invalid frames found.")
        return

    if args.check_only:
        print(f"Found {len(broken_results)} broken videos. No files were modified.")
        return

    print(f"Repairing {len(broken_results)} broken videos...")
    for result in broken_results:
        replacements, written_frames = repair_video(result.video_path, keep_backup=not args.no_backup)
        repaired = check_video(result.video_path)
        rel_path = result.video_path.relative_to(dataset_root)
        if repaired.has_issues:
            raise RuntimeError(f"Repair failed for {rel_path}: issues remain after rewrite.")
        print(
            f"[FIXED] {rel_path} replacements={replacements} "
            f"frames={written_frames}/{repaired.expected_frames if repaired.expected_frames is not None else 'unknown'}"
        )

    print("Repair completed.")


if __name__ == "__main__":
    main()

# 壊れているか確認
# uv run src/fix_data.py soundShake-m4-f10-s2-p0_0 --check-only

# 壊れているファイルを修復（--no-backupで.bakバックアップなし）
# uv run src/fix_data.py soundShake-m4-f10-s2-p0_0