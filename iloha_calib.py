#!/usr/bin/env python3
"""Camera-based joint-offset calibration for the right Iloha arm.

The dataset image at a frame is paired with ``observation.state`` from that
same frame.  For each selected reference pose this script moves the robot to
that state plus a trial offset, captures the real cameras, and minimizes a
photometric/edge difference by coordinate descent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from iloha_calibration import (
    DEFAULT_CALIBRATION_PATH,
    dataset_to_hardware_action,
    load_joint_offsets,
    make_calibration_document,
    offsets_summary,
    offsets_to_mapping,
    save_calibration_file,
)
from iloha_eval import (
    CAMERA_MAX_FRAME_AGE_MS,
    initialize_cameras,
    load_local_dataset,
    reset_robot_to_home,
    resolve_auto_robstride_ports,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.iloha import Iloha, IlohaConfig
from soundreal_utils import (
    CAMERA_FPS,
    RIGHT_ARM_DIM,
    RIGHT_ARM_FEATURE_NAMES,
    make_full_action_from_right,
    preprocess_soundreal_camera_frame,
)


DEFAULT_DATASET_PATHS = (
    Path("datasets/soundRealAll-m4-f10-s2-p0"),
    Path("datasets/soundRealShake-m4-f10-s2-p0"),
)
DEFAULT_CAMERA_KEYS = ("side",)
POSE_FEATURE = "observation.state"


@dataclass(frozen=True)
class ReferencePose:
    dataset_path: Path
    dataset_index: int
    episode_index: int
    frame_index: int
    target: np.ndarray
    images: dict[str, np.ndarray]

    def description(self) -> str:
        return (
            f"{self.dataset_path.name}: episode={self.episode_index}, "
            f"frame={self.frame_index}, index={self.dataset_index}"
        )


@dataclass(frozen=True)
class ReferenceTrajectory:
    dataset_path: Path
    episode_index: int
    start_frame_index: int
    actions: np.ndarray
    images: dict[str, np.ndarray]
    edge_maps: dict[str, np.ndarray]
    edge_distance_maps: dict[str, np.ndarray]
    fps: float
    preroll_frames: int

    def description(self) -> str:
        end_frame = self.start_frame_index + len(self.actions) - 1
        return (
            f"{self.dataset_path.name}: episode={self.episode_index}, "
            f"frames={self.start_frame_index}..{end_frame}"
        )


def parse_csv_names(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required")
    return values


def parse_joint_indices(raw: str) -> list[int]:
    try:
        joints = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--joints must be comma-separated integers") from exc
    if not joints:
        raise argparse.ArgumentTypeError("At least one joint is required")
    if len(set(joints)) != len(joints):
        raise argparse.ArgumentTypeError("--joints contains a duplicate joint")
    if any(joint < 1 or joint > RIGHT_ARM_DIM for joint in joints):
        raise argparse.ArgumentTypeError(f"Joint numbers must be in 1..{RIGHT_ARM_DIM}")
    return [joint - 1 for joint in joints]


def tensor_image_to_uint8(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D image, got shape {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] > 3:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.5:
            array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _prepare_gray(image: np.ndarray) -> np.ndarray:
    uint8_image = tensor_image_to_uint8(image)
    gray = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32) / 255.0


def structural_edge_map(image: np.ndarray) -> np.ndarray:
    """Extract long edges from stationary workspace structures.

    Short edges from cans, boxes and texture are intentionally discarded.  The
    bottom strip is also ignored because the end-effector camera can see parts
    of its own mount there; those edges do not describe camera pose in the
    workspace.
    """

    gray = (_prepare_gray(image) * 255).astype(np.uint8)
    edges = cv2.Canny(gray, 45, 120)
    height, width = edges.shape
    edges[int(height * 0.90) :] = 0
    min_line_length = max(40, int(min(height, width) * 0.28))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=24,
        minLineLength=min_line_length,
        maxLineGap=14,
    )
    line_map = np.zeros_like(edges)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_map, (x1, y1), (x2, y2), 255, 2, cv2.LINE_AA)
    if not line_map.any():
        line_map = edges
    return line_map > 0


def _symmetric_chamfer_distance(
    reference_edges: np.ndarray,
    current_edges: np.ndarray,
    *,
    max_distance: float,
) -> float:
    if not reference_edges.any() and not current_edges.any():
        return 0.0
    if not reference_edges.any() or not current_edges.any():
        return 1.0
    distance_to_current = cv2.distanceTransform(
        (~current_edges).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_reference = cv2.distanceTransform(
        (~reference_edges).astype(np.uint8), cv2.DIST_L2, 3
    )
    return 0.5 * (
        float(np.mean(np.minimum(distance_to_current[reference_edges], max_distance)))
        + float(np.mean(np.minimum(distance_to_reference[current_edges], max_distance)))
    ) / max_distance


def structural_image_difference(reference: np.ndarray, current: np.ndarray) -> float:
    reference_edges = structural_edge_map(reference)
    current_edges = structural_edge_map(current)
    return _symmetric_chamfer_distance(
        reference_edges,
        current_edges,
        max_distance=20.0,
    )


def background_edge_map(image: np.ndarray) -> np.ndarray:
    """Dense workspace edges, excluding the camera mount at the image bottom."""

    gray = (_prepare_gray(image) * 255).astype(np.uint8)
    edges = cv2.Canny(gray, 45, 120) > 0
    edges[int(edges.shape[0] * 0.78) :] = False
    return edges


def _trimmed_symmetric_chamfer_distance(
    reference_edges: np.ndarray,
    current_edges: np.ndarray,
    *,
    max_distance: float,
    keep_fraction: float,
) -> float:
    if not reference_edges.any() and not current_edges.any():
        return 0.0
    if not reference_edges.any() or not current_edges.any():
        return 1.0

    distance_to_current = cv2.distanceTransform(
        (~current_edges).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_reference = cv2.distanceTransform(
        (~reference_edges).astype(np.uint8), cv2.DIST_L2, 3
    )

    def trimmed_mean(values: np.ndarray) -> float:
        values = np.minimum(values, max_distance)
        keep_count = max(1, int(math.ceil(values.size * keep_fraction)))
        return float(np.mean(np.partition(values, keep_count - 1)[:keep_count]))

    return 0.5 * (
        trimmed_mean(distance_to_current[reference_edges])
        + trimmed_mean(distance_to_reference[current_edges])
    ) / max_distance


def _edge_distance_map(edges: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)


def _trimmed_chamfer_from_distance_maps(
    reference_edges: np.ndarray,
    current_edges: np.ndarray,
    distance_to_reference: np.ndarray,
    distance_to_current: np.ndarray,
    *,
    max_distance: float,
    keep_fraction: float,
) -> float:
    if not reference_edges.any() and not current_edges.any():
        return 0.0
    if not reference_edges.any() or not current_edges.any():
        return 1.0

    def trimmed_mean(values: np.ndarray) -> float:
        values = np.minimum(values, max_distance)
        keep_count = max(1, int(math.ceil(values.size * keep_fraction)))
        return float(np.mean(np.partition(values, keep_count - 1)[:keep_count]))

    return 0.5 * (
        trimmed_mean(distance_to_current[reference_edges])
        + trimmed_mean(distance_to_reference[current_edges])
    ) / max_distance


def background_image_difference(reference: np.ndarray, current: np.ndarray) -> float:
    """Compare stable workspace edges while trimming unmatched movable objects."""

    return _trimmed_symmetric_chamfer_distance(
        background_edge_map(reference),
        background_edge_map(current),
        max_distance=20.0,
        keep_fraction=0.60,
    )


def image_difference(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    metric: str = "full",
) -> float:
    """Return a lower-is-better image distance robust to modest lighting changes."""

    if metric == "structural":
        return structural_image_difference(reference, current)
    if metric == "background":
        return background_image_difference(reference, current)
    if metric != "full":
        raise ValueError(f"Unknown image metric: {metric}")

    reference_gray = _prepare_gray(reference)
    current_gray = _prepare_gray(current)
    if reference_gray.shape != current_gray.shape:
        current_gray = cv2.resize(
            current_gray,
            (reference_gray.shape[1], reference_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    def standardized(image: np.ndarray) -> np.ndarray:
        std = max(float(image.std()), 1.0 / 255.0)
        return np.clip((image - float(image.mean())) / std, -3.0, 3.0)

    pixel_score = float(
        np.mean(np.abs(standardized(reference_gray) - standardized(current_gray))) / 6.0
    )

    def gradient_magnitude(image: np.ndarray) -> np.ndarray:
        dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(dx, dy)
        scale = max(float(np.percentile(magnitude, 95)), 1e-6)
        return np.clip(magnitude / scale, 0.0, 1.0)

    reference_gradient = gradient_magnitude(reference_gray)
    current_gradient = gradient_magnitude(current_gray)
    gradient_score = float(np.mean(np.abs(reference_gradient - current_gradient)))

    reference_edges = cv2.Canny((reference_gray * 255).astype(np.uint8), 60, 140) > 0
    current_edges = cv2.Canny((current_gray * 255).astype(np.uint8), 60, 140) > 0
    chamfer_score = _symmetric_chamfer_distance(
        reference_edges,
        current_edges,
        max_distance=12.0,
    )

    return 0.30 * pixel_score + 0.35 * gradient_score + 0.35 * chamfer_score


def aggregate_image_difference(
    references: Sequence[ReferencePose],
    captured_images: Sequence[dict[str, np.ndarray]],
    camera_keys: Sequence[str],
    *,
    metric: str = "full",
) -> tuple[float, list[dict[str, float]]]:
    if len(references) != len(captured_images):
        raise ValueError("Reference/capture count mismatch")
    per_pose = []
    pose_scores = []
    for reference, capture in zip(references, captured_images, strict=True):
        camera_scores = {
            camera_key: image_difference(
                reference.images[camera_key],
                capture[camera_key],
                metric=metric,
            )
            for camera_key in camera_keys
        }
        score = float(np.mean(list(camera_scores.values())))
        per_pose.append({"score": score, **camera_scores})
        pose_scores.append(score)
    return float(np.median(pose_scores)), per_pose


def select_diverse_pose_indices(
    dataset: LeRobotDataset,
    count: int,
    episode_margin_frames: int,
) -> list[int]:
    candidates: list[tuple[int, np.ndarray]] = []
    for episode in dataset.meta.episodes:
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        low = start + episode_margin_frames
        high = end - episode_margin_frames
        if high <= low:
            low, high = start, end
        index = low + max(0, high - low - 1) // 2
        item = dataset.hf_dataset[index]
        target = np.asarray(item[POSE_FEATURE], dtype=np.float32)
        if target.shape == (RIGHT_ARM_DIM,) and np.all(np.isfinite(target)):
            candidates.append((index, target))

    if len(candidates) < count:
        raise ValueError(
            f"Dataset {dataset.root} has only {len(candidates)} usable episodes, "
            f"but {count} reference poses were requested"
        )

    states = np.stack([target for _, target in candidates])
    scale = np.maximum(np.ptp(states, axis=0), 1e-3)
    normalized = states / scale
    median = np.median(normalized, axis=0)
    selected = [int(np.argmin(np.linalg.norm(normalized - median, axis=1)))]
    while len(selected) < count:
        distances = np.min(
            np.stack(
                [np.linalg.norm(normalized - normalized[index], axis=1) for index in selected]
            ),
            axis=0,
        )
        distances[selected] = -np.inf
        selected.append(int(np.argmax(distances)))
    return [candidates[index][0] for index in selected]


def load_reference_pose(
    dataset: LeRobotDataset,
    dataset_path: Path,
    index: int,
    camera_keys: Sequence[str],
) -> ReferencePose:
    item = dataset.hf_dataset[index]
    episode_index = int(item["episode_index"].item())
    timestamp = float(item["timestamp"].item())
    video_keys = {
        camera_key: f"observation.images.{camera_key}" for camera_key in camera_keys
    }
    video_frames = dataset._query_videos(  # noqa: SLF001 - LeRobot has no selected-camera public API.
        {video_key: [timestamp] for video_key in video_keys.values()},
        episode_index,
    )
    return ReferencePose(
        dataset_path=dataset_path,
        dataset_index=index,
        episode_index=episode_index,
        frame_index=int(item["frame_index"].item()),
        target=np.asarray(item[POSE_FEATURE], dtype=np.float32),
        images={
            camera_key: tensor_image_to_uint8(video_frames[video_key])
            for camera_key, video_key in video_keys.items()
        },
    )


def load_reference_poses(
    dataset_paths: Sequence[Path],
    camera_keys: Sequence[str],
    poses_per_dataset: int,
    episode_margin_frames: int,
) -> list[ReferencePose]:
    references = []
    for dataset_path in dataset_paths:
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        dataset = load_local_dataset(dataset_path)
        missing = [key for key in camera_keys if f"observation.images.{key}" not in dataset.features]
        if missing:
            raise ValueError(f"Dataset {dataset_path} is missing camera(s): {missing}")
        if POSE_FEATURE not in dataset.features:
            raise ValueError(f"Dataset {dataset_path} is missing {POSE_FEATURE}")
        indices = select_diverse_pose_indices(dataset, poses_per_dataset, episode_margin_frames)
        for index in indices:
            references.append(
                load_reference_pose(dataset, dataset_path, index, camera_keys)
            )
    return references


def load_reference_trajectory(
    dataset: LeRobotDataset,
    dataset_path: Path,
    center_index: int,
    camera_keys: Sequence[str],
    trajectory_frames: int,
    preroll_frames: int,
) -> ReferenceTrajectory:
    center_item = dataset.hf_dataset[center_index]
    episode_index = int(center_item["episode_index"].item())
    episode = dataset.meta.episodes[episode_index]
    episode_start = int(episode["dataset_from_index"])
    episode_end = int(episode["dataset_to_index"])
    minimum_total_frames = trajectory_frames + preroll_frames
    if episode_end - episode_start < minimum_total_frames:
        raise ValueError(
            f"Episode {episode_index} in {dataset_path} has only "
            f"{episode_end - episode_start} frames; {minimum_total_frames} are required"
        )

    score_start = center_index - trajectory_frames // 2
    score_start = max(
        episode_start + preroll_frames,
        min(score_start, episode_end - trajectory_frames),
    )
    # Replaying only a short lead-in left the arm's dynamic state dependent on
    # how it reached the segment.  Start at the episode boundary so the scored
    # window has the same complete command history as data collection.
    start = episode_start
    score_end = score_start + trajectory_frames
    indices = list(range(start, score_end))
    rows = dataset.hf_dataset.select(indices)
    timestamps = [float(timestamp) for timestamp in rows["timestamp"]]
    video_keys = {
        camera_key: f"observation.images.{camera_key}" for camera_key in camera_keys
    }
    video_frames = dataset._query_videos(  # noqa: SLF001
        {video_key: timestamps for video_key in video_keys.values()},
        episode_index,
    )
    images = {
        camera_key: np.stack(
            [tensor_image_to_uint8(frame) for frame in video_frames[video_key]]
        )
        for camera_key, video_key in video_keys.items()
    }
    edge_maps = {
        camera_key: np.stack(
            [background_edge_map(image) for image in camera_images]
        )
        for camera_key, camera_images in images.items()
    }
    return ReferenceTrajectory(
        dataset_path=dataset_path,
        episode_index=episode_index,
        start_frame_index=int(rows[0]["frame_index"].item()),
        actions=np.asarray(rows["action"], dtype=np.float32),
        images=images,
        edge_maps=edge_maps,
        edge_distance_maps={
            camera_key: np.stack(
                [_edge_distance_map(edges) for edges in camera_edge_maps]
            )
            for camera_key, camera_edge_maps in edge_maps.items()
        },
        fps=float(dataset.fps),
        preroll_frames=score_start - episode_start,
    )


def load_reference_trajectories(
    dataset_paths: Sequence[Path],
    camera_keys: Sequence[str],
    trajectories_per_dataset: int,
    trajectory_frames: int,
    preroll_frames: int,
    episode_margin_frames: int,
) -> list[ReferenceTrajectory]:
    references = []
    for dataset_path in dataset_paths:
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        dataset = load_local_dataset(dataset_path)
        missing = [
            key
            for key in camera_keys
            if f"observation.images.{key}" not in dataset.features
        ]
        if missing:
            raise ValueError(f"Dataset {dataset_path} is missing camera(s): {missing}")
        if "action" not in dataset.features:
            raise ValueError(f"Dataset {dataset_path} is missing action")
        centers = select_diverse_pose_indices(
            dataset,
            trajectories_per_dataset,
            max(episode_margin_frames, (trajectory_frames + preroll_frames) // 2),
        )
        references.extend(
            load_reference_trajectory(
                dataset,
                dataset_path,
                center,
                camera_keys,
                trajectory_frames,
                preroll_frames,
            )
            for center in centers
        )
    return references


async def move_robot_to_target(
    robot: Iloha,
    right_target: np.ndarray,
    *,
    command_hz: float,
    timeout_s: float,
) -> None:
    full_target = make_full_action_from_right(right_target)
    deadline = time.monotonic() + timeout_s
    while True:
        error = np.abs(full_target - robot.old_action)
        if float(np.max(error)) <= 1e-5:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out while ramping to a calibration pose; "
                f"remaining max command delta={float(np.max(error)):.4f}"
            )
        await robot.async_send_action(
            full_target,
            use_relative=True,
            use_filter=False,
            use_unwrap=False,
        )
        await asyncio.sleep(1.0 / command_hz)


async def capture_calibration_images(
    cameras: dict,
    camera_keys: Sequence[str],
    *,
    frame_count: int,
    frame_interval_s: float,
) -> dict[str, np.ndarray]:
    frames: dict[str, list[np.ndarray]] = {key: [] for key in camera_keys}
    for frame_index in range(frame_count):
        for camera_key in camera_keys:
            camera = cameras[camera_key]
            try:
                frame = camera.read_latest(max_age_ms=CAMERA_MAX_FRAME_AGE_MS)
            except Exception:
                frame = camera.async_read(timeout_ms=CAMERA_MAX_FRAME_AGE_MS)
            frames[camera_key].append(
                preprocess_soundreal_camera_frame(camera_key, frame)
            )
        if frame_index + 1 < frame_count:
            await asyncio.sleep(frame_interval_s)
    return {
        key: np.median(np.stack(camera_frames), axis=0).astype(np.uint8)
        for key, camera_frames in frames.items()
    }


def capture_latest_calibration_images(
    cameras: dict,
    camera_keys: Sequence[str],
) -> dict[str, np.ndarray]:
    captured = {}
    for camera_key in camera_keys:
        camera = cameras[camera_key]
        try:
            frame = camera.read_latest(max_age_ms=CAMERA_MAX_FRAME_AGE_MS)
        except Exception:
            frame = camera.async_read(timeout_ms=CAMERA_MAX_FRAME_AGE_MS)
        captured[camera_key] = preprocess_soundreal_camera_frame(camera_key, frame)
    return captured


async def replay_reference_trajectory(
    robot: Iloha,
    cameras: dict,
    reference: ReferenceTrajectory,
    camera_keys: Sequence[str],
    offsets: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    # Dataset episodes are recorded after the server's staged reset.  Reproduce
    # that reset before every episode; never jump directly from the preceding
    # trajectory endpoint to the next episode's initial pose.
    await reset_robot_to_home(robot)
    await asyncio.sleep(args.post_home_wait_s)
    hardware_actions = np.stack(
        [dataset_to_hardware_action(action, offsets) for action in reference.actions]
    )

    interpolation_steps = max(1, round(args.replay_hz / reference.fps))
    replay_hz = reference.fps * interpolation_steps
    period_s = 1.0 / replay_hz
    captures: dict[str, list[np.ndarray]] = {key: [] for key in camera_keys}
    await robot.async_send_action(
        make_full_action_from_right(hardware_actions[0]),
        use_relative=False,
        use_filter=False,
        use_unwrap=False,
    )
    next_tick = time.perf_counter()

    for frame_index, current_action in enumerate(hardware_actions):
        if frame_index > 0:
            previous_action = hardware_actions[frame_index - 1]
            for substep in range(1, interpolation_steps + 1):
                next_tick += period_s
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                fraction = substep / interpolation_steps
                interpolated = previous_action + fraction * (
                    current_action - previous_action
                )
                await robot.async_send_action(
                    make_full_action_from_right(interpolated),
                    use_relative=False,
                    use_filter=False,
                    use_unwrap=False,
                )

        if frame_index >= reference.preroll_frames:
            current_images = capture_latest_calibration_images(cameras, camera_keys)
            for camera_key in camera_keys:
                captures[camera_key].append(current_images[camera_key])

    return {camera_key: np.stack(frames) for camera_key, frames in captures.items()}


def trajectory_image_difference(
    reference: ReferenceTrajectory,
    captured: dict[str, np.ndarray],
    camera_keys: Sequence[str],
    *,
    metric: str,
    max_frame_lag: int,
) -> tuple[float, dict[str, float | int]]:
    first_scored_frame = reference.preroll_frames
    scored_count = len(reference.actions) - first_scored_frame
    captured_edge_maps = {}
    captured_edge_distance_maps = {}
    if metric == "background":
        captured_edge_maps = {
            camera_key: np.stack(
                [background_edge_map(image) for image in captured[camera_key]]
            )
            for camera_key in camera_keys
        }
        captured_edge_distance_maps = {
            camera_key: np.stack(
                [_edge_distance_map(edges) for edges in captured_edge_maps[camera_key]]
            )
            for camera_key in camera_keys
        }
    lag_results = []
    for lag in range(-max_frame_lag, max_frame_lag + 1):
        capture_start = max(0, -lag)
        reference_start = first_scored_frame + max(0, lag)
        pair_count = scored_count - abs(lag)
        if pair_count <= 0:
            continue
        camera_scores = {}
        for camera_key in camera_keys:
            frame_scores = []
            for offset in range(pair_count):
                reference_index = reference_start + offset
                capture_index = capture_start + offset
                if metric == "background":
                    score = _trimmed_chamfer_from_distance_maps(
                        reference.edge_maps[camera_key][reference_index],
                        captured_edge_maps[camera_key][capture_index],
                        reference.edge_distance_maps[camera_key][reference_index],
                        captured_edge_distance_maps[camera_key][capture_index],
                        max_distance=20.0,
                        keep_fraction=0.60,
                    )
                else:
                    score = image_difference(
                        reference.images[camera_key][reference_index],
                        captured[camera_key][capture_index],
                        metric=metric,
                    )
                frame_scores.append(score)
            camera_scores[camera_key] = float(np.median(frame_scores))
        lag_results.append(
            {
                "lag": lag,
                "score": float(np.mean(list(camera_scores.values()))),
                **camera_scores,
            }
        )
    best = min(lag_results, key=lambda result: result["score"])
    return float(best["score"]), best


async def score_trajectory_offset_candidate(
    robot: Iloha,
    cameras: dict,
    references: Sequence[ReferenceTrajectory],
    camera_keys: Sequence[str],
    offsets: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, list[dict[str, float | int]]]:
    per_trajectory = []
    for trajectory_index, reference in enumerate(references, start=1):
        print(
            f"      trajectory {trajectory_index}/{len(references)}: "
            f"{reference.description()}",
            flush=True,
        )
        captured = await replay_reference_trajectory(
            robot,
            cameras,
            reference,
            camera_keys,
            offsets,
            args,
        )
        score, details = trajectory_image_difference(
            reference,
            captured,
            camera_keys,
            metric=args.image_metric,
            max_frame_lag=args.max_frame_lag,
        )
        per_trajectory.append({"score": score, **details})
    return float(np.median([item["score"] for item in per_trajectory])), per_trajectory


async def score_static_offset_candidate(
    robot: Iloha,
    cameras: dict,
    references: Sequence[ReferencePose],
    camera_keys: Sequence[str],
    offsets: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, list[dict[str, float]]]:
    captures = []
    for pose_index, reference in enumerate(references, start=1):
        target = dataset_to_hardware_action(reference.target, offsets)
        print(
            f"      pose {pose_index}/{len(references)}: {reference.description()}",
            flush=True,
        )
        await move_robot_to_target(
            robot,
            target,
            command_hz=args.command_hz,
            timeout_s=args.move_timeout_s,
        )
        await asyncio.sleep(args.settle_s)
        captures.append(
            await capture_calibration_images(
                cameras,
                camera_keys,
                frame_count=args.capture_frames,
                frame_interval_s=args.capture_interval_s,
            )
        )
    return aggregate_image_difference(
        references,
        captures,
        camera_keys,
        metric=args.image_metric,
    )


async def score_offset_candidate(
    robot: Iloha,
    cameras: dict,
    references: Sequence[ReferencePose] | Sequence[ReferenceTrajectory],
    camera_keys: Sequence[str],
    offsets: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, list[dict[str, Any]]]:
    if references and isinstance(references[0], ReferenceTrajectory):
        return await score_trajectory_offset_candidate(
            robot,
            cameras,
            references,
            camera_keys,
            offsets,
            args,
        )
    return await score_static_offset_candidate(
        robot,
        cameras,
        references,
        camera_keys,
        offsets,
        args,
    )


def _add_image_label(image: np.ndarray, label: str) -> np.ndarray:
    image = tensor_image_to_uint8(image)
    header_height = 30
    labeled = np.zeros((image.shape[0] + header_height, image.shape[1], 3), dtype=np.uint8)
    labeled[header_height:] = image
    cv2.putText(
        labeled,
        label,
        (6, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def _save_rgb_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to save comparison image: {path}")


def make_comparison_panel(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    pose_number: int,
    camera_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    reference = tensor_image_to_uint8(reference)
    current = tensor_image_to_uint8(current)
    if current.shape != reference.shape:
        current = cv2.resize(
            current,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    absolute_difference = cv2.absdiff(reference, current)
    difference_gray = cv2.cvtColor(absolute_difference, cv2.COLOR_RGB2GRAY)
    enhanced_difference = np.clip(difference_gray.astype(np.float32) * 4.0, 0, 255).astype(
        np.uint8
    )
    heatmap_bgr = cv2.applyColorMap(enhanced_difference, cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    prefix = f"pose{pose_number:02d} {camera_key}"
    panel = np.concatenate(
        [
            _add_image_label(reference, f"{prefix}: dataset"),
            _add_image_label(current, f"{prefix}: calibrated"),
            _add_image_label(heatmap, f"{prefix}: abs diff x4"),
        ],
        axis=1,
    )
    return panel, heatmap


async def save_calibration_comparisons(
    robot: Iloha,
    cameras: dict,
    references: Sequence[ReferencePose],
    camera_keys: Sequence[str],
    offsets: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    output_dir = Path(args.comparison_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_rows = []
    manifest_references = []

    for pose_number, reference in enumerate(references, start=1):
        print(f"比較画像 pose {pose_number}/{len(references)}: {reference.description()}")
        target = dataset_to_hardware_action(reference.target, offsets)
        await move_robot_to_target(
            robot,
            target,
            command_hz=args.command_hz,
            timeout_s=args.move_timeout_s,
        )
        await asyncio.sleep(args.settle_s)
        captured = await capture_calibration_images(
            cameras,
            camera_keys,
            frame_count=args.capture_frames,
            frame_interval_s=args.capture_interval_s,
        )

        camera_results = {}
        for camera_key in camera_keys:
            stem = f"pose{pose_number:02d}_{camera_key}"
            reference_image = reference.images[camera_key]
            current_image = captured[camera_key]
            panel, heatmap = make_comparison_panel(
                reference_image,
                current_image,
                pose_number=pose_number,
                camera_key=camera_key,
            )
            _save_rgb_image(output_dir / f"{stem}_dataset.png", reference_image)
            _save_rgb_image(output_dir / f"{stem}_calibrated.png", current_image)
            _save_rgb_image(output_dir / f"{stem}_difference.png", heatmap)
            _save_rgb_image(output_dir / f"{stem}_comparison.png", panel)
            overview_rows.append(panel)
            camera_results[camera_key] = {
                "image_score": image_difference(
                    reference_image,
                    current_image,
                    metric=args.image_metric,
                ),
                "dataset_image": f"{stem}_dataset.png",
                "calibrated_image": f"{stem}_calibrated.png",
                "difference_image": f"{stem}_difference.png",
                "comparison_image": f"{stem}_comparison.png",
            }

        manifest_references.append(
            {
                "dataset_path": str(reference.dataset_path),
                "dataset_index": reference.dataset_index,
                "episode_index": reference.episode_index,
                "frame_index": reference.frame_index,
                "state": reference.target.tolist(),
                "cameras": camera_results,
            }
        )

    overview = np.concatenate(overview_rows, axis=0)
    _save_rgb_image(output_dir / "overview.png", overview)
    manifest = {
        "calibration_path": str(Path(args.initial_calibration_path).resolve()),
        "joint_offsets": offsets_to_mapping(offsets),
        "image_metric": args.image_metric,
        "difference_visualization": "absolute RGB difference converted to heatmap, intensity x4",
        "references": manifest_references,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"比較画像を保存しました: {output_dir}")
    return output_dir


def make_trajectory_comparison_panel(
    reference: np.ndarray,
    baseline: np.ndarray,
    calibrated: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    reference = tensor_image_to_uint8(reference)
    baseline = tensor_image_to_uint8(baseline)
    calibrated = tensor_image_to_uint8(calibrated)
    size = (reference.shape[1], reference.shape[0])
    if baseline.shape != reference.shape:
        baseline = cv2.resize(baseline, size, interpolation=cv2.INTER_AREA)
    if calibrated.shape != reference.shape:
        calibrated = cv2.resize(calibrated, size, interpolation=cv2.INTER_AREA)

    heatmaps = []
    for current in (baseline, calibrated):
        difference = cv2.cvtColor(cv2.absdiff(reference, current), cv2.COLOR_RGB2GRAY)
        enhanced = np.clip(difference.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
        heatmaps.append(
            cv2.cvtColor(cv2.applyColorMap(enhanced, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        )
    return np.concatenate(
        [
            _add_image_label(reference, f"{label}: dataset"),
            _add_image_label(baseline, f"{label}: no offset"),
            _add_image_label(calibrated, f"{label}: calibrated"),
            _add_image_label(heatmaps[0], f"{label}: no offset diff x4"),
            _add_image_label(heatmaps[1], f"{label}: calibrated diff x4"),
        ],
        axis=1,
    )


async def save_trajectory_calibration_comparisons(
    robot: Iloha,
    cameras: dict,
    references: Sequence[ReferenceTrajectory],
    camera_keys: Sequence[str],
    baseline_offsets: np.ndarray,
    calibrated_offsets: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    output_dir = Path(args.comparison_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_rows = []
    manifest_references = []

    for trajectory_number, reference in enumerate(references, start=1):
        print(f"比較画像 trajectory {trajectory_number}/{len(references)}: {reference.description()}")
        baseline = await replay_reference_trajectory(
            robot, cameras, reference, camera_keys, baseline_offsets, args
        )
        baseline_score, baseline_details = trajectory_image_difference(
            reference,
            baseline,
            camera_keys,
            metric=args.image_metric,
            max_frame_lag=args.max_frame_lag,
        )
        calibrated = await replay_reference_trajectory(
            robot, cameras, reference, camera_keys, calibrated_offsets, args
        )
        calibrated_score, calibrated_details = trajectory_image_difference(
            reference,
            calibrated,
            camera_keys,
            metric=args.image_metric,
            max_frame_lag=args.max_frame_lag,
        )

        baseline_lag = int(baseline_details["lag"])
        calibrated_lag = int(calibrated_details["lag"])
        first_reference = reference.preroll_frames + max(0, baseline_lag, calibrated_lag)
        last_reference = len(reference.actions) - 1 + min(0, baseline_lag, calibrated_lag)
        sample_indices = np.linspace(first_reference, last_reference, 3, dtype=int)
        saved_frames = []
        for sample_number, reference_index in enumerate(sample_indices, start=1):
            baseline_index = reference_index - reference.preroll_frames - baseline_lag
            calibrated_index = reference_index - reference.preroll_frames - calibrated_lag
            for camera_key in camera_keys:
                stem = f"trajectory{trajectory_number:02d}_frame{sample_number:02d}_{camera_key}"
                reference_image = reference.images[camera_key][reference_index]
                baseline_image = baseline[camera_key][baseline_index]
                calibrated_image = calibrated[camera_key][calibrated_index]
                panel = make_trajectory_comparison_panel(
                    reference_image,
                    baseline_image,
                    calibrated_image,
                    label=f"trajectory{trajectory_number:02d} frame{reference_index}",
                )
                _save_rgb_image(output_dir / f"{stem}_dataset.png", reference_image)
                _save_rgb_image(output_dir / f"{stem}_no_offset.png", baseline_image)
                _save_rgb_image(output_dir / f"{stem}_calibrated.png", calibrated_image)
                _save_rgb_image(output_dir / f"{stem}_comparison.png", panel)
                overview_rows.append(panel)
                saved_frames.append(
                    {
                        "camera": camera_key,
                        "reference_frame_index": int(reference_index),
                        "comparison_image": f"{stem}_comparison.png",
                    }
                )

        improvement = (baseline_score - calibrated_score) / max(abs(baseline_score), 1e-9)
        print(
            f"  no offset={baseline_score:.6f}, calibrated={calibrated_score:.6f}, "
            f"improvement={improvement:.2%}"
        )
        manifest_references.append(
            {
                "dataset_path": str(reference.dataset_path),
                "episode_index": reference.episode_index,
                "baseline_score": baseline_score,
                "calibrated_score": calibrated_score,
                "relative_improvement": improvement,
                "baseline_lag": baseline_lag,
                "calibrated_lag": calibrated_lag,
                "frames": saved_frames,
            }
        )

    _save_rgb_image(output_dir / "overview.png", np.concatenate(overview_rows, axis=0))
    manifest = {
        "calibration_path": str(Path(args.compare_calibration_path).resolve()),
        "baseline_offsets": offsets_to_mapping(baseline_offsets),
        "calibrated_offsets": offsets_to_mapping(calibrated_offsets),
        "image_metric": args.image_metric,
        "references": manifest_references,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"軌道比較画像を保存しました: {output_dir}")
    return output_dir


def candidate_values(center: float, step: float, max_offset: float) -> list[float]:
    values = np.clip(
        np.asarray([center - step, center, center + step], dtype=np.float64),
        -max_offset,
        max_offset,
    )
    return sorted({round(float(value), 9) for value in values})


async def optimize_offsets(
    robot: Iloha,
    cameras: dict,
    references: Sequence[ReferencePose] | Sequence[ReferenceTrajectory],
    camera_keys: Sequence[str],
    initial_offsets: np.ndarray,
    joint_indices: Sequence[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    offsets = initial_offsets.astype(np.float32).copy()
    history: list[dict[str, Any]] = []

    for iteration in range(args.iterations):
        step = args.initial_step_rad / (2**iteration)
        print(f"\n=== iteration {iteration + 1}/{args.iterations}, step={step:.5f} ===")
        for joint_index in joint_indices:
            joint_name = RIGHT_ARM_FEATURE_NAMES[joint_index]
            center = float(offsets[joint_index])
            values = candidate_values(center, step, args.max_offset_rad)
            evaluations = []
            print(f"  {joint_name}: candidates={values}")
            for value in values:
                trial_offsets = offsets.copy()
                trial_offsets[joint_index] = value
                print(f"    {joint_name}={value:+.5f}")
                score, per_pose = await score_offset_candidate(
                    robot,
                    cameras,
                    references,
                    camera_keys,
                    trial_offsets,
                    args,
                )
                print(f"      image score={score:.6f}")
                evaluations.append(
                    {
                        "offset": value,
                        "score": score,
                        "per_pose": per_pose,
                    }
                )

            center_eval = min(evaluations, key=lambda result: abs(result["offset"] - center))
            best_eval = min(evaluations, key=lambda result: result["score"])
            improvement = (center_eval["score"] - best_eval["score"]) / max(
                abs(center_eval["score"]), 1e-9
            )
            accepted = (
                best_eval["offset"] != center_eval["offset"]
                and improvement >= args.min_relative_improvement
            )
            if accepted:
                offsets[joint_index] = best_eval["offset"]
            selected = float(offsets[joint_index])
            print(
                f"    selected={selected:+.5f}, relative improvement={improvement:.4%}"
                + ("" if accepted else " (unchanged)")
            )
            history.append(
                {
                    "iteration": iteration + 1,
                    "step": step,
                    "joint": joint_name,
                    "center_offset": center,
                    "selected_offset": selected,
                    "relative_improvement": improvement,
                    "accepted": accepted,
                    "evaluations": evaluations,
                }
            )

    return offsets, history


async def measure_repeatability(
    robot: Iloha,
    cameras: dict,
    references: Sequence[ReferencePose] | Sequence[ReferenceTrajectory],
    camera_keys: Sequence[str],
    offsets: np.ndarray,
    args: argparse.Namespace,
) -> list[float]:
    scores = []
    for run in range(args.repeatability_runs):
        print(f"\n=== repeatability {run + 1}/{args.repeatability_runs} ===")
        score, per_reference = await score_offset_candidate(
            robot,
            cameras,
            references,
            camera_keys,
            offsets,
            args,
        )
        scores.append(score)
        print(f"repeatability score={score:.6f}")
        for reference_index, details in enumerate(per_reference, start=1):
            lag = details.get("lag")
            lag_text = "" if lag is None else f", lag={lag:+d} frames"
            print(
                f"  reference {reference_index}: "
                f"score={details['score']:.6f}{lag_text}"
            )
    mean_score = float(np.mean(scores))
    relative_range = (max(scores) - min(scores)) / max(abs(mean_score), 1e-9)
    print(
        f"repeatability mean={mean_score:.6f}, "
        f"range={max(scores) - min(scores):.6f} ({relative_range:.2%})"
    )
    return scores


def print_reference_plan(
    references: Sequence[ReferencePose],
    initial_offsets: np.ndarray,
    args: argparse.Namespace,
) -> None:
    targets = np.stack([reference.target for reference in references])
    print("=" * 72)
    print(f"参照姿勢数: {len(references)}")
    for number, reference in enumerate(references, start=1):
        print(f"  {number}: {reference.description()}")
        print(f"     state={np.array2string(reference.target, precision=4)}")
    print(f"参照 state 最小値: {np.array2string(targets.min(axis=0), precision=4)}")
    print(f"参照 state 最大値: {np.array2string(targets.max(axis=0), precision=4)}")
    print(f"初期オフセット: {offsets_summary(initial_offsets)}")
    print(
        "探索関節: "
        + ", ".join(RIGHT_ARM_FEATURE_NAMES[index] for index in args.joints)
    )
    moves = len(references) * 3 * len(args.joints) * args.iterations
    print(f"最大探索移動回数: 約 {moves}")
    print("=" * 72)


def print_trajectory_plan(
    references: Sequence[ReferenceTrajectory],
    initial_offsets: np.ndarray,
    args: argparse.Namespace,
) -> None:
    print("=" * 72)
    print(f"参照軌道数: {len(references)}")
    for number, reference in enumerate(references, start=1):
        scored_frames = len(reference.actions) - reference.preroll_frames
        print(f"  {number}: {reference.description()}")
        print(
            f"     助走={reference.preroll_frames} frames, "
            f"損失対象={scored_frames} frames, fps={reference.fps:g}"
        )
    print(f"初期オフセット: {offsets_summary(initial_offsets)}")
    print(
        "探索関節: "
        + ", ".join(RIGHT_ARM_FEATURE_NAMES[index] for index in args.joints)
    )
    candidate_runs = 3 * len(args.joints) * args.iterations
    trajectory_seconds = sum(len(reference.actions) / reference.fps for reference in references)
    print(
        f"候補評価回数: 最大 {candidate_runs}、"
        f"軌道再生時間: 約 {candidate_runs * trajectory_seconds / 60:.1f} 分"
    )
    print("=" * 72)


def make_robot_config(args: argparse.Namespace) -> IlohaConfig:
    return IlohaConfig(
        right_dynamixel_port=args.right_dynamixel_port,
        right_robstride_port=args.right_robstride_port,
        left_robstride_port=args.left_robstride_port,
        left_dynamixel_port=args.left_dynamixel_port,
        enable_left_arm=False,
        enable_right_arm=True,
        max_relative_target_1=args.max_relative_step,
        max_relative_target_2=args.max_relative_step,
        max_relative_target_3=args.max_relative_step,
        max_relative_target_4=args.max_relative_step,
        max_relative_target_5=args.max_relative_step,
        max_relative_target_6=args.max_relative_step,
        current_limit_gripper_R=args.gripper_current_limit,
        current_limit_gripper_L=0.3,
    )


async def main(args: argparse.Namespace) -> None:
    dataset_paths = [Path(path).resolve() for path in args.dataset_paths]
    camera_keys = args.camera_keys
    if args.verify_only and args.initial_calibration_path is None:
        args.initial_calibration_path = args.output_path
    initial_offsets = load_joint_offsets(
        args.initial_calibration_path,
        required=args.initial_calibration_path is not None,
    )
    comparison_offsets = None
    if args.compare_calibration_path is not None:
        comparison_offsets = load_joint_offsets(args.compare_calibration_path, required=True)
    if np.max(np.abs(initial_offsets)) > args.max_offset_rad:
        raise ValueError(
            "Initial calibration contains an offset larger than --max-offset-rad"
        )

    if args.verify_only:
        print("比較用の参照姿勢をデータセットから読み込んでいます...")
        references = load_reference_poses(
            dataset_paths,
            camera_keys,
            poses_per_dataset=args.poses_per_dataset,
            episode_margin_frames=args.episode_margin_frames,
        )
        print_reference_plan(references, initial_offsets, args)
    else:
        print("参照軌道をデータセットから読み込んでいます...")
        references = load_reference_trajectories(
            dataset_paths,
            camera_keys,
            trajectories_per_dataset=args.poses_per_dataset,
            trajectory_frames=args.trajectory_frames,
            preroll_frames=args.trajectory_preroll_frames,
            episode_margin_frames=args.episode_margin_frames,
        )
        print_trajectory_plan(references, initial_offsets, args)
    if args.dry_run:
        print("--dry-run のため、ロボットには接続せず終了します。")
        return

    if not args.yes:
        answer = input(
            "周囲と非常停止手段を確認しましたか？ ロボットを動かすには 'yes' と入力: "
        ).strip()
        if answer != "yes":
            print("キャリブレーションを中止しました。")
            return

    robot: Iloha | None = None
    cameras: dict = {}
    try:
        config = make_robot_config(args)
        await resolve_auto_robstride_ports(config)
        robot = Iloha(config, debug=False)
        print("ロボットに接続しています...")
        await robot.connect()
        await reset_robot_to_home(robot)

        cameras = initialize_cameras()
        missing_cameras = [key for key in camera_keys if key not in cameras]
        if missing_cameras:
            raise RuntimeError(f"Requested camera(s) were not initialized: {missing_cameras}")

        if args.repeatability_only:
            await measure_repeatability(
                robot,
                cameras,
                references,
                camera_keys,
                initial_offsets,
                args,
            )
            return

        if comparison_offsets is not None:
            await save_trajectory_calibration_comparisons(
                robot,
                cameras,
                references,
                camera_keys,
                initial_offsets,
                comparison_offsets,
                args,
            )
            return

        if args.verify_only:
            await save_calibration_comparisons(
                robot,
                cameras,
                references,
                camera_keys,
                initial_offsets,
                args,
            )
            return

        for warmup_run in range(args.calibration_warmup_runs):
            print(
                f"\n=== trajectory warmup "
                f"{warmup_run + 1}/{args.calibration_warmup_runs} ==="
            )
            await score_offset_candidate(
                robot,
                cameras,
                references,
                camera_keys,
                initial_offsets,
                args,
            )

        offsets, history = await optimize_offsets(
            robot,
            cameras,
            references,
            camera_keys,
            initial_offsets,
            args.joints,
            args,
        )

        details = {
            "method": "camera_trajectory_coordinate_descent",
            "trajectory_feature": "action",
            "joints_calibrated": [RIGHT_ARM_FEATURE_NAMES[index] for index in args.joints],
            "initial_offsets": offsets_to_mapping(initial_offsets),
            "parameters": {
                "poses_per_dataset": args.poses_per_dataset,
                "episode_margin_frames": args.episode_margin_frames,
                "iterations": args.iterations,
                "initial_step_rad": args.initial_step_rad,
                "max_offset_rad": args.max_offset_rad,
                "min_relative_improvement": args.min_relative_improvement,
                "capture_frames": args.capture_frames,
                "settle_s": args.settle_s,
                "image_metric": args.image_metric,
                "trajectory_frames": args.trajectory_frames,
                "trajectory_preroll_frames": args.trajectory_preroll_frames,
                "replay_hz": args.replay_hz,
                "max_frame_lag": args.max_frame_lag,
            },
            "references": [
                {
                    "dataset_path": str(reference.dataset_path),
                    "episode_index": reference.episode_index,
                    "start_frame_index": reference.start_frame_index,
                    "frame_count": len(reference.actions),
                    "preroll_frames": reference.preroll_frames,
                }
                for reference in references
            ],
            "history": history,
        }
        document = make_calibration_document(
            offsets,
            dataset_paths=dataset_paths,
            camera_keys=camera_keys,
            calibration_details=details,
        )
        output_path = save_calibration_file(args.output_path, document)
        print("=" * 72)
        print(f"キャリブレーション完了: {offsets_summary(offsets)}")
        print(f"保存先: {output_path}")
        print(f"評価時: uv run iloha_eval.py ... --calibration-path {output_path}")
    finally:
        for name, camera in cameras.items():
            try:
                camera.disconnect()
                print(f"{name} を切断しました")
            except Exception as exc:
                print(f"{name} 切断エラー: {exc}")
        if robot is not None:
            try:
                await reset_robot_to_home(robot, init=False)
            except Exception as exc:
                print(f"2段階初期化エラー: {exc}")
            try:
                await robot.disconnect()
                print("ロボット切断完了")
            except Exception as exc:
                print(f"ロボット切断エラー: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iloha カメラ画像ベース関節オフセット校正")
    parser.add_argument(
        "--dataset-paths",
        nargs="+",
        default=[str(path) for path in DEFAULT_DATASET_PATHS],
        help="参照する LeRobot dataset（複数指定可）",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_CALIBRATION_PATH),
        help="iloha_eval.py が読み込む calibration JSON の保存先",
    )
    parser.add_argument(
        "--initial-calibration-path",
        default=None,
        help="既存 calibration を初期値として再校正する場合の JSON",
    )
    parser.add_argument(
        "--camera-keys",
        type=parse_csv_names,
        default=list(DEFAULT_CAMERA_KEYS),
        help="画像差分に使うカメラ名（カンマ区切り）",
    )
    parser.add_argument(
        "--joints",
        type=parse_joint_indices,
        default=list(range(6)),
        help="校正する右腕関節番号（既定: 1,2,3,4,5,6）。joint7 は正規化グリッパー値",
    )
    parser.add_argument("--poses-per-dataset", type=int, default=1)
    parser.add_argument("--episode-margin-frames", type=int, default=15)
    parser.add_argument(
        "--trajectory-frames",
        type=int,
        default=90,
        help="助走後に画像損失へ使う連続フレーム数",
    )
    parser.add_argument(
        "--trajectory-preroll-frames",
        type=int,
        default=30,
        help="軌道先頭で再生するが画像損失から除外するフレーム数",
    )
    parser.add_argument(
        "--replay-hz",
        type=float,
        default=60.0,
        help="action 基準点間を補間してロボットへ送る周期",
    )
    parser.add_argument(
        "--max-frame-lag",
        type=int,
        default=2,
        help="動画比較時に許す時間方向の最大ずれ（フレーム）",
    )
    parser.add_argument(
        "--calibration-warmup-runs",
        type=int,
        default=1,
        help="最適化前に損失へ使わず再生する全軌道の巡回数",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--initial-step-rad", type=float, default=0.04)
    parser.add_argument("--max-offset-rad", type=float, default=0.15)
    parser.add_argument("--min-relative-improvement", type=float, default=0.002)
    parser.add_argument(
        "--image-metric",
        choices=("full", "structural", "background"),
        default="background",
        help="full: 全画像、structural: 長い線、background: 外れ値を除いた背景エッジ",
    )
    parser.add_argument("--capture-frames", type=int, default=3)
    parser.add_argument("--capture-interval-s", type=float, default=0.04)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--post-home-wait-s", type=float, default=2.0)
    parser.add_argument("--command-hz", type=float, default=float(CAMERA_FPS))
    parser.add_argument("--move-timeout-s", type=float, default=30.0)
    parser.add_argument("--max-relative-step", type=float, default=0.03)
    parser.add_argument("--gripper-current-limit", type=float, default=0.2)
    parser.add_argument("--right-dynamixel-port", default="/dev/ttyUSB_RightDynamixel")
    parser.add_argument("--left-dynamixel-port", default="/dev/ttyUSB_LeftDynamixel")
    parser.add_argument("--right-robstride-port", default="auto")
    parser.add_argument("--left-robstride-port", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="参照姿勢だけ確認し実機接続しない")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="保存済み calibration で参照姿勢へ移動し、比較画像だけ保存する",
    )
    parser.add_argument(
        "--repeatability-only",
        action="store_true",
        help="同じオフセットを繰り返し測定し、スコア再現性だけ確認する",
    )
    parser.add_argument("--repeatability-runs", type=int, default=3)
    parser.add_argument(
        "--comparison-dir",
        default="calibration_comparison",
        help="--verify-only の比較画像保存先",
    )
    parser.add_argument(
        "--compare-calibration-path",
        default=None,
        help="同じ参照軌道をオフセットなし／指定calibrationありで再生して比較画像を保存する",
    )
    parser.add_argument("--yes", action="store_true", help="実機移動前の確認入力を省略する")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if sum((args.verify_only, args.repeatability_only, args.compare_calibration_path is not None)) > 1:
        parser.error(
            "--verify-only, --repeatability-only, --compare-calibration-path are mutually exclusive"
        )
    positive_fields = (
        "poses_per_dataset",
        "iterations",
        "initial_step_rad",
        "max_offset_rad",
        "capture_frames",
        "capture_interval_s",
        "command_hz",
        "move_timeout_s",
        "max_relative_step",
        "repeatability_runs",
        "trajectory_frames",
        "replay_hz",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.episode_margin_frames < 0:
        parser.error("--episode-margin-frames must be non-negative")
    if args.trajectory_preroll_frames < 0:
        parser.error("--trajectory-preroll-frames must be non-negative")
    if args.max_frame_lag < 0:
        parser.error("--max-frame-lag must be non-negative")
    if args.max_frame_lag >= args.trajectory_frames:
        parser.error("--max-frame-lag must be smaller than --trajectory-frames")
    if args.calibration_warmup_runs < 0:
        parser.error("--calibration-warmup-runs must be non-negative")
    if args.settle_s < 0:
        parser.error("--settle-s must be non-negative")
    if args.post_home_wait_s < 0:
        parser.error("--post-home-wait-s must be non-negative")
    if not 0 <= args.min_relative_improvement < 1:
        parser.error("--min-relative-improvement must be in [0, 1)")
    if any(index == 6 for index in args.joints) and not math.isclose(
        args.initial_step_rad, 0.04
    ):
        print(
            "注意: joint7 の単位は rad ではなく normalized_gripper です。"
            " --initial-step-rad の数値をその単位として使用します。"
        )


if __name__ == "__main__":
    argument_parser = build_parser()
    arguments = argument_parser.parse_args()
    validate_args(argument_parser, arguments)
    asyncio.run(main(arguments))
