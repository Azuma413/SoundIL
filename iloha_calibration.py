"""Shared calibration file support for the Iloha real-robot scripts."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from soundreal_utils import RIGHT_ARM_DIM, RIGHT_ARM_FEATURE_NAMES


CALIBRATION_FORMAT_VERSION = 1
DEFAULT_CALIBRATION_PATH = Path("iloha_calibration.json")
CALIBRATION_ROBOT_TYPE = "iloha_single_arm"
JOINT_UNITS = ("rad", "rad", "rad", "rad", "rad", "rad", "normalized_gripper")


def zero_joint_offsets() -> np.ndarray:
    return np.zeros(RIGHT_ARM_DIM, dtype=np.float32)


def _validate_offsets(offsets: Sequence[float]) -> np.ndarray:
    array = np.asarray(offsets, dtype=np.float64)
    if array.shape != (RIGHT_ARM_DIM,):
        raise ValueError(
            f"Expected {RIGHT_ARM_DIM} Iloha joint offsets, got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Iloha joint offsets must all be finite")
    return array.astype(np.float32)


def offsets_to_mapping(offsets: Sequence[float]) -> dict[str, float]:
    array = _validate_offsets(offsets)
    return {
        joint_name: float(array[index])
        for index, joint_name in enumerate(RIGHT_ARM_FEATURE_NAMES)
    }


def offsets_from_mapping(offsets: Mapping[str, Any]) -> np.ndarray:
    missing = [name for name in RIGHT_ARM_FEATURE_NAMES if name not in offsets]
    extra = sorted(set(offsets) - set(RIGHT_ARM_FEATURE_NAMES))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError("Invalid joint_offsets keys: " + ", ".join(details))
    return _validate_offsets([offsets[name] for name in RIGHT_ARM_FEATURE_NAMES])


def make_calibration_document(
    offsets: Sequence[float],
    *,
    dataset_paths: Sequence[str | Path],
    camera_keys: Sequence[str],
    calibration_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    array = _validate_offsets(offsets)
    return {
        "format_version": CALIBRATION_FORMAT_VERSION,
        "robot_type": CALIBRATION_ROBOT_TYPE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "joint_offsets": offsets_to_mapping(array),
        "joint_units": dict(zip(RIGHT_ARM_FEATURE_NAMES, JOINT_UNITS, strict=True)),
        "convention": (
            "hardware_command = dataset_command + joint_offset; "
            "dataset_state = hardware_command_state - joint_offset"
        ),
        "dataset_paths": [str(Path(path).resolve()) for path in dataset_paths],
        "camera_keys": list(camera_keys),
        "calibration": dict(calibration_details or {}),
    }


def save_calibration_file(path: str | Path, document: Mapping[str, Any]) -> Path:
    """Atomically save a calibration document."""

    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            file.write(serialized)
            file.flush()
            os.fsync(file.fileno())
            temporary_path = Path(file.name)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path


def load_calibration_file(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    calibration_path = Path(path).resolve()
    try:
        document = json.loads(calibration_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid calibration JSON: {calibration_path}: {exc}") from exc

    if not isinstance(document, dict):
        raise ValueError(f"Calibration file must contain a JSON object: {calibration_path}")
    if document.get("format_version") != CALIBRATION_FORMAT_VERSION:
        raise ValueError(
            "Unsupported calibration format_version: "
            f"{document.get('format_version')!r} "
            f"(expected {CALIBRATION_FORMAT_VERSION})"
        )
    if document.get("robot_type") != CALIBRATION_ROBOT_TYPE:
        raise ValueError(
            f"Calibration robot_type must be {CALIBRATION_ROBOT_TYPE!r}, "
            f"got {document.get('robot_type')!r}"
        )
    joint_offsets = document.get("joint_offsets")
    if not isinstance(joint_offsets, dict):
        raise ValueError("Calibration file is missing the joint_offsets object")

    offsets = offsets_from_mapping(joint_offsets)
    return offsets, document


def load_joint_offsets(
    path: str | Path | None,
    *,
    required: bool = False,
) -> np.ndarray:
    if path is None:
        if required:
            raise ValueError("A calibration path is required")
        return zero_joint_offsets()

    calibration_path = Path(path).resolve()
    if not calibration_path.is_file():
        if required:
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
        return zero_joint_offsets()
    offsets, _ = load_calibration_file(calibration_path)
    return offsets


def dataset_to_hardware_action(
    right_action: Sequence[float],
    offsets: Sequence[float],
) -> np.ndarray:
    return _validate_offsets(right_action) + _validate_offsets(offsets)


def hardware_to_dataset_state(
    right_state: Sequence[float],
    offsets: Sequence[float],
) -> np.ndarray:
    return _validate_offsets(right_state) - _validate_offsets(offsets)


def offsets_summary(offsets: Sequence[float]) -> str:
    array = _validate_offsets(offsets)
    parts = []
    for index, (name, unit) in enumerate(
        zip(RIGHT_ARM_FEATURE_NAMES, JOINT_UNITS, strict=True)
    ):
        value = float(array[index])
        if unit == "rad":
            parts.append(f"{name}={value:+.5f} rad ({math.degrees(value):+.2f} deg)")
        else:
            parts.append(f"{name}={value:+.5f} {unit}")
    return ", ".join(parts)
