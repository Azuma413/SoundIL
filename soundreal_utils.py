from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import sounddevice as sd
import torch
from pydub import AudioSegment

from env.tasks.sound_camera import SoundCamera, SoundConfig


SOUNDREAL_TASK_NAME = "soundReal-m4-f10-s2-p0"
DEFAULT_SOUND_CONFIG = SoundConfig()

OBSERVATION_HEIGHT = 224
OBSERVATION_WIDTH = 224
CAMERA_FPS = 30
AUDIO_FPS = 10
AUDIO_SAMPLE_RATE = 16000
AUDIO_WINDOW_SECONDS = DEFAULT_SOUND_CONFIG.processing_time
SPECTROGRAM_NFFT = DEFAULT_SOUND_CONFIG.nfft
SPECTROGRAM_NUM_PEAKS = DEFAULT_SOUND_CONFIG.num_peaks
DEFAULT_MIC_CHANNELS = 8

RIGHT_ARM_FEATURE_NAMES = tuple(f"joint{i}" for i in range(1, 8))
RIGHT_ARM_START_INDEX = 7
RIGHT_ARM_DIM = len(RIGHT_ARM_FEATURE_NAMES)

SOUNDREAL_CAMERA_CONFIGS = {
    "front": {
        "serial_number_or_name": "029522250086",
        "width": 640,
        "height": 480,
        "fps": CAMERA_FPS,
    },
    "side": {
        "serial_number_or_name": "146222252104",
        "width": 640,
        "height": 480,
        "fps": CAMERA_FPS,
    },
}

DEFAULT_CAMERA_CONFIGS = {
    "cam_high": {
        "serial_number_or_name": "029522250086",
        "width": 640,
        "height": 480,
        "fps": CAMERA_FPS,
    },
    "cam_left_wrist": {
        "serial_number_or_name": "341522301205",
        "width": 640,
        "height": 480,
        "fps": CAMERA_FPS,
    },
    "cam_right_wrist": {
        "serial_number_or_name": "146222252104",
        "width": 640,
        "height": 480,
        "fps": CAMERA_FPS,
    },
}

SOUNDREAL_IMAGE_KEYS = ("front", "side", "sound0", "sound1", "spec")

# 順序に意味はない。以下のコマンドで調べられる。
# uv run workspace/tamago_test.py --list-devices
TAMAGO_DEVICE_IDS = [0, 1, 2, 3]
# 右上のTAMAGOから時計回りに並べたときのhwインデックス
RECTANGLE_HW_ORDER_CLOCKWISE = [2, 1, 0, 5]

AZIMUTH_OFFSET_DEG = 0.0
RECTANGLE_LONG_SIDE_M = 1.2
RECTANGLE_SHORT_SIDE_M = 0.6
ARRAY_HEIGHT_M = 0.1
ROOM_CENTER_XY = np.array([5.0, 5.0], dtype=np.float32)
MAP_SIZE_M = 1.4
DOA_DISTANCE_FLOOR_M = 0.0
DOA_DISTANCE_DECAY_EXPONENT = 0.0
COMBINED_MAP_POWER = 4.0
SOUNDDEVICE_CAPTURE_CHUNK_SECONDS = 0.02
SOUNDDEVICE_CAPTURE_LATENCY = "low"
SOUNDDEVICE_READ_TIMEOUT_S = 2.0
SOUNDS_DIR = Path("sounds")
SOUND_FILES = {
    0: SOUNDS_DIR / "0.wav",
    1: SOUNDS_DIR / "1.wav",
}
SOUND_LABELS = {
    0: "A",
    1: "B",
}


@dataclass(frozen=True)
class SoundEpisodeCondition:
    sound_index: int
    sound_label: str
    sound_path: Path
    speaker: str


class DummyTarget:
    def get_pos(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


def is_soundreal_task(task: str) -> bool:
    return task == SOUNDREAL_TASK_NAME


def get_camera_configs(task: str) -> dict:
    if is_soundreal_task(task):
        return SOUNDREAL_CAMERA_CONFIGS
    return DEFAULT_CAMERA_CONFIGS


def build_soundreal_dataset_features() -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (RIGHT_ARM_DIM,),
            "names": RIGHT_ARM_FEATURE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (RIGHT_ARM_DIM,),
            "names": RIGHT_ARM_FEATURE_NAMES,
        },
    }
    for key in SOUNDREAL_IMAGE_KEYS:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": (OBSERVATION_HEIGHT, OBSERVATION_WIDTH, 3),
            "names": ("height", "width", "channels"),
        }
    return features


def right_arm_full_slice(full_action: np.ndarray) -> np.ndarray:
    return np.asarray(full_action, dtype=np.float32)[RIGHT_ARM_START_INDEX : RIGHT_ARM_START_INDEX + RIGHT_ARM_DIM]


def full_action_to_right_feature_dict(full_action: np.ndarray) -> dict[str, float]:
    right_action = right_arm_full_slice(full_action)
    return {name: float(right_action[idx]) for idx, name in enumerate(RIGHT_ARM_FEATURE_NAMES)}


def right_array_to_feature_dict(right_action: np.ndarray) -> dict[str, float]:
    right_action = np.asarray(right_action, dtype=np.float32)
    return {name: float(right_action[idx]) for idx, name in enumerate(RIGHT_ARM_FEATURE_NAMES)}


def feature_dict_to_right_array(values: dict[str, float]) -> np.ndarray:
    return np.array([values[name] for name in RIGHT_ARM_FEATURE_NAMES], dtype=np.float32)


def make_full_action_from_right(right_action: np.ndarray) -> np.ndarray:
    full_action = np.zeros(14, dtype=np.float32)
    full_action[RIGHT_ARM_START_INDEX : RIGHT_ARM_START_INDEX + RIGHT_ARM_DIM] = np.asarray(
        right_action, dtype=np.float32
    )
    return full_action


def decode_right_arm_action_packet(data: bytes) -> tuple[int, Optional[np.ndarray]]:
    if not data:
        return 0, None

    mode = data[0]
    if len(data) >= 57:
        raw = [
            float(np.frombuffer(data[1 + idx * 4 : 5 + idx * 4], dtype="<f4")[0])
            for idx in range(14)
        ]
        right = raw[RIGHT_ARM_START_INDEX : RIGHT_ARM_START_INDEX + RIGHT_ARM_DIM]
    elif len(data) >= 29:
        right = [
            float(np.frombuffer(data[1 + idx * 4 : 5 + idx * 4], dtype="<f4")[0])
            for idx in range(RIGHT_ARM_DIM)
        ]
    else:
        return mode, None

    right[0] -= np.pi / 2
    right[1] -= np.pi / 2
    right[2] = -right[2] - np.pi / 2
    right[3] = -right[3]
    right[4] = -right[4]
    right[5] = -right[5]
    return mode, np.asarray(right, dtype=np.float32)


def preprocess_camera_frame(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    resized = cv2.resize(
        image,
        (OBSERVATION_WIDTH, OBSERVATION_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    return np.ascontiguousarray(resized)


def sample_sound_episode_condition(rng: np.random.Generator) -> SoundEpisodeCondition:
    sound_index = int(rng.integers(0, len(SOUND_FILES)))
    speaker = "left" if int(rng.integers(0, 2)) == 0 else "right"
    return SoundEpisodeCondition(
        sound_index=sound_index,
        sound_label=SOUND_LABELS[sound_index],
        sound_path=SOUND_FILES[sound_index],
        speaker=speaker,
    )


def resolve_output_device(device: int | None = None) -> tuple[int, dict]:
    if device is None:
        _, default_output = sd.default.device
        if default_output is None or int(default_output) < 0:
            raise RuntimeError("No default output device is configured.")
        device = int(default_output)
    return int(device), sd.query_devices(device, "output")


def load_output_audio(sound_path: Path, samplerate: int) -> np.ndarray:
    segment = AudioSegment.from_file(sound_path)
    segment = segment.set_frame_rate(samplerate).set_channels(1)
    scale = float(1 << (8 * segment.sample_width - 1))
    mono = np.asarray(segment.get_array_of_samples(), dtype=np.float32) / scale
    if mono.size == 0:
        raise RuntimeError(f"Empty audio file: {sound_path}")
    return mono


class LoopingStereoPlayer:
    def __init__(self, output_device: int | None = None):
        self.output_device = output_device
        self.stream: Optional[sd.OutputStream] = None
        self.audio = np.zeros((1, 2), dtype=np.float32)
        self.position = 0
        self.samplerate = AUDIO_SAMPLE_RATE
        self._lock = threading.Lock()

    def start(self, condition: SoundEpisodeCondition) -> None:
        self.stop()
        device_id, device_info = resolve_output_device(self.output_device)
        self.samplerate = int(device_info["default_samplerate"])
        mono = load_output_audio(condition.sound_path, self.samplerate)
        stereo = np.zeros((mono.shape[0], 2), dtype=np.float32)
        channel_index = 0 if condition.speaker == "left" else 1
        stereo[:, channel_index] = mono
        with self._lock:
            self.audio = stereo
            self.position = 0

        def callback(outdata, frames, _time_info, status):
            if status:
                print(f"[soundreal] OutputStream status: {status}")
            with self._lock:
                total = self.audio.shape[0]
                if total == 0:
                    outdata.fill(0)
                    return
                remaining = frames
                offset = 0
                while remaining > 0:
                    chunk = min(remaining, total - self.position)
                    outdata[offset : offset + chunk] = self.audio[self.position : self.position + chunk]
                    self.position = (self.position + chunk) % total
                    remaining -= chunk
                    offset += chunk

        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=2,
            dtype="float32",
            callback=callback,
            device=device_id,
            blocksize=0,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.stop()
        finally:
            self.stream.close()
            self.stream = None


def query_input_devices() -> list[tuple[int, dict]]:
    return [(idx, device) for idx, device in enumerate(sd.query_devices()) if int(device["max_input_channels"]) > 0]


def extract_hw_index(device_name: str) -> int | None:
    match = re.search(r"\(hw:(\d+),\d+\)", device_name)
    return int(match.group(1)) if match else None


def parse_device_ids_env(var_name: str) -> list[int] | None:
    raw = os.environ.get(var_name)
    if not raw:
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def select_tamago_devices(explicit_ids: list[int] | None = None) -> list[tuple[int, dict]]:
    inputs = dict(query_input_devices())
    if explicit_ids:
        missing = [device_id for device_id in explicit_ids if device_id not in inputs]
        if missing:
            raise RuntimeError(f"Input device(s) not found: {missing}")
        return [(device_id, inputs[device_id]) for device_id in explicit_ids]

    selected = [
        (device_id, inputs[device_id])
        for device_id in TAMAGO_DEVICE_IDS
        if device_id in inputs and "tamago" in str(inputs[device_id]["name"]).lower()
    ]
    if len(selected) == len(TAMAGO_DEVICE_IDS):
        return selected

    candidates = [
        (device_id, device)
        for device_id, device in inputs.items()
        if "tamago" in str(device["name"]).lower()
    ]
    if len(candidates) >= len(TAMAGO_DEVICE_IDS):
        return sorted(candidates, key=lambda item: item[0])[: len(TAMAGO_DEVICE_IDS)]

    available = "\n".join(
        f"[{idx}] {device['name']} | input_channels={int(device['max_input_channels'])}"
        for idx, device in query_input_devices()
    )
    raise RuntimeError(
        "Could not auto-select four TAMAGO devices. "
        "Set SOUNDREAL_DEVICE_IDS or connect the arrays.\n"
        f"Available input devices:\n{available}"
    )


def build_rectangular_mic_positions(devices: list[tuple[int, dict]]) -> list[list[float]]:
    half_long = RECTANGLE_LONG_SIDE_M / 2.0
    half_short = RECTANGLE_SHORT_SIDE_M / 2.0
    center_x, center_y = ROOM_CENTER_XY
    corners = [
        [center_x + half_long, center_y + half_short, ARRAY_HEIGHT_M],
        [center_x + half_long, center_y - half_short, ARRAY_HEIGHT_M],
        [center_x - half_long, center_y - half_short, ARRAY_HEIGHT_M],
        [center_x - half_long, center_y + half_short, ARRAY_HEIGHT_M],
    ]
    by_hw = dict(zip(RECTANGLE_HW_ORDER_CLOCKWISE, corners))
    positions = []
    for device_id, device in devices:
        hw_index = extract_hw_index(str(device["name"]))
        if hw_index not in by_hw:
            expected = ", ".join(f"hw:{idx}" for idx in RECTANGLE_HW_ORDER_CLOCKWISE)
            raise RuntimeError(
                f"Device [{device_id}] {device['name']} does not match the expected physical layout ({expected})."
            )
        positions.append(by_hw[hw_index])
    return positions

@dataclass
class SoundDeviceStreamState:
    device_id: int
    device_name: str
    stream: Optional[sd.InputStream]
    chunks: deque[np.ndarray]
    queued_frames: int
    condition: threading.Condition
    statuses: deque[str]
    error: Optional[str]


class ContinuousSoundDeviceCapture:
    def __init__(
        self,
        devices: list[tuple[int, dict]],
        samplerate: int,
        channels: int,
        chunk_frames: int,
        read_timeout_s: float = SOUNDDEVICE_READ_TIMEOUT_S,
    ):
        self.devices = devices
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.chunk_frames = max(1, int(chunk_frames))
        self.read_timeout_s = max(0.1, float(read_timeout_s))
        self.streams: list[SoundDeviceStreamState] = []

    def start(self) -> None:
        if self.streams:
            return
        try:
            for device_id, device in self.devices:
                available_channels = int(device["max_input_channels"])
                if available_channels < self.channels:
                    raise RuntimeError(
                        f"Device {device_id} has only {available_channels} input channels, but {self.channels} are required."
                    )
                condition = threading.Condition()
                chunk_queue: deque[np.ndarray] = deque()
                statuses: deque[str] = deque(maxlen=20)
                state = SoundDeviceStreamState(
                    device_id=device_id,
                    device_name=str(device["name"]),
                    stream=None,
                    chunks=chunk_queue,
                    queued_frames=0,
                    condition=condition,
                    statuses=statuses,
                    error=None,
                )
                stream = sd.InputStream(
                    device=device_id,
                    samplerate=self.samplerate,
                    channels=self.channels,
                    dtype="float32",
                    blocksize=self.chunk_frames,
                    latency=SOUNDDEVICE_CAPTURE_LATENCY,
                    callback=self._make_input_callback(state),
                )
                state.stream = stream
                stream.start()
                self.streams.append(state)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if not self.streams:
            return
        for state in self.streams:
            with state.condition:
                state.error = state.error or "capture stopped"
                state.condition.notify_all()
            try:
                if state.stream is not None:
                    state.stream.stop()
            except Exception:
                pass
            if state.stream is not None:
                state.stream.close()
        self.streams.clear()

    def read_chunk(self, num_frames: int) -> list[np.ndarray]:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if not self.streams:
            raise RuntimeError("ContinuousSoundDeviceCapture is not started.")
        return [self._read_stream_chunk(state, num_frames) for state in self.streams]

    def _read_stream_chunk(self, state: SoundDeviceStreamState, num_frames: int) -> np.ndarray:
        deadline = time.monotonic() + self.read_timeout_s + (num_frames / max(1, self.samplerate))
        with state.condition:
            while True:
                if state.error is not None:
                    raise RuntimeError(f"device {state.device_id}: {state.error}")
                self._discard_stale_frames_locked(state, keep_frames=num_frames)
                if state.queued_frames >= num_frames:
                    break
                remaining_timeout = deadline - time.monotonic()
                if remaining_timeout <= 0:
                    status_text = state.statuses[-1] if state.statuses else "timeout waiting for audio callback"
                    raise TimeoutError(f"device {state.device_id}: {status_text}")
                state.condition.wait(timeout=remaining_timeout)

            pulled_chunks = []
            remaining = num_frames
            while remaining > 0:
                chunk = state.chunks[0]
                if chunk.shape[0] <= remaining:
                    pulled_chunks.append(state.chunks.popleft())
                    state.queued_frames -= chunk.shape[0]
                    remaining -= chunk.shape[0]
                else:
                    pulled_chunks.append(chunk[:remaining].copy())
                    state.chunks[0] = chunk[remaining:]
                    state.queued_frames -= remaining
                    remaining = 0
            return np.concatenate(pulled_chunks, axis=0)

    @staticmethod
    def _make_input_callback(state: SoundDeviceStreamState):
        def callback(indata, _frames, _time_info, status):
            chunk = np.asarray(indata, dtype=np.float32).copy()
            status_text = str(status) if status else None
            with state.condition:
                state.chunks.append(chunk)
                state.queued_frames += chunk.shape[0]
                if status_text:
                    state.statuses.append(status_text)
                state.condition.notify_all()

        return callback

    @staticmethod
    def _discard_stale_frames_locked(state: SoundDeviceStreamState, keep_frames: int) -> None:
        stale_frames = max(0, state.queued_frames - keep_frames)
        while stale_frames > 0 and state.chunks:
            chunk = state.chunks[0]
            if chunk.shape[0] <= stale_frames:
                state.chunks.popleft()
                state.queued_frames -= chunk.shape[0]
                stale_frames -= chunk.shape[0]
            else:
                state.chunks[0] = chunk[stale_frames:]
                state.queued_frames -= stale_frames
                stale_frames = 0

def get_map_bounds() -> tuple[float, float, float, float]:
    half_map = MAP_SIZE_M / 2.0
    center_x, center_y = ROOM_CENTER_XY
    return center_x - half_map, center_x + half_map, center_y - half_map, center_y + half_map


def get_map_axes(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = get_map_bounds()
    x_coords = np.linspace(x_min, x_max, width, dtype=np.float32)
    y_coords = np.linspace(y_min, y_max, height, dtype=np.float32)
    return x_coords, y_coords


class RealSoundObservationSource:
    def __init__(
        self,
        explicit_device_ids: list[int] | None = None,
        observation_height: int = OBSERVATION_HEIGHT,
        observation_width: int = OBSERVATION_WIDTH,
    ):
        device_ids = explicit_device_ids or parse_device_ids_env("SOUNDREAL_DEVICE_IDS")
        self.devices = select_tamago_devices(device_ids)
        self.audio_fps = AUDIO_FPS
        self.sound_camera = SoundCamera(
            target=DummyTarget(),
            config=SoundConfig(
                mic_array_num=len(self.devices),
                mics_per_array=DEFAULT_MIC_CHANNELS,
                fs=AUDIO_SAMPLE_RATE,
                nfft=SPECTROGRAM_NFFT,
                num_peaks=SPECTROGRAM_NUM_PEAKS,
                observation_height=observation_height,
                observation_width=observation_width,
                use_spectrogram=True,
                use_soundmap=True,
                use_gaussian_filter=False,
                processing_time=AUDIO_WINDOW_SECONDS,
                azimuth_offset_deg=AZIMUTH_OFFSET_DEG,
                doa_distance_floor_m=DOA_DISTANCE_FLOOR_M,
                doa_distance_decay_exponent=DOA_DISTANCE_DECAY_EXPONENT,
                combined_map_power=COMBINED_MAP_POWER,
                combined_map_gaussian_sigma=1.0,
                peak_selection_mode="argmax",
            ),
        )
        self.sound_camera.mic_positions = build_rectangular_mic_positions(self.devices)
        self.map_x_coords, self.map_y_coords = get_map_axes(observation_width, observation_height)
        self.array_relative_positions = []
        for mic_center in self.sound_camera.mic_positions:
            mic_array_abs = self.sound_camera._generate_circular_array(
                mic_center,
                self.sound_camera.config.mics_per_array,
                self.sound_camera.config.mic_radius,
            ).T.astype(np.float32, copy=False)
            self.array_relative_positions.append(
                mic_array_abs - np.mean(mic_array_abs, axis=0, keepdims=True)
            )
        self.processing_window_s = float(self.sound_camera.config.processing_time)
        self.processing_period_s = 1.0 / self.audio_fps
        self.capture_chunk_duration_s = min(self.processing_period_s, SOUNDDEVICE_CAPTURE_CHUNK_SECONDS)
        self.processing_frames = max(1, int(round(self.processing_window_s * AUDIO_SAMPLE_RATE)))
        self.capture_chunk_frames = max(1, int(round(self.capture_chunk_duration_s * AUDIO_SAMPLE_RATE)))
        self.buffered_frames = 0
        self.buffer_write_pos = 0
        self.capture_sequence = 0
        self._audio_buffers = [
            np.zeros((self.processing_frames, DEFAULT_MIC_CHANNELS), dtype=np.float32)
            for _ in self.devices
        ]
        self._audio_lock = threading.Lock()
        self._lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._update_thread: Optional[threading.Thread] = None
        self._capture_session: Optional[ContinuousSoundDeviceCapture] = None
        self.start_monotonic: Optional[float] = None
        self.latest_capture_monotonic: Optional[float] = None
        self.latest_process_monotonic: Optional[float] = None
        self._recent_capture_times: deque[float] = deque(maxlen=64)
        self._recent_process_times: deque[float] = deque(maxlen=32)
        self.last_compute_duration_s = 0.0
        self.latest_update_monotonic: Optional[float] = None
        self.process_count = 0
        self.update_count = 0
        self.dropped_capture_count = 0
        self.late_cycle_count = 0
        self.last_capture_error: Optional[str] = None
        self.last_process_error: Optional[str] = None
        self.cached_images = {
            "sound0": np.zeros((observation_height, observation_width, 3), dtype=np.uint8),
            "sound1": np.zeros((observation_height, observation_width, 3), dtype=np.uint8),
            "spec": np.zeros((observation_height, observation_width, 3), dtype=np.uint8),
        }

    def start(self) -> None:
        if self._update_thread and self._update_thread.is_alive():
            return
        self._stop_event.clear()
        self._ready_event.clear()
        if torch.cuda.is_available():
            torch.empty(1, device="cuda")
            torch.cuda.synchronize()
        with self._audio_lock:
            self.buffered_frames = 0
            self.buffer_write_pos = 0
            self.capture_sequence = 0
            self._audio_buffers = [
                np.zeros((self.processing_frames, DEFAULT_MIC_CHANNELS), dtype=np.float32)
                for _ in self.devices
            ]
        self.last_compute_duration_s = 0.0
        self.latest_capture_monotonic = None
        self.latest_process_monotonic = None
        self._recent_capture_times.clear()
        self._recent_process_times.clear()
        self.latest_update_monotonic = None
        self.process_count = 0
        self.update_count = 0
        self.dropped_capture_count = 0
        self.late_cycle_count = 0
        self.last_capture_error = None
        self.last_process_error = None
        self.sound_camera.reset_nmf_state()
        for key, value in self.cached_images.items():
            self.cached_images[key] = np.zeros_like(value)
        self.start_monotonic = time.perf_counter()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._capture_thread.start()
        self._update_thread.start()

    @staticmethod
    def _compute_recent_rate(timestamps: deque[float]) -> Optional[float]:
        if len(timestamps) < 2:
            return None
        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0.0:
            return None
        return (len(timestamps) - 1) / elapsed

    def _get_last_error_locked(self) -> Optional[str]:
        return self.last_process_error or self.last_capture_error

    def stop(self) -> None:
        self._stop_event.set()
        self._stop_capture_session()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)
        self._capture_thread = None
        self._update_thread = None

    def wait_until_ready(self, timeout_s: float = 2.0) -> bool:
        return self._ready_event.wait(timeout_s)

    def get_latest_images(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {key: value.copy() for key, value in self.cached_images.items()}

    def get_latest_debug_snapshot(self) -> tuple[dict[str, np.ndarray], dict[str, float | int | str | None]]:
        with self._audio_lock:
            buffered_duration_s = self.buffered_frames / AUDIO_SAMPLE_RATE
        with self._lock:
            latest_age_s = None
            if self.latest_process_monotonic is not None:
                latest_age_s = max(0.0, time.perf_counter() - self.latest_process_monotonic)
            effective_fps = self._compute_recent_rate(self._recent_process_times)
            images = {key: value.copy() for key, value in self.cached_images.items()}
            status = {
                "process_count": self.process_count,
                "update_count": self.update_count,
                "dropped_capture_count": self.dropped_capture_count,
                "dropped_audio_s": self.dropped_capture_count * self.capture_chunk_duration_s,
                "late_cycle_count": self.late_cycle_count,
                "last_compute_duration_s": self.last_compute_duration_s,
                "latest_age_s": latest_age_s,
                "effective_fps": effective_fps,
                "buffered_duration_s": buffered_duration_s,
                "last_error": self._get_last_error_locked(),
            }
        return images, status

    def get_debug_status(self) -> dict[str, float | int | str | None]:
        with self._audio_lock:
            buffered_duration_s = self.buffered_frames / AUDIO_SAMPLE_RATE
        with self._lock:
            latest_age_s = None
            if self.latest_process_monotonic is not None:
                latest_age_s = max(0.0, time.perf_counter() - self.latest_process_monotonic)
            effective_fps = self._compute_recent_rate(self._recent_process_times)
            return {
                "process_count": self.process_count,
                "update_count": self.update_count,
                "dropped_capture_count": self.dropped_capture_count,
                "dropped_audio_s": self.dropped_capture_count * self.capture_chunk_duration_s,
                "late_cycle_count": self.late_cycle_count,
                "last_compute_duration_s": self.last_compute_duration_s,
                "latest_age_s": latest_age_s,
                "effective_fps": effective_fps,
                "buffered_duration_s": buffered_duration_s,
                "last_error": self._get_last_error_locked(),
            }

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                capture_session = self._ensure_capture_session()
                chunks = capture_session.read_chunk(self.capture_chunk_frames)
                self._append_audio_chunks(chunks)
                captured_now = time.perf_counter()
                with self._lock:
                    self.latest_capture_monotonic = captured_now
                    self._recent_capture_times.append(captured_now)
                    self.last_capture_error = None
            except Exception as exc:
                self._stop_capture_session()
                with self._lock:
                    self.last_capture_error = str(exc)
                if not self._stop_event.is_set():
                    print(f"[soundreal] Failed to capture audio chunk: {exc}")
                    self._stop_event.wait(0.1)

    def _update_loop(self) -> None:
        next_deadline = time.perf_counter()
        last_processed_capture_sequence = -1
        while not self._stop_event.is_set():
            remaining = next_deadline - time.perf_counter()
            if remaining > 0:
                self._stop_event.wait(remaining)
                if self._stop_event.is_set():
                    break
            start = time.perf_counter()
            try:
                snapshot_info = self._snapshot_audio_buffers()
                if snapshot_info is None:
                    next_deadline = time.perf_counter() + self.processing_period_s
                    continue
                snapshot, capture_sequence = snapshot_info
                if capture_sequence == last_processed_capture_sequence:
                    next_deadline = max(next_deadline + self.processing_period_s, time.perf_counter())
                    continue
                sound0, sound1, spec = self._build_images_from_snapshot(snapshot)
                elapsed = time.perf_counter() - start
                published_now = time.perf_counter()
                dropped_captures = 0
                if last_processed_capture_sequence >= 0:
                    dropped_captures = max(0, capture_sequence - last_processed_capture_sequence - 1)
                with self._lock:
                    images_changed = not (
                        np.array_equal(self.cached_images["sound0"], sound0)
                        and np.array_equal(self.cached_images["sound1"], sound1)
                        and np.array_equal(self.cached_images["spec"], spec)
                    )
                    if images_changed:
                        self.cached_images["sound0"] = sound0
                        self.cached_images["sound1"] = sound1
                        self.cached_images["spec"] = spec
                        self.latest_update_monotonic = published_now
                        self.update_count += 1
                    self.latest_process_monotonic = published_now
                    self._recent_process_times.append(published_now)
                    self.process_count += 1
                    self.dropped_capture_count += dropped_captures
                    self.last_compute_duration_s = elapsed
                    self.last_process_error = None
                    if elapsed > self.processing_period_s:
                        self.late_cycle_count += 1
                last_processed_capture_sequence = capture_sequence
                if images_changed:
                    self._ready_event.set()
            except Exception as exc:
                with self._lock:
                    self.last_process_error = str(exc)
                if not self._stop_event.is_set():
                    print(f"[soundreal] Failed to update sound observations: {exc}")
            next_deadline = max(next_deadline + self.processing_period_s, time.perf_counter())

    def _ensure_capture_session(self) -> ContinuousSoundDeviceCapture:
        with self._capture_lock:
            if self._capture_session is None:
                capture_session = ContinuousSoundDeviceCapture(
                    devices=self.devices,
                    samplerate=AUDIO_SAMPLE_RATE,
                    channels=DEFAULT_MIC_CHANNELS,
                    chunk_frames=self.capture_chunk_frames,
                )
                capture_session.start()
                self._capture_session = capture_session
            return self._capture_session

    def _stop_capture_session(self) -> None:
        with self._capture_lock:
            if self._capture_session is None:
                return
            self._capture_session.stop()
            self._capture_session = None

    def _append_audio_chunks(self, chunks: list[np.ndarray]) -> None:
        if len(chunks) != len(self._audio_buffers):
            raise RuntimeError(
                f"Expected {len(self._audio_buffers)} audio chunks, but received {len(chunks)}."
            )

        with self._audio_lock:
            chunk_frames = None
            for index, chunk in enumerate(chunks):
                if chunk.ndim != 2 or chunk.shape[1] != DEFAULT_MIC_CHANNELS:
                    raise RuntimeError(
                        f"Unexpected chunk shape for array {index}: {chunk.shape}, expected (_, {DEFAULT_MIC_CHANNELS})."
                    )
                if chunk_frames is None:
                    chunk_frames = chunk.shape[0]
                elif chunk.shape[0] != chunk_frames:
                    raise RuntimeError(
                        "Audio chunks must have the same frame count for all arrays: "
                        f"expected {chunk_frames}, got {chunk.shape[0]} for array {index}."
                    )
                if chunk_frames >= self.processing_frames:
                    self._audio_buffers[index][:] = chunk[-self.processing_frames :]
                else:
                    buffer = self._audio_buffers[index]
                    write_end = self.buffer_write_pos + chunk_frames
                    if write_end <= self.processing_frames:
                        buffer[self.buffer_write_pos : write_end] = chunk
                    else:
                        first = self.processing_frames - self.buffer_write_pos
                        buffer[self.buffer_write_pos :] = chunk[:first]
                        buffer[: write_end - self.processing_frames] = chunk[first:]
            if chunk_frames is None:
                return
            if chunk_frames >= self.processing_frames:
                self.buffer_write_pos = 0
            else:
                self.buffer_write_pos = (self.buffer_write_pos + chunk_frames) % self.processing_frames
            self.buffered_frames = min(self.processing_frames, self.buffered_frames + chunk_frames)
            self.capture_sequence += 1

    def _snapshot_audio_buffers(self) -> tuple[list[np.ndarray], int] | None:
        with self._audio_lock:
            if self.buffered_frames <= 0:
                return None
            if self.buffered_frames < self.processing_frames:
                snapshot = [buffer[: self.buffered_frames].copy() for buffer in self._audio_buffers]
            elif self.buffer_write_pos == 0:
                snapshot = [buffer.copy() for buffer in self._audio_buffers]
            else:
                snapshot = [
                    np.concatenate((buffer[self.buffer_write_pos :], buffer[: self.buffer_write_pos]), axis=0)
                    for buffer in self._audio_buffers
                ]
            return snapshot, self.capture_sequence

    def _build_images_from_snapshot(self, snapshot: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mic_signals_list = [audio.T for audio in snapshot]
        return self.sound_camera.build_sound_observation_from_mic_signals(
            mic_signals_list,
            x_coords=self.map_x_coords,
            y_coords=self.map_y_coords,
            azimuth_offset_deg=AZIMUTH_OFFSET_DEG,
            distance_floor_m=DOA_DISTANCE_FLOOR_M,
            distance_decay_exponent=DOA_DISTANCE_DECAY_EXPONENT,
            combined_map_power=COMBINED_MAP_POWER,
            combined_map_gaussian_sigma=1.0,
            peak_selection_mode="argmax",
            array_relative_positions=self.array_relative_positions,
            create_spectrogram=True,
        )


def create_sound_debug_panel(
    images: dict[str, np.ndarray],
    status: dict[str, float | int | str | None],
    condition: SoundEpisodeCondition | None = None,
) -> np.ndarray:
    tile_keys = ("sound0", "sound1", "spec")
    tiles = []
    for key in tile_keys:
        image = images.get(key)
        if image is None:
            image = np.zeros((OBSERVATION_HEIGHT, OBSERVATION_WIDTH, 3), dtype=np.uint8)
        tile = np.ascontiguousarray(image.copy())
        cv2.putText(
            tile,
            key,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    panel = np.concatenate(tiles, axis=1)
    info_lines = [
        ""
        if condition is None
        else f"speaker={condition.speaker} sound={condition.sound_label} ({condition.sound_path.name})"
    ]
    effective_fps = status.get("effective_fps")
    latest_age_s = status.get("latest_age_s")
    buffered_duration_s = float(status.get("buffered_duration_s") or 0.0)
    dropped_audio_s = float(status.get("dropped_audio_s") or 0.0)
    info_lines.append(
        "proc={proc} drop={drop:.3f}s fps={fps} compute={compute:.1f}ms age={age} buffer={buffer:.3f}/{window:.1f}s".format(
            proc=status.get("process_count"),
            drop=dropped_audio_s,
            fps="n/a" if effective_fps is None else f"{effective_fps:.2f}",
            compute=1000.0 * float(status.get("last_compute_duration_s") or 0.0),
            age="n/a" if latest_age_s is None else f"{latest_age_s:.3f}s",
            buffer=buffered_duration_s,
            window=AUDIO_WINDOW_SECONDS,
        )
    )
    overlay_height = 28 * len(info_lines) + 12
    overlay = np.zeros((overlay_height, panel.shape[1], 3), dtype=np.uint8)
    for idx, line in enumerate(info_lines):
        cv2.putText(
            overlay,
            str(line),
            (8, 26 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return np.concatenate([overlay, panel], axis=0)


def make_debug_output_path(output_path: str | None) -> Path:
    if output_path:
        path = Path(output_path)
        return path if path.suffix.lower() == ".mp4" else path.with_suffix(".mp4")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("debug") / f"soundreal_debug_{timestamp}.mp4"


class DebugVideoRecorder:
    def __init__(self, output_path: Path, fps: int):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="soundreal_debug_frames_")
        self.frames_dir = Path(self._tmp_dir.name)
        self.frame_index = 0
        self.closed = False

    def write(self, frame: np.ndarray) -> None:
        if self.closed:
            raise RuntimeError("DebugVideoRecorder is already closed.")
        frame_path = self.frames_dir / f"frame-{self.frame_index:06d}.png"
        ok = cv2.imwrite(str(frame_path), frame)
        if not ok:
            raise RuntimeError(f"Failed to write debug frame: {frame_path}")
        self.frame_index += 1

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.frame_index == 0:
                return
            command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(self.fps),
                "-i",
                str(self.frames_dir / "frame-%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.output_path),
            ]
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg encoding failed for {self.output_path}:\n{result.stderr.strip()}"
                )
        finally:
            self._tmp_dir.cleanup()


def parse_cli_device_ids(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def run_sound_debug_session(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    sampled_condition = sample_sound_episode_condition(rng)
    sound_index = sampled_condition.sound_index if args.sound_index is None else int(args.sound_index)
    speaker = sampled_condition.speaker if args.speaker is None else str(args.speaker)
    condition = SoundEpisodeCondition(
        sound_index=sound_index,
        sound_label=SOUND_LABELS[sound_index],
        sound_path=SOUND_FILES[sound_index],
        speaker=speaker,
    )

    sound_source = RealSoundObservationSource(
        explicit_device_ids=parse_device_ids_env("SOUNDREAL_DEVICE_IDS")
        if args.input_device_ids is None
        else parse_cli_device_ids(args.input_device_ids),
        observation_height=args.height,
        observation_width=args.width,
    )
    audio_player = None if args.no_audio else LoopingStereoPlayer(output_device=args.output_device)
    display_enabled = not args.no_display
    writer = None
    output_path = None
    window_name = "soundreal_debug"
    frame_interval_s = 1.0 / max(1, args.video_fps)
    last_frame_time = 0.0
    last_logged_process_count = -1
    last_logged_error = None

    try:
        sound_source.start()
        if not sound_source.wait_until_ready(timeout_s=args.ready_timeout_s):
            print("[soundreal] Audio capture backend is still warming up. Initial debug frames may be zeros.")

        if audio_player is not None:
            audio_player.start(condition)
            print(
                "[soundreal] Playback started: "
                f"speaker={condition.speaker}, sound={condition.sound_label}, file={condition.sound_path}"
            )
        else:
            print("[soundreal] Running without playback audio.")

        print("[soundreal] Press 'q' in the preview window or Ctrl+C to stop.")
        start_time = time.perf_counter()

        while True:
            now = time.perf_counter()
            if now - start_time >= args.duration_s:
                break

            images, status = sound_source.get_latest_debug_snapshot()
            panel = create_sound_debug_panel(images, status, condition)

            if writer is None and (args.save_mp4 or not display_enabled):
                output_path = make_debug_output_path(args.output_path)
                writer = DebugVideoRecorder(output_path, fps=args.video_fps)
                print(f"[soundreal] Saving debug video to {output_path}")

            if display_enabled:
                try:
                    cv2.imshow(window_name, panel)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                except cv2.error as exc:
                    print(f"[soundreal] Preview disabled because OpenCV window creation failed: {exc}")
                    display_enabled = False
                    if writer is None:
                        output_path = make_debug_output_path(args.output_path)
                        writer = DebugVideoRecorder(output_path, fps=args.video_fps)
                        print(f"[soundreal] Falling back to debug video save: {output_path}")

            if writer is not None and now - last_frame_time >= frame_interval_s:
                writer.write(panel)
                last_frame_time = now

            current_process_count = int(status.get("process_count") or 0)
            current_error = status.get("last_error")

            if current_error and current_error != last_logged_error:
                print(f"[soundreal] error={current_error}")
                last_logged_error = current_error

            if current_process_count != last_logged_process_count:
                print(
                    "[soundreal] proc#{proc} drop={drop:.3f}s fps={fps} compute={compute:.1f}ms age={age} buffer={buffer:.3f}s".format(
                        proc=current_process_count,
                        drop=float(status.get("dropped_audio_s") or 0.0),
                        fps="n/a"
                        if status.get("effective_fps") is None
                        else f"{float(status['effective_fps']):.2f}",
                        compute=1000.0 * float(status.get("last_compute_duration_s") or 0.0),
                        age="n/a"
                        if status.get("latest_age_s") is None
                        else f"{float(status['latest_age_s']):.3f}s",
                        buffer=float(status.get("buffered_duration_s") or 0.0),
                    )
                )
                last_logged_process_count = current_process_count

            time.sleep(max(0.0, min(0.02, frame_interval_s / 4.0)))

    except KeyboardInterrupt:
        print("\n[soundreal] Debug session interrupted by user.")
    finally:
        if writer is not None:
            writer.close()
            if output_path is not None:
                print(f"[soundreal] Debug video saved: {output_path}")
        if display_enabled:
            cv2.destroyAllWindows()
        if audio_player is not None:
            audio_player.stop()
        sound_source.stop()


def print_audio_device_list() -> None:
    print("Input devices:")
    for device_id, device in query_input_devices():
        print(
            f"  [{device_id}] {device['name']} | "
            f"input_channels={int(device['max_input_channels'])} | "
            f"default_sr={device['default_samplerate']}"
        )
    print("Output devices:")
    for device_id, device in enumerate(sd.query_devices()):
        if int(device["max_output_channels"]) <= 0:
            continue
        print(
            f"  [{device_id}] {device['name']} | "
            f"output_channels={int(device['max_output_channels'])} | "
            f"default_sr={device['default_samplerate']}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="soundReal audio pipeline debug utility")
    parser.add_argument("--duration_s", type=float, default=15.0, help="debug session duration in seconds")
    parser.add_argument("--video_fps", type=int, default=AUDIO_FPS, help="preview/save FPS")
    parser.add_argument("--height", type=int, default=OBSERVATION_HEIGHT, help="observation height")
    parser.add_argument("--width", type=int, default=OBSERVATION_WIDTH, help="observation width")
    parser.add_argument("--save_mp4", action="store_true", help="save the debug panel as mp4")
    parser.add_argument("--output_path", type=str, default=None, help="output mp4 path")
    parser.add_argument("--no_display", action="store_true", help="disable real-time preview window")
    parser.add_argument("--no_audio", action="store_true", help="disable speaker playback")
    parser.add_argument("--output_device", type=int, default=None, help="sounddevice output device id")
    parser.add_argument(
        "--input_device_ids",
        type=str,
        default=None,
        help="comma-separated TAMAGO input device ids; default is auto-detect",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        choices=("left", "right"),
        default=None,
        help="speaker to play through; default is random",
    )
    parser.add_argument(
        "--sound_index",
        type=int,
        choices=tuple(sorted(SOUND_FILES.keys())),
        default=None,
        help="sound index to play; default is random",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed for debug condition selection")
    parser.add_argument(
        "--ready_timeout_s",
        type=float,
        default=5.0,
        help="seconds to wait for the first sound observation",
    )
    parser.add_argument("--list_devices", action="store_true", help="print available audio devices and exit")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.list_devices:
        print_audio_device_list()
    else:
        run_sound_debug_session(args)
