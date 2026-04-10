from __future__ import annotations

import argparse
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyroomacoustics as pra
import sounddevice as sd
from pydub import AudioSegment
from scipy.ndimage import gaussian_filter
from scipy.signal import stft

from env.tasks.sound_camera import SoundCamera, SoundConfig


SOUNDREAL_TASK_NAME = "soundReal-m4-f10-s2-p0"

OBSERVATION_HEIGHT = 224
OBSERVATION_WIDTH = 224
CAMERA_FPS = 30
AUDIO_FPS = 10
AUDIO_SAMPLE_RATE = 16000
AUDIO_WINDOW_SECONDS = 1.0
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

TAMAGO_DEVICE_IDS = [0, 1, 2, 3]
AZIMUTH_OFFSET_DEG = 0.0
RECTANGLE_LONG_SIDE_M = 1.2
RECTANGLE_SHORT_SIDE_M = 0.6
ARRAY_HEIGHT_M = 0.1
ROOM_CENTER_XY = np.array([5.0, 5.0], dtype=np.float32)
RECTANGLE_HW_ORDER_CLOCKWISE = [2, 0, 1, 3]
MAP_SIZE_M = 1.4
DOA_DISTANCE_FLOOR_M = 0.0
DOA_DISTANCE_DECAY_EXPONENT = 0.0
COMBINED_MAP_POWER = 4.0

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


def get_map_bounds() -> tuple[float, float, float, float]:
    half_map = MAP_SIZE_M / 2.0
    center_x, center_y = ROOM_CENTER_XY
    return center_x - half_map, center_x + half_map, center_y - half_map, center_y + half_map


def get_map_axes(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = get_map_bounds()
    x_coords = np.linspace(x_min, x_max, width, dtype=np.float32)
    y_coords = np.linspace(y_min, y_max, height, dtype=np.float32)
    return x_coords, y_coords


def apply_azimuth_offset_rad(angle_rad: np.ndarray | float, offset_deg: float = AZIMUTH_OFFSET_DEG) -> np.ndarray:
    return np.asarray(angle_rad) + np.deg2rad(offset_deg)


def estimate_music(sound_camera: SoundCamera, mic_signals: np.ndarray, array_index: int) -> pra.doa.MUSIC:
    z = np.asarray(
        [
            stft(
                mic_signals[ch],
                fs=sound_camera.config.fs,
                nperseg=sound_camera.config.nfft,
                noverlap=sound_camera.config.nfft // 2,
            )[2]
            for ch in range(mic_signals.shape[0])
        ]
    )
    mic_positions = sound_camera._generate_circular_array(
        sound_camera.mic_positions[array_index],
        sound_camera.config.mics_per_array,
        sound_camera.config.mic_radius,
    )
    doa = pra.doa.MUSIC(
        mic_positions,
        fs=sound_camera.config.fs,
        nfft=sound_camera.config.nfft,
        c=sound_camera.config.sound_speed,
        num_src=sound_camera.config.music_num_src,
    )
    doa.locate_sources(z)
    return doa


def generate_soundmap_from_doa_fast(sound_camera: SoundCamera, doa: pra.doa.MUSIC, mic_center: np.ndarray) -> np.ndarray:
    spec = np.log10(np.mean(doa.Pssl, axis=1))
    spec_sum = np.sum(spec)
    if spec_sum != 0:
        spec = spec / spec_sum

    x_coords, y_coords = get_map_axes(sound_camera.config.observation_width, sound_camera.config.observation_height)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="xy")
    theta = apply_azimuth_offset_rad(np.arctan2(y_grid - mic_center[1], x_grid - mic_center[0]))
    angle_idx = (theta / (2 * np.pi / len(spec))).astype(np.int32) % len(spec)
    distance = np.hypot(y_grid - mic_center[1], x_grid - mic_center[0])
    distance_weight = 1.0 / np.power(distance + DOA_DISTANCE_FLOOR_M, DOA_DISTANCE_DECAY_EXPONENT)
    return (spec[angle_idx] * distance_weight).astype(np.float32)


def build_combined_sound_map(sound_maps: list[np.ndarray], power: float = COMBINED_MAP_POWER) -> np.ndarray:
    return gaussian_filter(
        np.sum([np.power(sound_map, power, dtype=np.float32) for sound_map in sound_maps], axis=0),
        sigma=1.0,
    )


def compute_top_peaks(sound_camera: SoundCamera, combined_sound_map: np.ndarray) -> list[tuple[float, float]]:
    topk = sound_camera.config.num_peaks
    indices = np.argpartition(combined_sound_map.ravel(), -topk)[-topk:]
    indices = indices[np.argsort(combined_sound_map.ravel()[indices])[::-1]]
    x_coords, y_coords = get_map_axes(sound_camera.config.observation_width, sound_camera.config.observation_height)
    return [
        (float(x_coords[col]), float(y_coords[row]))
        for row, col in (np.unravel_index(int(flat_index), combined_sound_map.shape) for flat_index in indices)
    ]


def reconstruct_spotformed_wavs(
    sound_camera: SoundCamera,
    sound_maps: list[np.ndarray],
    mic_signals_list: list[np.ndarray],
    top_peaks: list[tuple[float, float]],
) -> list[np.ndarray]:
    if not top_peaks:
        return []

    reconstructed = []
    for peak_x, peak_y in top_peaks:
        beamformed_signals = []
        for array_index, mic_signals in enumerate(mic_signals_list):
            mic_center = sound_camera.mic_positions[array_index]
            mic_array_abs = sound_camera._generate_circular_array(
                mic_center,
                sound_camera.config.mics_per_array,
                sound_camera.config.mic_radius,
            ).T
            theta_deg = sound_camera._pixel_to_azimuth(peak_x, peak_y, mic_center)
            beamformed_signals.append(
                sound_camera._ds_beamform(
                    mic_signals,
                    mic_array_abs - np.mean(mic_array_abs, axis=0),
                    theta_deg,
                )
            )
        final_wav, _, _ = sound_camera._perform_spotforming_nmf(
            beamformed_signals=beamformed_signals,
            fs=sound_camera.config.fs,
            n_components=sound_camera.config.nmf_components,
            threshold=sound_camera.config.nmf_threshold,
            nfft=sound_camera.config.nfft,
            noverlap=sound_camera.config.nfft // 2,
        )
        reconstructed.append(final_wav[: int(sound_camera.config.fs * sound_camera.config.processing_time)])
    return reconstructed


def create_spectrogram_image_from_audio(sound_camera: SoundCamera, audio: np.ndarray) -> np.ndarray:
    _, _, zxx = stft(
        audio,
        fs=sound_camera.config.fs,
        nperseg=sound_camera.config.nfft,
        noverlap=sound_camera.config.nfft // 2,
    )
    image = sound_camera._convert_spectrogram_to_image(10 * np.log10(np.abs(zxx) ** 2 + 1e-10))
    return sound_camera._pad_to_3ch(image)


class MultiDeviceAudioRingBuffer:
    def __init__(
        self,
        devices: list[tuple[int, dict]],
        samplerate: int = AUDIO_SAMPLE_RATE,
        channels: int = DEFAULT_MIC_CHANNELS,
        window_seconds: float = AUDIO_WINDOW_SECONDS,
    ):
        self.devices = devices
        self.samplerate = samplerate
        self.channels = channels
        self.window_frames = int(round(window_seconds * samplerate))
        self.streams: list[sd.InputStream] = []
        self._lock = threading.Lock()
        self._buffers = {
            device_id: np.zeros((self.window_frames, channels), dtype=np.float32)
            for device_id, _ in devices
        }
        self._write_positions = {device_id: 0 for device_id, _ in devices}
        self._frame_counts = {device_id: 0 for device_id, _ in devices}
        self._running = False

    def _make_callback(self, device_id: int):
        def callback(indata, frames, _time_info, status):
            if status:
                print(f"[soundreal] InputStream status on device {device_id}: {status}")
            data = np.asarray(indata, dtype=np.float32)
            if data.ndim == 1:
                data = data[:, None]
            if data.shape[1] < self.channels:
                padded = np.zeros((data.shape[0], self.channels), dtype=np.float32)
                padded[:, : data.shape[1]] = data
                data = padded
            elif data.shape[1] > self.channels:
                data = data[:, : self.channels]

            if data.shape[0] >= self.window_frames:
                data = data[-self.window_frames :]

            frames_to_write = data.shape[0]
            with self._lock:
                buffer = self._buffers[device_id]
                write_pos = self._write_positions[device_id]
                first = min(frames_to_write, self.window_frames - write_pos)
                buffer[write_pos : write_pos + first] = data[:first]
                remaining = frames_to_write - first
                if remaining > 0:
                    buffer[:remaining] = data[first:]
                self._write_positions[device_id] = (write_pos + frames_to_write) % self.window_frames
                self._frame_counts[device_id] = min(
                    self.window_frames,
                    self._frame_counts[device_id] + frames_to_write,
                )

        return callback

    def start(self) -> None:
        if self._running:
            return
        for device_id, device_info in self.devices:
            available_channels = int(device_info["max_input_channels"])
            if available_channels < self.channels:
                raise RuntimeError(
                    f"Device {device_id} has only {available_channels} input channels, but {self.channels} are required."
                )
            stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype="float32",
                device=device_id,
                callback=self._make_callback(device_id),
                blocksize=0,
            )
            stream.start()
            self.streams.append(stream)
        self._running = True

    def stop(self) -> None:
        for stream in self.streams:
            try:
                stream.stop()
            finally:
                stream.close()
        self.streams = []
        self._running = False

    def ready(self) -> bool:
        with self._lock:
            return all(frame_count >= self.window_frames for frame_count in self._frame_counts.values())

    def snapshot(self) -> Optional[list[np.ndarray]]:
        with self._lock:
            if not all(frame_count >= self.window_frames for frame_count in self._frame_counts.values()):
                return None

            snapshots = []
            for device_id, _ in self.devices:
                buffer = self._buffers[device_id]
                write_pos = self._write_positions[device_id]
                ordered = np.concatenate([buffer[write_pos:], buffer[:write_pos]], axis=0)
                snapshots.append(ordered.copy())
            return snapshots


class RealSoundObservationSource:
    def __init__(
        self,
        explicit_device_ids: list[int] | None = None,
        observation_height: int = OBSERVATION_HEIGHT,
        observation_width: int = OBSERVATION_WIDTH,
        audio_fps: int = AUDIO_FPS,
    ):
        device_ids = explicit_device_ids or parse_device_ids_env("SOUNDREAL_DEVICE_IDS")
        self.devices = select_tamago_devices(device_ids)
        self.audio_fps = audio_fps
        self.sound_camera = SoundCamera(
            target=DummyTarget(),
            config=SoundConfig(
                mic_array_num=len(self.devices),
                mics_per_array=DEFAULT_MIC_CHANNELS,
                fs=AUDIO_SAMPLE_RATE,
                nfft=512,
                num_peaks=1,
                observation_height=observation_height,
                observation_width=observation_width,
                use_spectrogram=True,
                use_soundmap=True,
                use_gaussian_filter=False,
                processing_time=AUDIO_WINDOW_SECONDS,
            ),
        )
        self.sound_camera.mic_positions = build_rectangular_mic_positions(self.devices)
        self.capture = MultiDeviceAudioRingBuffer(
            devices=self.devices,
            samplerate=AUDIO_SAMPLE_RATE,
            channels=DEFAULT_MIC_CHANNELS,
            window_seconds=AUDIO_WINDOW_SECONDS,
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._update_thread: Optional[threading.Thread] = None
        self.start_monotonic: Optional[float] = None
        self.last_compute_duration_s = 0.0
        self.latest_update_monotonic: Optional[float] = None
        self.update_count = 0
        self.late_cycle_count = 0
        self.last_error: Optional[str] = None
        self.cached_images = {
            "sound0": np.zeros((observation_height, observation_width, 3), dtype=np.uint8),
            "sound1": np.zeros((observation_height, observation_width, 3), dtype=np.uint8),
            "spec": np.zeros((observation_height, observation_width, 3), dtype=np.uint8),
        }

    def start(self) -> None:
        if self._update_thread and self._update_thread.is_alive():
            return
        self.capture.start()
        self._stop_event.clear()
        self.start_monotonic = time.perf_counter()
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)
        self._update_thread = None
        self.capture.stop()

    def wait_until_ready(self, timeout_s: float = 2.0) -> bool:
        return self._ready_event.wait(timeout_s)

    def get_latest_images(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {key: value.copy() for key, value in self.cached_images.items()}

    def get_debug_status(self) -> dict[str, float | int | str | None]:
        with self._lock:
            latest_age_s = None
            if self.latest_update_monotonic is not None:
                latest_age_s = max(0.0, time.perf_counter() - self.latest_update_monotonic)
            effective_fps = None
            if self.update_count > 0 and self.start_monotonic is not None and self.latest_update_monotonic is not None:
                runtime_s = max(1e-6, self.latest_update_monotonic - self.start_monotonic)
                effective_fps = self.update_count / runtime_s
            return {
                "update_count": self.update_count,
                "late_cycle_count": self.late_cycle_count,
                "last_compute_duration_s": self.last_compute_duration_s,
                "latest_age_s": latest_age_s,
                "effective_fps": effective_fps,
                "last_error": self.last_error,
            }

    def _update_loop(self) -> None:
        period = 1.0 / self.audio_fps
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            snapshot = self.capture.snapshot()
            if snapshot is not None:
                try:
                    self._update_from_snapshot(snapshot)
                    elapsed = time.perf_counter() - loop_start
                    with self._lock:
                        self.last_compute_duration_s = elapsed
                        self.latest_update_monotonic = time.perf_counter()
                        self.update_count += 1
                        if elapsed > period:
                            self.late_cycle_count += 1
                        self.last_error = None
                    self._ready_event.set()
                except Exception as exc:
                    with self._lock:
                        self.last_error = str(exc)
                    print(f"[soundreal] Failed to update sound observations: {exc}")
            elapsed = time.perf_counter() - loop_start
            sleep_duration = period - elapsed
            if sleep_duration > 0:
                self._stop_event.wait(sleep_duration)

    def _update_from_snapshot(self, snapshot: list[np.ndarray]) -> None:
        mic_signals_list = [audio.T for audio in snapshot]
        music_results = [
            estimate_music(self.sound_camera, mic_signals, idx)
            for idx, mic_signals in enumerate(mic_signals_list)
        ]
        sound_maps = [
            generate_soundmap_from_doa_fast(
                self.sound_camera,
                music_results[idx],
                np.asarray(self.sound_camera.mic_positions[idx], dtype=np.float32),
            )
            for idx in range(len(music_results))
        ]

        if sound_maps:
            stacked_maps = np.stack(sound_maps, axis=2)
            per_array_uint8 = self.sound_camera._normalize_to_uint8(stacked_maps)
            sound0 = self.sound_camera._split_channels(per_array_uint8, 0, 3)
            sound1 = self.sound_camera._split_channels(per_array_uint8, 3, 6)

            combined_map = build_combined_sound_map(sound_maps)
            top_peaks = compute_top_peaks(self.sound_camera, combined_map)
            reconstructed = reconstruct_spotformed_wavs(
                self.sound_camera,
                sound_maps,
                mic_signals_list,
                top_peaks,
            )
            if reconstructed:
                spec = create_spectrogram_image_from_audio(self.sound_camera, reconstructed[0])
            else:
                spec = np.zeros_like(sound0)
        else:
            sound0 = np.zeros_like(self.cached_images["sound0"])
            sound1 = np.zeros_like(self.cached_images["sound1"])
            spec = np.zeros_like(self.cached_images["spec"])

        with self._lock:
            self.cached_images["sound0"] = sound0
            self.cached_images["sound1"] = sound1
            self.cached_images["spec"] = spec


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
    info_lines = []
    if condition is not None:
        info_lines.append(
            f"speaker={condition.speaker} sound={condition.sound_label} ({condition.sound_path.name})"
        )
    effective_fps = status.get("effective_fps")
    latest_age_s = status.get("latest_age_s")
    info_lines.append(
        "updates={updates} late={late} fps={fps} compute={compute:.1f}ms age={age}".format(
            updates=status.get("update_count"),
            late=status.get("late_cycle_count"),
            fps="n/a" if effective_fps is None else f"{effective_fps:.2f}",
            compute=1000.0 * float(status.get("last_compute_duration_s") or 0.0),
            age="n/a" if latest_age_s is None else f"{latest_age_s:.3f}s",
        )
    )
    last_error = status.get("last_error")
    if last_error:
        info_lines.append(f"last_error={last_error}")

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
        return Path(output_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("debug") / f"soundreal_debug_{timestamp}.mp4"


def open_debug_video_writer(output_path: Path, frame_size: tuple[int, int], fps: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
    return writer


def run_sound_debug_session(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    condition = (
        sample_sound_episode_condition(rng)
        if args.sound_index is None or args.speaker is None
        else SoundEpisodeCondition(
            sound_index=int(args.sound_index),
            sound_label=SOUND_LABELS[int(args.sound_index)],
            sound_path=SOUND_FILES[int(args.sound_index)],
            speaker=str(args.speaker),
        )
    )

    sound_source = RealSoundObservationSource(
        explicit_device_ids=parse_device_ids_env("SOUNDREAL_DEVICE_IDS")
        if args.input_device_ids is None
        else [int(part.strip()) for part in args.input_device_ids.split(",") if part.strip()],
        observation_height=args.height,
        observation_width=args.width,
        audio_fps=args.audio_fps,
    )
    audio_player = None if args.no_audio else LoopingStereoPlayer(output_device=args.output_device)
    display_enabled = not args.no_display
    writer = None
    window_name = "soundreal_debug"
    frame_interval_s = 1.0 / max(1, args.video_fps)
    last_frame_time = 0.0

    try:
        sound_source.start()
        if not sound_source.wait_until_ready(timeout_s=args.ready_timeout_s):
            print("[soundreal] Audio buffers are still warming up. Initial debug frames may be zeros.")

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

            images = sound_source.get_latest_images()
            status = sound_source.get_debug_status()
            panel = create_sound_debug_panel(images, status, condition)

            if writer is None and (args.save_mp4 or not display_enabled):
                output_path = make_debug_output_path(args.output_path)
                writer = open_debug_video_writer(
                    output_path,
                    frame_size=(panel.shape[1], panel.shape[0]),
                    fps=args.video_fps,
                )
                print(f"[soundreal] Saving debug video to {output_path}")

            if display_enabled:
                cv2.imshow(window_name, panel)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if writer is not None and now - last_frame_time >= frame_interval_s:
                writer.write(panel)
                last_frame_time = now

            time.sleep(max(0.0, min(0.02, frame_interval_s / 4.0)))

    except KeyboardInterrupt:
        print("\n[soundreal] Debug session interrupted by user.")
    finally:
        if writer is not None:
            writer.release()
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
    parser.add_argument("--audio_fps", type=int, default=AUDIO_FPS, help="target sound observation FPS")
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
        default=2.0,
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
