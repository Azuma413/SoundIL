from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import sounddevice as sd
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
from scipy.signal import spectrogram, stft

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.tasks.sound_camera import SoundCamera, SoundConfig


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
DEFAULT_MIC_CHANNELS = 8
ARECORD_SAMPLE_FORMAT = "S24_3LE"


@dataclass(frozen=True)
class CaptureResult:
    device_id: int
    device_name: str
    audio: np.ndarray


class DummyTarget:
    def get_pos(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


def apply_azimuth_offset_rad(angle_rad: np.ndarray | float, offset_deg: float = AZIMUTH_OFFSET_DEG) -> np.ndarray:
    return np.asarray(angle_rad) + np.deg2rad(offset_deg)


def wrap_angle_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def extract_hw_index(device_name: str) -> int | None:
    match = re.search(r"\(hw:(\d+),\d+\)", device_name)
    return int(match.group(1)) if match else None


def get_map_bounds() -> tuple[float, float, float, float]:
    half_map = MAP_SIZE_M / 2.0
    x, y = ROOM_CENTER_XY
    return x - half_map, x + half_map, y - half_map, y + half_map


def get_map_axes(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = get_map_bounds()
    x_coords = np.linspace(x_min, x_max, width, dtype=np.float32)
    y_coords = np.linspace(y_min, y_max, height, dtype=np.float32)
    return x_coords, y_coords


def pcm_to_float32(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float32, copy=False)
    if audio.dtype == np.int16:
        scale = 32768.0
    elif audio.dtype == np.int32:
        scale = 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV dtype: {audio.dtype}")
    return audio.astype(np.float32) / scale


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="4台のTAMAGO-03から同時録音し、DOAベースのSound Mapとスペクトログラムを可視化します。"
    )
    for name, arg_type, default, help_text in [
        ("duration", float, 3.0, "録音時間 [sec]"),
        ("samplerate", int, 16000, "サンプリング周波数 [Hz]"),
        ("channels", int, DEFAULT_MIC_CHANNELS, "各TAMAGOデバイスから使うチャンネル数"),
        ("blocksize", int, 0, "InputStreamのブロックサイズ。0 の場合は backend 既定値を使います。"),
        ("nfft", int, 512, "MUSIC/STFTで使うFFT長"),
        ("combined-map-power", float, 4.0, "正規化後のCombined Sound Mapに掛ける指数。1より大きいと分布がシャープになります。"),
        ("startup-wait", float, 0.0, "全入力ストリーム開始後に待つウォームアップ時間 [sec]"),
        ("capture-timeout", float, 2.0, "録音完了を待つ追加タイムアウト [sec]"),
        ("num-peaks", int, 1, "Sound Mapから抽出するピーク数"),
    ]:
        parser.add_argument(f"--{name}", type=arg_type, default=default, help=help_text)
    parser.add_argument(
        "--device-ids",
        type=int,
        nargs="*",
        default=None,
        help="使用する入力デバイスID。未指定時は 0 1 8 9 を優先して自動選択します。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tamago_sound_map"),
        help="録音WAVと可視化画像の保存先",
    )
    for flag, help_text in [
        ("save-wav", "各デバイスの録音結果をWAV保存します。"),
        ("no-show", "matplotlibのウィンドウを表示せず保存だけ行います。"),
        ("spotforming-spectrogram", "重いNMFベースのスポットフォーミングスペクトログラムを計算します。"),
        ("save-reconstructed-wav", "NMFベースで再構成した音声をWAV保存します。"),
    ]:
        parser.add_argument(f"--{flag}", action="store_true", help=help_text)
    return parser


def query_input_devices() -> list[tuple[int, dict]]:
    return [(idx, device) for idx, device in enumerate(sd.query_devices()) if int(device["max_input_channels"]) > 0]


def select_tamago_devices(explicit_ids: list[int] | None) -> list[tuple[int, dict]]:
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
    if len(selected) == 4:
        return selected

    available = "\n".join(
        f"[{idx}] {device['name']} | input_channels={int(device['max_input_channels'])}"
        for idx, device in query_input_devices()
    )
    raise RuntimeError(
        "4台のTAMAGO-03を自動選択できませんでした。--device-ids を指定してください。\n"
        f"Available input devices:\n{available}"
    )


def build_rectangular_mic_positions(devices: list[tuple[int, dict]]) -> list[list[float]]:
    half_long = RECTANGLE_LONG_SIDE_M / 2.0
    half_short = RECTANGLE_SHORT_SIDE_M / 2.0
    cx, cy = ROOM_CENTER_XY
    corners = [
        [cx + half_long, cy + half_short, ARRAY_HEIGHT_M],
        [cx + half_long, cy - half_short, ARRAY_HEIGHT_M],
        [cx - half_long, cy - half_short, ARRAY_HEIGHT_M],
        [cx - half_long, cy + half_short, ARRAY_HEIGHT_M],
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


def print_rectangular_layout(devices: list[tuple[int, dict]], mic_positions: list[list[float]]) -> None:
    print(f"Using rectangular array layout (clockwise): {' -> '.join(f'hw:{idx}' for idx in RECTANGLE_HW_ORDER_CLOCKWISE)}")
    print(f"Rectangle size: short={RECTANGLE_SHORT_SIDE_M:.2f} m, long={RECTANGLE_LONG_SIDE_M:.2f} m")
    for (device_id, device), mic_center in zip(devices, mic_positions):
        print(f"  device=[{device_id}] hw:{extract_hw_index(str(device['name']))} | center={np.round(mic_center, 3)}")


def capture_from_devices(
    devices: list[tuple[int, dict]],
    duration: float,
    samplerate: int,
    channels: int,
    blocksize: int,
    startup_wait: float,
    capture_timeout: float,
) -> list[CaptureResult]:
    if channels <= 0:
        raise ValueError("channels must be >= 1")
    del blocksize
    warmup_frames = max(0, int(round(startup_wait * samplerate)))
    target_frames = int(round(duration * samplerate))
    total_frames = warmup_frames + target_frames
    if target_frames <= 0:
        raise ValueError("duration is too short to capture any audio frames")
    print("Capturing from devices: " + ", ".join(f"[{device_id}] {device['name']}" for device_id, device in devices))

    timeout_sec = duration + startup_wait + capture_timeout + 5.0
    captures: dict[int, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="tamago_sound_map_") as tmpdir:
        output_dir = Path(tmpdir)
        processes: list[tuple[int, dict, Path, subprocess.Popen[str]]] = []
        for device_id, device in devices:
            available_channels = int(device["max_input_channels"])
            if available_channels < channels:
                raise RuntimeError(
                    f"Device {device_id} has only {available_channels} input channels, but {channels} channels are required."
                )
            hw_index = extract_hw_index(str(device["name"]))
            if hw_index is None:
                raise RuntimeError(f"Could not extract ALSA hw index from device [{device_id}] {device['name']}.")
            wav_path = output_dir / f"device_{device_id}.wav"
            cmd = [
                "arecord",
                "-q",
                "-M",
                "-D",
                f"hw:{hw_index},0",
                "-t",
                "wav",
                "-c",
                str(channels),
                "-f",
                ARECORD_SAMPLE_FORMAT,
                "-r",
                str(samplerate),
                "--samples",
                str(total_frames),
                "--fatal-errors",
                str(wav_path),
            ]
            processes.append(
                (
                    device_id,
                    device,
                    wav_path,
                    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True),
                )
            )

        errors = []
        for device_id, device, wav_path, process in processes:
            try:
                _, stderr = process.communicate(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                process.kill()
                _, stderr = process.communicate()
                errors.append(f"device {device_id}: timed out ({stderr.strip() or 'no stderr'})")
                continue
            if process.returncode != 0:
                errors.append(f"device {device_id}: {stderr.strip() or f'arecord exited with code {process.returncode}'}")
                continue
            read_samplerate, audio = wavfile.read(wav_path)
            if read_samplerate != samplerate:
                errors.append(f"device {device_id}: unexpected samplerate {read_samplerate}")
                continue
            if audio.ndim == 1:
                audio = audio[:, None]
            if audio.shape[0] < total_frames:
                errors.append(f"device {device_id}: {audio.shape[0]}/{total_frames} frames captured")
                continue
            captures[device_id] = pcm_to_float32(audio[warmup_frames:total_frames]).copy()

        if errors:
            raise RuntimeError("Audio capture failed: " + ", ".join(errors))

    results = [CaptureResult(device_id=device_id, device_name=str(device["name"]), audio=captures[device_id]) for device_id, device in devices]
    min_samples = min(result.audio.shape[0] for result in results)
    return [CaptureResult(result.device_id, result.device_name, result.audio[:min_samples]) for result in results]


def create_sound_camera(num_arrays: int, samplerate: int, nfft: int, num_peaks: int) -> SoundCamera:
    config = SoundConfig(
        mic_array_num=num_arrays,
        mics_per_array=DEFAULT_MIC_CHANNELS,
        fs=samplerate,
        nfft=nfft,
        num_peaks=num_peaks,
        observation_height=128,
        observation_width=128,
        use_spectrogram=True,
        use_soundmap=True,
        use_gaussian_filter=False,
        processing_time=3.0,
    )
    return SoundCamera(target=DummyTarget(), config=config)


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
    if (spec_sum := np.sum(spec)) != 0:
        spec = spec / spec_sum

    x_coords, y_coords = get_map_axes(sound_camera.config.observation_width, sound_camera.config.observation_height)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="xy")
    theta = apply_azimuth_offset_rad(np.arctan2(y_grid - mic_center[1], x_grid - mic_center[0]))
    angle_idx = (theta / (2 * np.pi / len(spec))).astype(np.int32) % len(spec)
    distance = np.hypot(y_grid - mic_center[1], x_grid - mic_center[0])
    distance_weight = 1.0 / np.power(distance + DOA_DISTANCE_FLOOR_M, DOA_DISTANCE_DECAY_EXPONENT)
    return (spec[angle_idx] * distance_weight).astype(np.float32)


def build_combined_sound_map(sound_maps: list[np.ndarray], power: float) -> np.ndarray:
    return gaussian_filter(np.sum([np.power(sound_map, power, dtype=np.float32) for sound_map in sound_maps], axis=0), sigma=1.0)


def compute_top_peaks(sound_camera: SoundCamera, combined_sound_map: np.ndarray) -> list[tuple[float, float]]:
    topk = sound_camera.config.num_peaks
    indices = np.argpartition(combined_sound_map.ravel(), -topk)[-topk:]
    indices = indices[np.argsort(combined_sound_map.ravel()[indices])[::-1]]
    x_coords, y_coords = get_map_axes(sound_camera.config.observation_width, sound_camera.config.observation_height)
    return [
        (float(x_coords[col]), float(y_coords[row]))
        for row, col in (np.unravel_index(int(flat_index), combined_sound_map.shape) for flat_index in indices)
    ]


def summarize_arrays(sound_camera: SoundCamera, music_results: list[pra.doa.MUSIC]) -> list[dict]:
    return [
        {
            "index": idx,
            "center": sound_camera.mic_positions[idx],
            "azimuth_deg": float(
                wrap_angle_deg(float(np.rad2deg(apply_azimuth_offset_rad(np.atleast_1d(music.azimuth_recon)[0]))))
            ),
        }
        for idx, music in enumerate(music_results)
    ]


def save_multichannel_wav(path: Path, samplerate: int, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, samplerate, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))


def reconstruct_spotformed_wavs(
    sound_camera: SoundCamera,
    sound_maps: list[np.ndarray],
    mic_signals_list: list[np.ndarray],
) -> list[np.ndarray]:
    smoothed_map = gaussian_filter(np.sum(sound_maps, axis=0), sound_camera.config.gaussian_sigma)
    top_peaks = sound_camera._find_top_k_peaks(smoothed_map, sound_camera.config.num_peaks)
    if not top_peaks:
        print("Spotforming: No peaks found.")
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
            beamformed_signals.append(sound_camera._ds_beamform(mic_signals, mic_array_abs - np.mean(mic_array_abs, axis=0), theta_deg))
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
    return sound_camera._convert_spectrogram_to_image(10 * np.log10(np.abs(zxx) ** 2 + 1e-10))


def create_visualization(
    sound_camera: SoundCamera,
    captures: list[CaptureResult],
    sound_maps: list[np.ndarray],
    combined_sound_map: np.ndarray,
    top_peaks: list[tuple[float, float]],
    spectrogram_image: np.ndarray | None,
    output_path: Path,
) -> None:
    x_min, x_max, y_min, y_max = get_map_bounds()
    extent = [x_min, x_max, y_min, y_max]
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.0, 1.0])

    ax_map = fig.add_subplot(gs[0, 0])
    map_im = ax_map.imshow(combined_sound_map, origin="lower", cmap="inferno", extent=extent, aspect="equal")
    fig.colorbar(map_im, ax=ax_map, fraction=0.046, pad=0.04, label="Sound map intensity")
    ax_map.set_title("Combined Sound Map")
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    for idx, center in enumerate(sound_camera.mic_positions):
        ax_map.scatter(center[0], center[1], c="cyan", s=40)
        ax_map.text(center[0] + 0.05, center[1] + 0.05, f"Array {idx}", color="white", fontsize=9)
    for peak_x, peak_y in top_peaks:
        ax_map.scatter(peak_x, peak_y, c="lime", marker="x", s=120)

    ax_spec = fig.add_subplot(gs[0, 1])
    if spectrogram_image is not None:
        ax_spec.imshow(spectrogram_image, origin="lower", aspect="auto", cmap="magma")
        ax_spec.set_title("Spotformed Spectrogram")
    else:
        freqs, times, spec = spectrogram(
            captures[0].audio[:, 0],
            fs=sound_camera.config.fs,
            nperseg=sound_camera.config.nfft,
            noverlap=sound_camera.config.nfft // 2,
        )
        mesh = ax_spec.pcolormesh(times, freqs, 10.0 * np.log10(spec + 1e-12), shading="gouraud", cmap="magma")
        fig.colorbar(mesh, ax=ax_spec, fraction=0.046, pad=0.04, label="Power [dB]")
        ax_spec.set_title("Raw Spectrogram (Array 0, Ch 0)")
        ax_spec.set_xlabel("Time [s]")
        ax_spec.set_ylabel("Frequency [Hz]")

    ax_wave = fig.add_subplot(gs[0, 2])
    ax_wave.plot(np.arange(captures[0].audio.shape[0]) / sound_camera.config.fs, captures[0].audio[:, 0], linewidth=0.8)
    ax_wave.set_title("Raw Waveform (Array 0, Ch 0)")
    ax_wave.set_xlabel("Time [s]")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.grid(True, alpha=0.3)

    per_array_uint8 = sound_camera._normalize_to_uint8(np.stack(sound_maps, axis=2))
    slots = [(1, 0), (1, 1), (1, 2), (2, 0)]
    for idx, (row, col) in enumerate(slots):
        sub_ax = fig.add_subplot(gs[row, col])
        if idx < len(sound_maps):
            sub_ax.imshow(per_array_uint8[:, :, idx], origin="lower", cmap="inferno", extent=extent, aspect="equal")
            sub_ax.set_title(f"Array {idx} Sound Map")
            sub_ax.set_xlabel("X [m]")
            sub_ax.set_ylabel("Y [m]")
        else:
            sub_ax.axis("off")
    fig.add_subplot(gs[2, 1]).axis("off")
    fig.add_subplot(gs[2, 2]).axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")


def main() -> None:
    overall_start = time.perf_counter()
    args = build_parser().parse_args()
    step_start = time.perf_counter()
    devices = select_tamago_devices(args.device_ids)
    print(f"[timing] device selection: {time.perf_counter() - step_start:.3f} sec")

    def setup_camera() -> SoundCamera:
        camera = create_sound_camera(len(devices), args.samplerate, args.nfft, args.num_peaks)
        camera.mic_positions = build_rectangular_mic_positions(devices)
        camera.config.processing_time = args.duration
        print_rectangular_layout(devices, camera.mic_positions)
        return camera

    step_start = time.perf_counter()
    sound_camera = setup_camera()
    print(f"[timing] sound camera setup: {time.perf_counter() - step_start:.3f} sec")

    step_start = time.perf_counter()
    captures = capture_from_devices(
        devices=devices,
        duration=args.duration,
        samplerate=args.samplerate,
        channels=args.channels,
        blocksize=args.blocksize,
        startup_wait=args.startup_wait,
        capture_timeout=args.capture_timeout,
    )
    print(f"[timing] audio capture: {time.perf_counter() - step_start:.3f} sec")

    mic_signals_list = [capture.audio.T for capture in captures]
    step_start = time.perf_counter()
    music_results = [estimate_music(sound_camera, mic_signals, idx) for idx, mic_signals in enumerate(mic_signals_list)]
    print(f"[timing] music estimation: {time.perf_counter() - step_start:.3f} sec")

    step_start = time.perf_counter()
    sound_maps = [
        generate_soundmap_from_doa_fast(sound_camera, music_result, np.asarray(sound_camera.mic_positions[idx]))
        for idx, music_result in enumerate(music_results)
    ]
    combined_sound_map = build_combined_sound_map(sound_maps, args.combined_map_power)
    top_peaks = compute_top_peaks(sound_camera, combined_sound_map)
    print(f"[timing] sound map generation: {time.perf_counter() - step_start:.3f} sec")

    spectrogram_image = None
    reconstructed_wavs: list[np.ndarray] = []
    if args.spotforming_spectrogram or args.save_reconstructed_wav:
        step_start = time.perf_counter()
        reconstructed_wavs = reconstruct_spotformed_wavs(sound_camera, sound_maps, mic_signals_list)
        if args.spotforming_spectrogram and reconstructed_wavs:
            spectrogram_image = create_spectrogram_image_from_audio(sound_camera, reconstructed_wavs[0])
        print(f"[timing] spotforming reconstruction: {time.perf_counter() - step_start:.3f} sec")

    for summary, capture in zip(summarize_arrays(sound_camera, music_results), captures):
        print(
            f"Array {summary['index']} | device=[{capture.device_id}] {capture.device_name} "
            f"| center={np.round(summary['center'], 3)} | azimuth={summary['azimuth_deg']:.2f} deg"
        )
    if top_peaks:
        print("Top peaks on sound map:")
        for idx, (peak_x, peak_y) in enumerate(top_peaks):
            print(f"  Peak {idx}: x={peak_x:.2f} m, y={peak_y:.2f} m")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_wav:
        step_start = time.perf_counter()
        for idx, capture in enumerate(captures):
            save_multichannel_wav(args.output_dir / f"tamago_array_{idx}_device_{capture.device_id}.wav", args.samplerate, capture.audio)
        print(f"[timing] wav export: {time.perf_counter() - step_start:.3f} sec")
    if args.save_reconstructed_wav:
        step_start = time.perf_counter()
        for idx, reconstructed_wav in enumerate(reconstructed_wavs):
            save_multichannel_wav(args.output_dir / f"reconstructed_peak_{idx}.wav", args.samplerate, reconstructed_wav)
        print(f"[timing] reconstructed wav export: {time.perf_counter() - step_start:.3f} sec")

    step_start = time.perf_counter()
    create_visualization(
        sound_camera=sound_camera,
        captures=captures,
        sound_maps=sound_maps,
        combined_sound_map=combined_sound_map,
        top_peaks=top_peaks,
        spectrogram_image=spectrogram_image,
        output_path=args.output_dir / "tamago_sound_map.png",
    )
    print(f"[timing] visualization export: {time.perf_counter() - step_start:.3f} sec")
    print(f"[timing] total: {time.perf_counter() - overall_start:.3f} sec")

    can_show = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if not args.no_show and can_show:
        plt.show()
    elif not args.no_show:
        print("Skipping plt.show() because no GUI display is available. Use --no-show to suppress this message.")


if __name__ == "__main__":
    main()
