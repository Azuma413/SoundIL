from __future__ import annotations

import argparse
import contextlib
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

from env.tasks.sound_camera import SoundCamera, SoundConfig


TAMAGO_DEVICE_IDS = [0, 1, 8, 9]


@dataclass
class CaptureResult:
    device_id: int
    device_name: str
    audio: np.ndarray


class DummyTarget:
    def get_pos(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="4台のTAMAGO-03から同時録音し、DOAベースのSound Mapとスペクトログラムを可視化します。"
    )
    parser.add_argument("--duration", type=float, default=3.0, help="録音時間 [sec]")
    parser.add_argument("--samplerate", type=int, default=16000, help="サンプリング周波数 [Hz]")
    parser.add_argument("--channels", type=int, default=8, help="各TAMAGOデバイスから使うチャンネル数")
    parser.add_argument("--blocksize", type=int, default=1024, help="InputStreamのブロックサイズ")
    parser.add_argument("--nfft", type=int, default=512, help="MUSIC/STFTで使うFFT長")
    parser.add_argument(
        "--device-ids",
        type=int,
        nargs="*",
        default=None,
        help="使用する入力デバイスID。未指定時は 0 1 8 9 を優先して自動選択します。",
    )
    parser.add_argument(
        "--num-peaks",
        type=int,
        default=1,
        help="Sound Mapから抽出するピーク数",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tamago_sound_map"),
        help="録音WAVと可視化画像の保存先",
    )
    parser.add_argument(
        "--save-wav",
        action="store_true",
        help="各デバイスの録音結果をWAV保存します。",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="matplotlibのウィンドウを表示せず保存だけ行います。",
    )
    return parser


def query_input_devices() -> list[tuple[int, dict]]:
    devices = sd.query_devices()
    return [
        (idx, device)
        for idx, device in enumerate(devices)
        if int(device["max_input_channels"]) > 0
    ]


def select_tamago_devices(explicit_ids: list[int] | None) -> list[tuple[int, dict]]:
    inputs = {idx: device for idx, device in query_input_devices()}
    if explicit_ids:
        selected = []
        for device_id in explicit_ids:
            if device_id not in inputs:
                raise RuntimeError(f"Input device {device_id} was not found.")
            selected.append((device_id, inputs[device_id]))
        return selected

    selected = []
    for device_id in TAMAGO_DEVICE_IDS:
        device = inputs.get(device_id)
        if device is None:
            continue
        if "tamago" in device["name"].lower():
            selected.append((device_id, device))

    if len(selected) != 4:
        available = "\n".join(
            f"[{idx}] {device['name']} | input_channels={int(device['max_input_channels'])}"
            for idx, device in query_input_devices()
        )
        raise RuntimeError(
            "4台のTAMAGO-03を自動選択できませんでした。--device-ids を指定してください。\n"
            f"Available input devices:\n{available}"
        )
    return selected


def capture_from_devices(
    devices: list[tuple[int, dict]],
    duration: float,
    samplerate: int,
    channels: int,
    blocksize: int,
) -> list[CaptureResult]:
    buffers: dict[int, list[np.ndarray]] = {device_id: [] for device_id, _ in devices}

    def make_callback(device_id: int):
        def callback(indata, frames, time_info, status):
            if status:
                print(f"[device {device_id}] {status}")
            buffers[device_id].append(indata.copy())

        return callback

    streams = []
    with contextlib.ExitStack() as stack:
        for device_id, device in devices:
            available_channels = int(device["max_input_channels"])
            use_channels = min(channels, available_channels)
            stream = sd.InputStream(
                samplerate=samplerate,
                blocksize=blocksize,
                dtype="float32",
                channels=use_channels,
                device=device_id,
                callback=make_callback(device_id),
            )
            streams.append((device_id, device, stack.enter_context(stream)))

        print(
            "Capturing from devices: "
            + ", ".join(f"[{device_id}] {device['name']}" for device_id, device, _ in streams)
        )
        time.sleep(duration)

    results: list[CaptureResult] = []
    for device_id, device in [(device_id, device) for device_id, device, _ in streams]:
        if not buffers[device_id]:
            raise RuntimeError(f"No audio frames were captured from device {device_id}.")
        audio = np.concatenate(buffers[device_id], axis=0)
        results.append(
            CaptureResult(
                device_id=device_id,
                device_name=str(device["name"]),
                audio=audio,
            )
        )

    min_samples = min(result.audio.shape[0] for result in results)
    return [
        CaptureResult(
            device_id=result.device_id,
            device_name=result.device_name,
            audio=result.audio[:min_samples],
        )
        for result in results
    ]


def create_sound_camera(num_arrays: int, samplerate: int, nfft: int, num_peaks: int) -> SoundCamera:
    config = SoundConfig(
        mic_array_num=num_arrays,
        mics_per_array=8,
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


def estimate_music(
    sound_camera: SoundCamera,
    mic_signals: np.ndarray,
    array_index: int,
) -> pra.doa.MUSIC:
    stft_results = []
    for ch in range(mic_signals.shape[0]):
        _, _, zxx = stft(
            mic_signals[ch],
            fs=sound_camera.config.fs,
            nperseg=sound_camera.config.nfft,
            noverlap=sound_camera.config.nfft // 2,
        )
        stft_results.append(zxx)
    Z = np.asarray(stft_results)

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
    doa.locate_sources(Z)
    return doa


def normalize_sound_maps(sound_camera: SoundCamera, sound_maps: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(sound_maps, axis=2)
    return sound_camera._normalize_to_uint8(stacked)


def build_combined_sound_map(sound_maps: list[np.ndarray]) -> np.ndarray:
    combined = np.sum(sound_maps, axis=0)
    return gaussian_filter(combined, sigma=1.0)


def compute_top_peaks(
    sound_camera: SoundCamera,
    combined_sound_map: np.ndarray,
) -> list[tuple[float, float]]:
    return sound_camera._find_top_k_peaks(combined_sound_map, sound_camera.config.num_peaks)


def summarize_arrays(
    sound_camera: SoundCamera,
    music_results: list[pra.doa.MUSIC],
) -> list[dict]:
    summaries = []
    for idx, music in enumerate(music_results):
        mic_center = sound_camera.mic_positions[idx]
        azimuth_deg = np.rad2deg(np.atleast_1d(music.azimuth_recon)[0])
        summaries.append(
            {
                "index": idx,
                "center": mic_center,
                "azimuth_deg": float(azimuth_deg),
            }
        )
    return summaries


def save_multichannel_wav(path: Path, samplerate: int, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, samplerate, (clipped * 32767).astype(np.int16))


def create_visualization(
    sound_camera: SoundCamera,
    captures: list[CaptureResult],
    sound_maps: list[np.ndarray],
    combined_sound_map: np.ndarray,
    top_peaks: list[tuple[float, float]],
    spectrogram_image: np.ndarray | None,
    output_path: Path,
) -> None:
    per_array_uint8 = normalize_sound_maps(sound_camera, sound_maps)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.0, 1.0])

    ax_map = fig.add_subplot(gs[0, 0])
    map_im = ax_map.imshow(combined_sound_map.T, origin="lower", cmap="inferno", extent=[0, 10, 0, 10])
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
        spec_db = 10.0 * np.log10(spec + 1e-12)
        mesh = ax_spec.pcolormesh(times, freqs, spec_db, shading="gouraud", cmap="magma")
        fig.colorbar(mesh, ax=ax_spec, fraction=0.046, pad=0.04, label="Power [dB]")
        ax_spec.set_title("Raw Spectrogram (Array 0, Ch 0)")
        ax_spec.set_xlabel("Time [s]")
        ax_spec.set_ylabel("Frequency [Hz]")

    ax_wave = fig.add_subplot(gs[0, 2])
    time_axis = np.arange(captures[0].audio.shape[0]) / sound_camera.config.fs
    ax_wave.plot(time_axis, captures[0].audio[:, 0], linewidth=0.8)
    ax_wave.set_title("Raw Waveform (Array 0, Ch 0)")
    ax_wave.set_xlabel("Time [s]")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.grid(True, alpha=0.3)

    array_positions = [(1, 0), (1, 1), (1, 2), (2, 0)]
    for idx in range(min(4, len(sound_maps))):
        row, col = array_positions[idx]
        sub_ax = fig.add_subplot(gs[row, col])
        sub_ax.imshow(per_array_uint8[:, :, idx].T, origin="lower", cmap="inferno", extent=[0, 10, 0, 10])
        sub_ax.set_title(f"Array {idx} Sound Map")
        sub_ax.set_xlabel("X [m]")
        sub_ax.set_ylabel("Y [m]")

    if len(sound_maps) < 4:
        for row, col in array_positions[len(sound_maps):]:
            fig.add_subplot(gs[row, col]).axis("off")
    else:
        fig.add_subplot(gs[2, 1]).axis("off")
        fig.add_subplot(gs[2, 2]).axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")


def main() -> None:
    args = build_parser().parse_args()
    devices = select_tamago_devices(args.device_ids)
    sound_camera = create_sound_camera(
        num_arrays=len(devices),
        samplerate=args.samplerate,
        nfft=args.nfft,
        num_peaks=args.num_peaks,
    )
    sound_camera.config.processing_time = args.duration

    captures = capture_from_devices(
        devices=devices,
        duration=args.duration,
        samplerate=args.samplerate,
        channels=args.channels,
        blocksize=args.blocksize,
    )

    mic_signals_list = [capture.audio.T for capture in captures]
    music_results = [estimate_music(sound_camera, mic_signals, idx) for idx, mic_signals in enumerate(mic_signals_list)]
    sound_maps = [
        sound_camera._generate_soundmap_from_doa(music_result, sound_camera.mic_positions[idx])
        for idx, music_result in enumerate(music_results)
    ]
    combined_sound_map = build_combined_sound_map(sound_maps)
    top_peaks = compute_top_peaks(sound_camera, combined_sound_map)
    spectrogram_image = sound_camera._generate_spectrogram(sound_maps, mic_signals_list)

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
        for idx, capture in enumerate(captures):
            save_multichannel_wav(
                args.output_dir / f"tamago_array_{idx}_device_{capture.device_id}.wav",
                args.samplerate,
                capture.audio,
            )

    image_path = args.output_dir / "tamago_sound_map.png"
    create_visualization(
        sound_camera=sound_camera,
        captures=captures,
        sound_maps=sound_maps,
        combined_sound_map=combined_sound_map,
        top_peaks=top_peaks,
        spectrogram_image=spectrogram_image,
        output_path=image_path,
    )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
