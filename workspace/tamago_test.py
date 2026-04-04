from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import spectrogram


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="接続されたマイクロフォンアレイから音声を録音し、波形とスペクトログラムを可視化します。"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="利用可能なオーディオデバイス一覧を表示して終了します。",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="入力デバイス番号。未指定時はシステム既定の入力デバイスを使います。",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="録音サンプリング周波数 [Hz]。",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="録音時間 [sec]。",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help="録音チャンネル数。未指定時はデバイスの最大入力チャンネル数を使います。",
    )
    parser.add_argument(
        "--plot-channels",
        type=int,
        default=4,
        help="可視化するチャンネル数の上限。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="録音結果を保存する WAV ファイルパス。",
    )
    return parser


def list_input_devices() -> None:
    devices = sd.query_devices()
    print("Available audio devices:")
    for idx, device in enumerate(devices):
        max_input_channels = int(device["max_input_channels"])
        if max_input_channels <= 0:
            continue
        default_rate = int(device["default_samplerate"])
        print(
            f"[{idx}] {device['name']} | input_channels={max_input_channels} "
            f"| default_samplerate={default_rate}"
        )


def resolve_input_device(device_id: int | None) -> tuple[int, dict]:
    if device_id is None:
        default_input, _ = sd.default.device
        if default_input is None or int(default_input) < 0:
            raise RuntimeError(
                "既定の入力デバイスが見つかりません。--list-devices で確認し、--device を指定してください。"
            )
        device_id = int(default_input)

    device_info = sd.query_devices(device_id, "input")
    return device_id, device_info


def record_multichannel_audio(
    samplerate: int,
    duration: float,
    device_id: int | None,
    channels: int | None,
) -> tuple[np.ndarray, int, dict]:
    resolved_device_id, device_info = resolve_input_device(device_id)
    available_channels = int(device_info["max_input_channels"])
    if available_channels <= 0:
        raise RuntimeError(f"Device {resolved_device_id} has no input channels.")

    if channels is None:
        channels = available_channels
    channels = min(channels, available_channels)
    if channels <= 0:
        raise ValueError("channels must be >= 1")

    frames = int(duration * samplerate)
    print(
        f"Recording from device [{resolved_device_id}] {device_info['name']} "
        f"at {samplerate} Hz, {channels} channels, {duration:.2f} sec..."
    )
    audio = sd.rec(
        frames,
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
        device=resolved_device_id,
    )
    sd.wait()
    print("Recording finished.")
    return audio, resolved_device_id, device_info


def save_audio(output_path: Path, samplerate: int, audio: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    wavfile.write(output_path, samplerate, (clipped * 32767).astype(np.int16))
    print(f"Saved recording to: {output_path}")


def plot_audio(audio: np.ndarray, samplerate: int, max_channels: int, title: str) -> None:
    num_samples, num_channels = audio.shape
    channels_to_plot = min(max_channels, num_channels)
    time_axis = np.arange(num_samples) / samplerate

    fig, axes = plt.subplots(
        channels_to_plot,
        2,
        figsize=(14, 3.5 * channels_to_plot),
        squeeze=False,
    )
    fig.suptitle(title)

    for ch in range(channels_to_plot):
        waveform_ax = axes[ch, 0]
        spec_ax = axes[ch, 1]

        waveform_ax.plot(time_axis, audio[:, ch], linewidth=0.8)
        waveform_ax.set_title(f"Channel {ch}: waveform")
        waveform_ax.set_xlabel("Time [s]")
        waveform_ax.set_ylabel("Amplitude")
        waveform_ax.grid(True, alpha=0.3)

        freqs, times, spec = spectrogram(audio[:, ch], fs=samplerate, nperseg=1024, noverlap=512)
        spec_db = 10.0 * np.log10(spec + 1e-12)
        mesh = spec_ax.pcolormesh(times, freqs, spec_db, shading="gouraud", cmap="magma")
        spec_ax.set_title(f"Channel {ch}: spectrogram")
        spec_ax.set_xlabel("Time [s]")
        spec_ax.set_ylabel("Frequency [Hz]")
        fig.colorbar(mesh, ax=spec_ax, label="Power [dB]")

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    audio, device_id, device_info = record_multichannel_audio(
        samplerate=args.samplerate,
        duration=args.duration,
        device_id=args.device,
        channels=args.channels,
    )

    peak_per_channel = np.max(np.abs(audio), axis=0)
    rms_per_channel = np.sqrt(np.mean(np.square(audio), axis=0))
    print(f"Device [{device_id}] {device_info['name']}")
    print(f"Captured shape: {audio.shape} (samples, channels)")
    print(f"Peak per channel: {np.array2string(peak_per_channel, precision=4)}")
    print(f"RMS per channel:  {np.array2string(rms_per_channel, precision=4)}")

    if args.output is not None:
        save_audio(args.output, args.samplerate, audio)

    plot_audio(
        audio=audio,
        samplerate=args.samplerate,
        max_channels=args.plot_channels,
        title=f"Microphone array capture: {device_info['name']}",
    )


if __name__ == "__main__":
    main()
