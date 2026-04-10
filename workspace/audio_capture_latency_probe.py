from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import select
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 8
DEFAULT_DURATION_S = 8.0
DEFAULT_CHUNK_MS = 20.0
ARECORD_FORMAT = "S24_3LE"
ARECORD_BYTES_PER_SAMPLE = 3
DEFAULT_BACKENDS = (
    "sd_callback",
    "sd_blocking",
    "arecord_basic",
    "arecord_mmap",
    "arecord_lowlat",
)


def query_input_devices() -> list[tuple[int, dict]]:
    return [(idx, device) for idx, device in enumerate(sd.query_devices()) if int(device["max_input_channels"]) > 0]


def extract_hw_index(device_name: str) -> int | None:
    match = re.search(r"\(hw:(\d+),\d+\)", device_name)
    return int(match.group(1)) if match else None


def parse_device_ids(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def select_tamago_devices(explicit_ids: list[int] | None = None) -> list[tuple[int, dict]]:
    inputs = dict(query_input_devices())
    if explicit_ids:
        missing = [device_id for device_id in explicit_ids if device_id not in inputs]
        if missing:
            raise RuntimeError(f"Input device(s) not found: {missing}")
        return [(device_id, inputs[device_id]) for device_id in explicit_ids]

    candidates = [
        (device_id, device)
        for device_id, device in inputs.items()
        if "tamago" in str(device["name"]).lower()
    ]
    if len(candidates) < 4:
        available = "\n".join(
            f"[{idx}] {device['name']} | input_channels={int(device['max_input_channels'])}"
            for idx, device in query_input_devices()
        )
        raise RuntimeError(
            "Could not auto-select four TAMAGO devices. "
            "Pass --device-ids explicitly.\n"
            f"Available input devices:\n{available}"
        )
    return sorted(candidates, key=lambda item: item[0])[:4]


def decode_s24_3le_to_float32(raw_pcm: bytes, channels: int) -> np.ndarray:
    frame_bytes = channels * ARECORD_BYTES_PER_SAMPLE
    if len(raw_pcm) % frame_bytes != 0:
        raise RuntimeError(
            f"Unexpected raw PCM byte count {len(raw_pcm)} for {channels} channels ({frame_bytes} bytes/frame)."
        )
    byte_view = np.frombuffer(raw_pcm, dtype=np.uint8).reshape(-1, 3)
    pcm24 = (
        byte_view[:, 0].astype(np.int32)
        | (byte_view[:, 1].astype(np.int32) << 8)
        | (byte_view[:, 2].astype(np.int32) << 16)
    )
    pcm24 = (pcm24 ^ 0x800000) - 0x800000
    audio = pcm24.astype(np.float32) / 8388608.0
    return audio.reshape(-1, channels)


def percentile_or_none(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def mean_or_none(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


@dataclass
class DeviceSummary:
    device_id: int
    device_name: str
    hw_index: int | None
    chunks: int
    frames: int
    captured_audio_s: float
    first_chunk_delay_s: float | None
    interval_mean_s: float | None
    interval_p95_s: float | None
    reported_latency_mean_s: float | None
    reported_latency_p95_s: float | None
    peak_ch0_mean: float | None
    peak_ch0_max: float | None
    status_count: int
    last_status: str | None
    error: str | None


@dataclass
class BackendSummary:
    backend: str
    success: bool
    samplerate: int
    channels: int
    duration_s: float
    chunk_ms: float
    all_devices_started_s: float | None
    startup_skew_s: float | None
    cycle_spread_mean_s: float | None
    cycle_spread_p95_s: float | None
    devices: list[DeviceSummary]
    notes: list[str]


class DeviceRecorder:
    def __init__(self, device_id: int, device_name: str, hw_index: int | None, samplerate: int):
        self.device_id = device_id
        self.device_name = device_name
        self.hw_index = hw_index
        self.samplerate = samplerate
        self.timestamps: list[float] = []
        self.frames = 0
        self.reported_latencies: list[float] = []
        self.peak_ch0: list[float] = []
        self.statuses: list[str] = []
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def add(
        self,
        timestamp: float,
        frames: int,
        peak_ch0: float,
        reported_latency_s: float | None = None,
        status: str | None = None,
    ) -> None:
        with self._lock:
            self.timestamps.append(timestamp)
            self.frames += int(frames)
            self.peak_ch0.append(float(peak_ch0))
            if reported_latency_s is not None:
                self.reported_latencies.append(float(reported_latency_s))
            if status:
                self.statuses.append(status)

    def set_error(self, error: str) -> None:
        with self._lock:
            self.error = error

    def summarize(self, start_time: float) -> DeviceSummary:
        with self._lock:
            timestamps = list(self.timestamps)
            reported_latencies = list(self.reported_latencies)
            peak_ch0 = list(self.peak_ch0)
            statuses = list(self.statuses)
            error = self.error
            frames = self.frames

        intervals = [curr - prev for prev, curr in zip(timestamps, timestamps[1:])]
        first_chunk_delay_s = None if not timestamps else timestamps[0] - start_time
        return DeviceSummary(
            device_id=self.device_id,
            device_name=self.device_name,
            hw_index=self.hw_index,
            chunks=len(timestamps),
            frames=frames,
            captured_audio_s=frames / max(1, self.samplerate),
            first_chunk_delay_s=first_chunk_delay_s,
            interval_mean_s=mean_or_none(intervals),
            interval_p95_s=percentile_or_none(intervals, 95.0),
            reported_latency_mean_s=mean_or_none(reported_latencies),
            reported_latency_p95_s=percentile_or_none(reported_latencies, 95.0),
            peak_ch0_mean=mean_or_none(peak_ch0),
            peak_ch0_max=max(peak_ch0) if peak_ch0 else None,
            status_count=len(statuses),
            last_status=statuses[-1] if statuses else None,
            error=error,
        )


class ArecordStream:
    def __init__(self, device_id: int, device_name: str, hw_index: int, process: subprocess.Popen[bytes]):
        self.device_id = device_id
        self.device_name = device_name
        self.hw_index = hw_index
        self.process = process
        self.stderr_lines: list[str] = []
        self.stderr_lock = threading.Lock()
        self.stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self.stderr_thread.start()

    def _drain_stderr(self) -> None:
        if self.process.stderr is None:
            return
        try:
            while True:
                line = self.process.stderr.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").strip()
                if decoded:
                    with self.stderr_lock:
                        self.stderr_lines.append(decoded)
                        if len(self.stderr_lines) > 50:
                            self.stderr_lines = self.stderr_lines[-50:]
        except (OSError, ValueError):
            return

    def stderr_text(self) -> str:
        with self.stderr_lock:
            return " | ".join(self.stderr_lines)


def build_arecord_command(
    hw_index: int,
    samplerate: int,
    channels: int,
    chunk_ms: float,
    backend: str,
) -> list[str]:
    command = [
        "arecord",
        "-q",
        "-D",
        f"hw:{hw_index},0",
        "-t",
        "raw",
        "-c",
        str(channels),
        "-f",
        ARECORD_FORMAT,
        "-r",
        str(samplerate),
    ]
    period_us = max(1, int(chunk_ms * 1000.0))
    if backend == "arecord_mmap":
        command.append("-M")
    elif backend == "arecord_lowlat":
        command.extend(
            [
                "-F",
                str(period_us),
                "-B",
                str(period_us * 4),
                "-A",
                str(period_us),
                "-R",
                "0",
            ]
        )
    command.append("--fatal-errors")
    return command


def read_exact_pipe_bytes(stream, num_bytes: int, timeout_s: float) -> bytes:
    deadline = time.monotonic() + timeout_s
    chunks = bytearray()
    fd = stream.fileno()
    while len(chunks) < num_bytes:
        remaining_timeout = deadline - time.monotonic()
        if remaining_timeout <= 0:
            raise TimeoutError(f"Timed out while reading {num_bytes} bytes from arecord pipe.")
        ready, _, _ = select.select([fd], [], [], remaining_timeout)
        if not ready:
            continue
        chunk = os.read(fd, num_bytes - len(chunks))
        if not chunk:
            raise EOFError("arecord stdout closed")
        chunks.extend(chunk)
    return bytes(chunks)


def format_arecord_error(stream: ArecordStream, fallback: str) -> str:
    stderr_text = stream.stderr_text()
    if stderr_text:
        return f"device {stream.device_id}: {stderr_text}"
    return f"device {stream.device_id}: {fallback}"


def stop_arecord_streams(streams: list[ArecordStream]) -> None:
    for stream in streams:
        if stream.process.poll() is None:
            stream.process.terminate()
    for stream in streams:
        if stream.process.poll() is None:
            try:
                stream.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                stream.process.kill()
        for pipe in (stream.process.stdout, stream.process.stderr):
            if pipe is None:
                continue
            try:
                pipe.close()
            except OSError:
                pass
        if stream.stderr_thread.is_alive():
            stream.stderr_thread.join(timeout=0.2)


def run_arecord_backend(
    devices: list[tuple[int, dict]],
    samplerate: int,
    channels: int,
    duration_s: float,
    chunk_ms: float,
    backend: str,
) -> BackendSummary:
    frames_per_chunk = max(1, int(round(samplerate * chunk_ms / 1000.0)))
    chunk_bytes = frames_per_chunk * channels * ARECORD_BYTES_PER_SAMPLE
    timeout_s = 2.0 + (frames_per_chunk / max(1, samplerate))
    recorders = [
        DeviceRecorder(device_id, str(device["name"]), extract_hw_index(str(device["name"])), samplerate)
        for device_id, device in devices
    ]
    streams: list[ArecordStream] = []
    notes = [
        f"chunk_frames={frames_per_chunk}",
        "reported_latency_mean_s is unavailable for arecord backends",
    ]
    cycle_spreads: list[float] = []
    start_time = time.perf_counter()
    success = True
    try:
        for device_id, device in devices:
            hw_index = extract_hw_index(str(device["name"]))
            if hw_index is None:
                raise RuntimeError(f"Could not extract ALSA hw index from device [{device_id}] {device['name']}.")
            process = subprocess.Popen(
                build_arecord_command(hw_index, samplerate, channels, chunk_ms, backend),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            if process.stdout is None or process.stderr is None:
                process.kill()
                raise RuntimeError(f"Failed to open arecord pipes for device [{device_id}] {device['name']}.")
            streams.append(ArecordStream(device_id, str(device["name"]), hw_index, process))

        end_time = start_time + duration_s
        stream_to_recorder = dict(zip(streams, recorders))
        fd_to_stream = {stream.process.stdout.fileno(): stream for stream in streams}
        while time.perf_counter() < end_time:
            cycle_start = time.perf_counter()
            cycle_deadline = cycle_start + timeout_s
            cycle_times = []
            pending_streams = list(streams)
            while pending_streams:
                remaining_timeout = cycle_deadline - time.perf_counter()
                if remaining_timeout <= 0:
                    stream = pending_streams[0]
                    recorder = stream_to_recorder[stream]
                    error = format_arecord_error(
                        stream,
                        f"Timed out waiting for chunk from device {stream.device_id}",
                    )
                    recorder.set_error(error)
                    raise TimeoutError(error)
                ready_fds, _, _ = select.select(
                    [stream.process.stdout.fileno() for stream in pending_streams],
                    [],
                    [],
                    remaining_timeout,
                )
                if not ready_fds:
                    continue
                for fd in ready_fds:
                    stream = fd_to_stream[fd]
                    recorder = stream_to_recorder[stream]
                    if stream.process.poll() is not None:
                        error = format_arecord_error(
                            stream,
                            f"arecord exited with code {stream.process.returncode}",
                        )
                        recorder.set_error(error)
                        raise RuntimeError(error)
                    try:
                        raw_pcm = read_exact_pipe_bytes(stream.process.stdout, chunk_bytes, remaining_timeout)
                    except EOFError:
                        error = format_arecord_error(stream, "arecord stdout closed")
                        recorder.set_error(error)
                        raise RuntimeError(error)
                    except Exception as exc:
                        error = format_arecord_error(stream, str(exc))
                        recorder.set_error(error)
                        raise RuntimeError(error)
                    timestamp = time.perf_counter()
                    audio = decode_s24_3le_to_float32(raw_pcm, channels)
                    recorder.add(
                        timestamp=timestamp,
                        frames=audio.shape[0],
                        peak_ch0=float(np.max(np.abs(audio[:, 0]))) if audio.size > 0 else 0.0,
                    )
                    cycle_times.append(timestamp)
                    pending_streams.remove(stream)
            if cycle_times:
                cycle_spreads.append(max(cycle_times) - min(cycle_times))
    except Exception as exc:
        success = False
        notes.append(f"error={exc}")
    finally:
        stop_arecord_streams(streams)

    device_summaries = [recorder.summarize(start_time) for recorder in recorders]
    first_chunk_delays = [
        summary.first_chunk_delay_s for summary in device_summaries if summary.first_chunk_delay_s is not None
    ]
    return BackendSummary(
        backend=backend,
        success=success and all(summary.chunks > 0 and summary.error is None for summary in device_summaries),
        samplerate=samplerate,
        channels=channels,
        duration_s=duration_s,
        chunk_ms=chunk_ms,
        all_devices_started_s=max(first_chunk_delays) if first_chunk_delays else None,
        startup_skew_s=(max(first_chunk_delays) - min(first_chunk_delays)) if len(first_chunk_delays) >= 2 else None,
        cycle_spread_mean_s=mean_or_none(cycle_spreads),
        cycle_spread_p95_s=percentile_or_none(cycle_spreads, 95.0),
        devices=device_summaries,
        notes=notes,
    )


def run_sounddevice_callback_backend(
    devices: list[tuple[int, dict]],
    samplerate: int,
    channels: int,
    duration_s: float,
    chunk_ms: float,
) -> BackendSummary:
    frames_per_chunk = max(1, int(round(samplerate * chunk_ms / 1000.0)))
    recorders = [
        DeviceRecorder(device_id, str(device["name"]), extract_hw_index(str(device["name"])), samplerate)
        for device_id, device in devices
    ]
    notes = [
        f"chunk_frames={frames_per_chunk}",
        "sounddevice latency request=low",
    ]
    start_time = time.perf_counter()
    success = True
    try:
        with contextlib.ExitStack() as stack:
            for (device_id, _device), recorder in zip(devices, recorders):
                def callback(indata, frames, time_info, status, *, recorder=recorder):
                    try:
                        reported_latency = float(time_info.currentTime - time_info.inputBufferAdcTime)
                    except Exception:
                        reported_latency = None
                    recorder.add(
                        timestamp=time.perf_counter(),
                        frames=frames,
                        peak_ch0=float(np.max(np.abs(indata[:, 0]))) if frames > 0 else 0.0,
                        reported_latency_s=reported_latency,
                        status=str(status) if status else None,
                    )

                stream = sd.InputStream(
                    device=device_id,
                    samplerate=samplerate,
                    channels=channels,
                    dtype="float32",
                    blocksize=frames_per_chunk,
                    latency="low",
                    callback=callback,
                )
                stack.enter_context(stream)
            time.sleep(duration_s)
    except Exception as exc:
        success = False
        notes.append(f"error={exc}")
        for recorder in recorders:
            if recorder.error is None:
                recorder.set_error(str(exc))

    device_summaries = [recorder.summarize(start_time) for recorder in recorders]
    first_chunk_delays = [
        summary.first_chunk_delay_s for summary in device_summaries if summary.first_chunk_delay_s is not None
    ]
    return BackendSummary(
        backend="sd_callback",
        success=success and all(summary.chunks > 0 and summary.error is None for summary in device_summaries),
        samplerate=samplerate,
        channels=channels,
        duration_s=duration_s,
        chunk_ms=chunk_ms,
        all_devices_started_s=max(first_chunk_delays) if first_chunk_delays else None,
        startup_skew_s=(max(first_chunk_delays) - min(first_chunk_delays)) if len(first_chunk_delays) >= 2 else None,
        cycle_spread_mean_s=None,
        cycle_spread_p95_s=None,
        devices=device_summaries,
        notes=notes,
    )


def run_sounddevice_blocking_backend(
    devices: list[tuple[int, dict]],
    samplerate: int,
    channels: int,
    duration_s: float,
    chunk_ms: float,
) -> BackendSummary:
    frames_per_chunk = max(1, int(round(samplerate * chunk_ms / 1000.0)))
    recorders = [
        DeviceRecorder(device_id, str(device["name"]), extract_hw_index(str(device["name"])), samplerate)
        for device_id, device in devices
    ]
    stop_event = threading.Event()
    notes = [
        f"chunk_frames={frames_per_chunk}",
        "sounddevice latency request=low",
    ]
    start_time = time.perf_counter()
    success = True

    def worker(device_id: int, recorder: DeviceRecorder) -> None:
        try:
            with sd.InputStream(
                device=device_id,
                samplerate=samplerate,
                channels=channels,
                dtype="float32",
                blocksize=frames_per_chunk,
                latency="low",
            ) as stream:
                while not stop_event.is_set():
                    audio, overflowed = stream.read(frames_per_chunk)
                    recorder.add(
                        timestamp=time.perf_counter(),
                        frames=audio.shape[0],
                        peak_ch0=float(np.max(np.abs(audio[:, 0]))) if audio.size > 0 else 0.0,
                        reported_latency_s=float(stream.latency) if stream.latency is not None else None,
                        status="overflow" if overflowed else None,
                    )
        except Exception as exc:
            recorder.set_error(str(exc))

    threads = [
        threading.Thread(target=worker, args=(device_id, recorder), daemon=True)
        for (device_id, _device), recorder in zip(devices, recorders)
    ]
    for thread in threads:
        thread.start()
    time.sleep(duration_s)
    stop_event.set()
    for thread in threads:
        thread.join(timeout=1.0)
        if thread.is_alive():
            success = False
            notes.append("warning=some blocking capture threads did not exit cleanly")

    device_summaries = [recorder.summarize(start_time) for recorder in recorders]
    if any(summary.error is not None for summary in device_summaries):
        success = False
        for summary in device_summaries:
            if summary.error:
                notes.append(f"device {summary.device_id} error={summary.error}")
    first_chunk_delays = [
        summary.first_chunk_delay_s for summary in device_summaries if summary.first_chunk_delay_s is not None
    ]
    return BackendSummary(
        backend="sd_blocking",
        success=success and all(summary.chunks > 0 and summary.error is None for summary in device_summaries),
        samplerate=samplerate,
        channels=channels,
        duration_s=duration_s,
        chunk_ms=chunk_ms,
        all_devices_started_s=max(first_chunk_delays) if first_chunk_delays else None,
        startup_skew_s=(max(first_chunk_delays) - min(first_chunk_delays)) if len(first_chunk_delays) >= 2 else None,
        cycle_spread_mean_s=None,
        cycle_spread_p95_s=None,
        devices=device_summaries,
        notes=notes,
    )


def print_device_selection(devices: list[tuple[int, dict]]) -> None:
    print("Using input devices:")
    for device_id, device in devices:
        print(
            f"  [{device_id}] hw:{extract_hw_index(str(device['name']))} | "
            f"input_channels={int(device['max_input_channels'])} | {device['name']}"
        )


def print_backend_summary(summary: BackendSummary) -> None:
    print(f"\n=== {summary.backend} ===")
    print(
        "success={success} all_started={all_started} startup_skew={startup_skew} cycle_spread_mean={cycle_mean} cycle_spread_p95={cycle_p95}".format(
            success=summary.success,
            all_started="n/a" if summary.all_devices_started_s is None else f"{summary.all_devices_started_s:.4f}s",
            startup_skew="n/a" if summary.startup_skew_s is None else f"{summary.startup_skew_s:.4f}s",
            cycle_mean="n/a" if summary.cycle_spread_mean_s is None else f"{summary.cycle_spread_mean_s:.4f}s",
            cycle_p95="n/a" if summary.cycle_spread_p95_s is None else f"{summary.cycle_spread_p95_s:.4f}s",
        )
    )
    for device in summary.devices:
        print(
            "  device=[{device_id}] hw:{hw} chunks={chunks} audio={audio:.3f}s first={first} "
            "interval_mean={imean} interval_p95={ip95} latency_mean={lmean} latency_p95={lp95} "
            "peak_mean={pmean} peak_max={pmax} statuses={statuses} error={error}".format(
                device_id=device.device_id,
                hw="n/a" if device.hw_index is None else device.hw_index,
                chunks=device.chunks,
                audio=device.captured_audio_s,
                first="n/a" if device.first_chunk_delay_s is None else f"{device.first_chunk_delay_s:.4f}s",
                imean="n/a" if device.interval_mean_s is None else f"{device.interval_mean_s:.4f}s",
                ip95="n/a" if device.interval_p95_s is None else f"{device.interval_p95_s:.4f}s",
                lmean="n/a" if device.reported_latency_mean_s is None else f"{device.reported_latency_mean_s:.4f}s",
                lp95="n/a" if device.reported_latency_p95_s is None else f"{device.reported_latency_p95_s:.4f}s",
                pmean="n/a" if device.peak_ch0_mean is None else f"{device.peak_ch0_mean:.5f}",
                pmax="n/a" if device.peak_ch0_max is None else f"{device.peak_ch0_max:.5f}",
                statuses=device.status_count,
                error=device.error or "none",
            )
        )
    for note in summary.notes:
        print(f"  note: {note}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="4台の8chマイクロフォンアレイの入力遅延を比較するベンチマークスクリプト"
    )
    parser.add_argument("--list-devices", action="store_true", help="利用可能な入力デバイスを表示して終了")
    parser.add_argument(
        "--device-ids",
        type=str,
        default=None,
        help="カンマ区切りの入力デバイスID。未指定時は tamago を4台自動選択",
    )
    parser.add_argument("--samplerate", type=int, default=DEFAULT_SAMPLE_RATE, help="サンプリング周波数")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, help="各デバイスの入力チャンネル数")
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S, help="各 backend の計測時間 [s]")
    parser.add_argument("--chunk-ms", type=float, default=DEFAULT_CHUNK_MS, help="チャンク長 [ms]")
    parser.add_argument(
        "--backends",
        type=str,
        default="all",
        help="all または カンマ区切り: " + ",".join(DEFAULT_BACKENDS),
    )
    parser.add_argument("--output-json", type=Path, default=None, help="結果を書き出す JSON ファイル")
    return parser


def resolve_backends(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(DEFAULT_BACKENDS)
    backends = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = [backend for backend in backends if backend not in DEFAULT_BACKENDS]
    if invalid:
        raise ValueError(f"Unknown backends: {invalid}")
    return backends


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_devices:
        print("Available input devices:")
        for device_id, device in query_input_devices():
            print(
                f"  [{device_id}] hw:{extract_hw_index(str(device['name']))} | "
                f"input_channels={int(device['max_input_channels'])} | default_sr={device['default_samplerate']} | {device['name']}"
            )
        return

    devices = select_tamago_devices(parse_device_ids(args.device_ids))
    print_device_selection(devices)

    backends = resolve_backends(args.backends)
    summaries: list[BackendSummary] = []
    for backend in backends:
        print(f"\nRunning backend={backend} for {args.duration_s:.2f}s ...")
        if backend == "sd_callback":
            summary = run_sounddevice_callback_backend(
                devices=devices,
                samplerate=args.samplerate,
                channels=args.channels,
                duration_s=args.duration_s,
                chunk_ms=args.chunk_ms,
            )
        elif backend == "sd_blocking":
            summary = run_sounddevice_blocking_backend(
                devices=devices,
                samplerate=args.samplerate,
                channels=args.channels,
                duration_s=args.duration_s,
                chunk_ms=args.chunk_ms,
            )
        else:
            summary = run_arecord_backend(
                devices=devices,
                samplerate=args.samplerate,
                channels=args.channels,
                duration_s=args.duration_s,
                chunk_ms=args.chunk_ms,
                backend=backend,
            )
        print_backend_summary(summary)
        summaries.append(summary)
        time.sleep(0.5)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps([asdict(summary) for summary in summaries], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved JSON summary to {args.output_json}")


if __name__ == "__main__":
    main()
