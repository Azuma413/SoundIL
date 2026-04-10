from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.signal import stft
from sklearn.decomposition import NMF as SklearnNMF
from sklearn.exceptions import ConvergenceWarning

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.tasks.sound_camera import SoundConfig
from soundreal_utils import (
    DEFAULT_MIC_CHANNELS,
    REALTIME_NMF_EPS,
    REALTIME_NMF_INIT_MAX_ITER,
    REALTIME_NMF_TOL,
    REALTIME_NMF_WARM_MAX_ITER,
    SPECTROGRAM_NFFT,
)


TORCHNMF_AVAILABLE = importlib.util.find_spec("torchnmf") is not None
if TORCHNMF_AVAILABLE:
    from torchnmf.nmf import NMF as TorchNMF


@dataclass
class BenchmarkResult:
    backend: str
    device: str
    warm_start: bool
    runs: int
    mean_s: float
    p50_s: float
    p95_s: float
    iter_mean: Optional[float]
    iter_last: Optional[int]
    notes: list[str]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def build_fake_beamformed_signals(
    fs: int,
    seconds: float,
    arrays: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    num_samples = int(round(fs * seconds))
    base_freqs = np.array([440.0, 660.0, 880.0], dtype=np.float64)
    signals = []
    t = np.arange(num_samples, dtype=np.float64) / fs
    for array_idx in range(arrays):
        sig = np.zeros(num_samples, dtype=np.float64)
        for freq_idx, freq in enumerate(base_freqs):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amp = 0.03 / (freq_idx + 1) * (1.0 + 0.1 * array_idx)
            sig += amp * np.sin(2.0 * np.pi * freq * t + phase)
        sig += 0.01 * rng.standard_normal(num_samples)
        signals.append(sig.astype(np.float32))
    return signals


def build_concatenated_spec(
    beamformed_signals: list[np.ndarray],
    fs: int,
    nfft: int,
    noverlap: int,
) -> np.ndarray:
    amp_specs = []
    for wav in beamformed_signals:
        _, _, zxx = stft(wav, fs=fs, nperseg=nfft, noverlap=noverlap)
        amp_specs.append(np.abs(zxx))
    return np.concatenate(amp_specs, axis=1)


def make_benchmark_matrix(seed: int, perturb_scale: float) -> np.ndarray:
    config = SoundConfig()
    beamformed_signals = build_fake_beamformed_signals(
        fs=config.fs,
        seconds=config.processing_time,
        arrays=4,
        seed=seed,
    )
    base = build_concatenated_spec(
        beamformed_signals,
        fs=config.fs,
        nfft=SPECTROGRAM_NFFT,
        noverlap=SPECTROGRAM_NFFT // 2,
    ).astype(np.float64, copy=False)
    if perturb_scale <= 0.0:
        return base
    rng = np.random.default_rng(seed + 123)
    perturbed = np.clip(base * (1.0 + perturb_scale * rng.standard_normal(base.shape)), 0.0, None)
    return perturbed


def effective_rank(matrix: np.ndarray, requested_rank: int) -> int:
    return max(1, min(int(requested_rank), int(matrix.shape[0]), int(matrix.shape[1])))


def run_sklearn_benchmark(
    backend_name: str,
    matrix: np.ndarray,
    runs: int,
    warm_start: bool,
    rank: int,
) -> BenchmarkResult:
    times = []
    iterations: list[int] = []
    prev_w = None
    prev_h = None
    init_mode = "nndsvd" if rank > 1 else "random"
    for _ in range(runs):
        model = SklearnNMF(
            n_components=rank,
            init="custom" if warm_start and prev_w is not None and prev_h is not None else init_mode,
            random_state=0,
            max_iter=REALTIME_NMF_WARM_MAX_ITER if warm_start else REALTIME_NMF_INIT_MAX_ITER,
            tol=REALTIME_NMF_TOL,
        )
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if warm_start and prev_w is not None and prev_h is not None:
                w = model.fit_transform(
                    matrix,
                    W=np.maximum(prev_w, REALTIME_NMF_EPS),
                    H=np.maximum(prev_h, REALTIME_NMF_EPS),
                )
            else:
                w = model.fit_transform(matrix)
        elapsed = time.perf_counter() - start
        h = model.components_
        times.append(elapsed)
        iterations.append(int(model.n_iter_))
        prev_w = w
        prev_h = h
    return BenchmarkResult(
        backend=backend_name,
        device="cpu",
        warm_start=warm_start,
        runs=runs,
        mean_s=float(np.mean(times)),
        p50_s=percentile(times, 50.0),
        p95_s=percentile(times, 95.0),
        iter_mean=float(np.mean(iterations)),
        iter_last=iterations[-1],
        notes=[f"shape={matrix.shape}", f"rank={rank}"],
    )


def run_torchnmf_benchmark(
    matrix: np.ndarray,
    runs: int,
    rank: int,
    device: str,
    warm_start: bool,
) -> BenchmarkResult:
    if not TORCHNMF_AVAILABLE:
        raise RuntimeError("torchnmf is not available in this environment.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    dev = torch.device(device)
    target = torch.from_numpy(matrix.astype(np.float32, copy=False)).to(dev)
    times = []
    iterations = []
    notes = [f"shape={matrix.shape}", f"rank={rank}"]

    model = None
    for run_idx in range(runs):
        if model is None or not warm_start:
            model = TorchNMF(target.shape, rank=rank).to(dev)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_iters = model.fit(
                target,
                beta=1,
                tol=REALTIME_NMF_TOL,
                max_iter=REALTIME_NMF_WARM_MAX_ITER if warm_start and run_idx > 0 else REALTIME_NMF_INIT_MAX_ITER,
                verbose=False,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        iterations.append(int(fit_iters) if fit_iters is not None else -1)

    iter_values = [value for value in iterations if value >= 0]
    return BenchmarkResult(
        backend="torchnmf",
        device=device,
        warm_start=warm_start,
        runs=runs,
        mean_s=float(np.mean(times)),
        p50_s=percentile(times, 50.0),
        p95_s=percentile(times, 95.0),
        iter_mean=float(np.mean(iter_values)) if iter_values else None,
        iter_last=iter_values[-1] if iter_values else None,
        notes=notes,
    )


def print_result(result: BenchmarkResult) -> None:
    print(
        "{backend:>20} device={device:<4} warm={warm:<5} mean={mean:.4f}s p50={p50:.4f}s p95={p95:.4f}s iter_mean={iter_mean} iter_last={iter_last}".format(
            backend=result.backend,
            device=result.device,
            warm=str(result.warm_start),
            mean=result.mean_s,
            p50=result.p50_s,
            p95=result.p95_s,
            iter_mean="n/a" if result.iter_mean is None else f"{result.iter_mean:.1f}",
            iter_last="n/a" if result.iter_last is None else result.iter_last,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NMF backend benchmark for soundreal spectrogram sizes")
    parser.add_argument("--runs", type=int, default=5, help="number of timed runs per backend")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--rank", type=int, default=50, help="requested NMF rank")
    parser.add_argument(
        "--perturb-scale",
        type=float,
        default=0.01,
        help="frame-to-frame perturbation scale used to emulate adjacent snapshots",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="all",
        help="all or comma-separated: sklearn_cold,sklearn_warm,torchnmf_cpu_cold,torchnmf_cpu_warm,torchnmf_cuda_cold,torchnmf_cuda_warm",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="optional JSON output path")
    return parser


def resolve_backends(raw: str) -> list[str]:
    all_backends = [
        "sklearn_cold",
        "sklearn_warm",
        "torchnmf_cpu_cold",
        "torchnmf_cpu_warm",
        "torchnmf_cuda_cold",
        "torchnmf_cuda_warm",
    ]
    if raw.strip().lower() == "all":
        return all_backends
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    invalid = [backend for backend in requested if backend not in all_backends]
    if invalid:
        raise ValueError(f"Unknown backends: {invalid}")
    return requested


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    matrix = make_benchmark_matrix(seed=args.seed, perturb_scale=args.perturb_scale)
    rank = effective_rank(matrix, args.rank)
    print(f"matrix_shape={matrix.shape} effective_rank={rank} torch_cuda={torch.cuda.is_available()} torchnmf={TORCHNMF_AVAILABLE}")

    results: list[BenchmarkResult] = []
    for backend in resolve_backends(args.backends):
        if backend == "sklearn_cold":
            result = run_sklearn_benchmark("sklearn", matrix, args.runs, warm_start=False, rank=rank)
        elif backend == "sklearn_warm":
            result = run_sklearn_benchmark("sklearn", matrix, args.runs, warm_start=True, rank=rank)
        elif backend == "torchnmf_cpu_cold":
            result = run_torchnmf_benchmark(matrix, args.runs, rank=rank, device="cpu", warm_start=False)
        elif backend == "torchnmf_cpu_warm":
            result = run_torchnmf_benchmark(matrix, args.runs, rank=rank, device="cpu", warm_start=True)
        elif backend == "torchnmf_cuda_cold":
            result = run_torchnmf_benchmark(matrix, args.runs, rank=rank, device="cuda", warm_start=False)
        elif backend == "torchnmf_cuda_warm":
            result = run_torchnmf_benchmark(matrix, args.runs, rank=rank, device="cuda", warm_start=True)
        else:
            raise AssertionError(f"Unhandled backend: {backend}")
        print_result(result)
        results.append(result)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"saved_json={args.output_json}")


if __name__ == "__main__":
    main()
