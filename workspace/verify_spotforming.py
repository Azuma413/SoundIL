"""Spotformingの有意性を信号レベルで検証するスクリプト。

学習を介さず、3種類のスペクトログラム生成方式
  mode 0: Spotforming (DSビームフォーミング + NMF)
  mode 1: 単一マイク
  mode 2: 全マイク単純平均
が生成する観測画像から、タスク音(A/B)の識別がどこまで可能かを、
干渉の種類と強度を振って比較する。

干渉条件:
  opposite    : 遠方の妨害音源が「逆側のタスク音」を再生（方向性・タスク混同性あり）
  white_point : 遠方の点音源がホワイトノイズを再生（方向性あり）
  sensor      : 各マイクに独立なホワイトノイズを付加（無指向性・対照条件）

音場の線形性を利用し、ターゲットと妨害音源を個別にシミュレーションして
受音RMS比 alpha (干渉RMS / ターゲットRMS, マイク受音端で定義) を正確に制御する。
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from pydub import AudioSegment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyroomacoustics as pra

from env.tasks.sound_camera import SoundCamera, SoundConfig

SOUND_A_PATH = "sounds/0.wav"  # soundDiff: 音A → 右の箱
SOUND_B_PATH = "sounds/3.wav"  # soundDiff: 音B → 左の箱
ROOM_DIM = [10.0, 10.0, 3.0]
WORKSPACE_CENTER = np.array([5.0, 5.0])
MODES = {0: "spotforming", 1: "single_mic", 2: "mic_average", 3: "beamform_only"}


class _DummyTarget:
    def get_pos(self):
        return np.zeros(3, dtype=np.float32)


def load_audio(path: str, fs: int) -> np.ndarray:
    sound = AudioSegment.from_file(path).set_frame_rate(fs).set_channels(1)
    sig = np.array(sound.get_array_of_samples()).astype(np.float32)
    peak = np.abs(sig).max()
    return sig / peak if peak > 0 else sig


def random_chunk(sig: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    start = int(rng.integers(0, len(sig)))
    idx = (start + np.arange(n)) % len(sig)
    return sig[idx].astype(np.float32)


def make_camera(height: int) -> SoundCamera:
    # soundDiff-m4-f10-s2-p0 相当 (use_legacy_sound_config=True)
    config = SoundConfig(
        observation_height=height,
        observation_width=height,
        mic_array_num=4,
        use_spectrogram=True,
        use_soundmap=True,
        spectrogram_display_min_hz=0.0,
        spectrogram_display_max_hz=None,
        spectrogram_normalization="minmax",
    )
    return SoundCamera(target=_DummyTarget(), config=config)


def sample_target_pos(rng: np.random.Generator) -> np.ndarray:
    # SoundTaskのcube配置範囲 + render()内のオフセット [4.5, 5.0, 0.0]
    x = rng.uniform(0.3, 0.7) + 4.5
    y = rng.uniform(-0.3, 0.3) + 5.0
    return np.array([x, y, 0.05])


def sample_distractor_pos(rng: np.random.Generator, radius: float) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi)
    x = WORKSPACE_CENTER[0] + radius * np.cos(theta)
    y = WORKSPACE_CENTER[1] + radius * np.sin(theta)
    return np.array([x, y, 0.1])


def simulate_mics(cam: SoundCamera, source_pos: np.ndarray, signal: np.ndarray):
    """1音源のみの部屋をシミュレーションし、アレイごとのマイク信号リストを返す"""
    cfg = cam.config
    room = pra.ShoeBox(ROOM_DIM, fs=cfg.fs, max_order=cfg.room_max_order)
    room.add_source(source_pos, signal=signal)
    for i in range(cfg.mic_array_num):
        room.add_microphone_array(
            cam._generate_circular_array(
                cam.mic_positions[i], cfg.mics_per_array, cfg.mic_radius
            )
        )
    room.simulate()
    sigs = room.mic_array.signals
    return [
        sigs[i * cfg.mics_per_array : (i + 1) * cfg.mics_per_array]
        for i in range(cfg.mic_array_num)
    ]


def global_rms(mic_signals_list) -> float:
    return float(np.sqrt(np.mean(np.concatenate([s.ravel() for s in mic_signals_list]) ** 2)))


def pad_to(mic_signals_list, n: int):
    out = []
    for s in mic_signals_list:
        if s.shape[1] < n:
            s = np.pad(s, ((0, 0), (0, n - s.shape[1])))
        out.append(s[:, :n])
    return out


def combine(target_mics, interference_mics, alpha: float):
    """受音RMS比 alpha でターゲットと干渉を重ね合わせる"""
    if alpha <= 0.0 or interference_mics is None:
        return [s.copy() for s in target_mics]
    n = max(target_mics[0].shape[1], interference_mics[0].shape[1])
    t = pad_to(target_mics, n)
    d = pad_to(interference_mics, n)
    scale = alpha * global_rms(t) / max(global_rms(d), 1e-12)
    return [ts + scale * ds for ts, ds in zip(t, d)]


def add_sensor_noise(target_mics, alpha: float, rng: np.random.Generator):
    """各マイクに独立なホワイトノイズを付加（受音RMS比 alpha）"""
    rms_t = global_rms(target_mics)
    return [
        s + rng.standard_normal(s.shape).astype(np.float32) * alpha * rms_t
        for s in target_mics
    ]


WORKSPACE_COORDS = None  # (x_coords, y_coords) キャッシュ


def workspace_coords(cam: SoundCamera):
    """render()の局所マップと同じ領域(0.6m四方)を部屋座標系で表すグリッド"""
    global WORKSPACE_COORDS
    if WORKSPACE_COORDS is None:
        half = cam.config.mic_array_radius  # 0.3m
        x = np.linspace(5.0 - half, 5.0 + half, cam.config.observation_width)
        y = np.linspace(5.0 - half, 5.0 + half, cam.config.observation_height)
        WORKSPACE_COORDS = (x, y)
    return WORKSPACE_COORDS


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def linear_sir_gain(cam, top_peaks, target_mics, interference_mics):
    """ピークに向けたDSビームフォーミング(+アレイ平均)の線形SIR改善率を計測"""
    if not top_peaks or interference_mics is None:
        return None
    peak_x, peak_y = top_peaks[0]
    n = max(target_mics[0].shape[1], interference_mics[0].shape[1])
    t_all, d_all = pad_to(target_mics, n), pad_to(interference_mics, n)
    beams_t, beams_d = [], []
    for i in range(len(t_all)):
        theta_deg = cam._pixel_to_azimuth(peak_x, peak_y, cam.mic_positions[i])
        abs_pos = cam._generate_circular_array(
            cam.mic_positions[i], cam.config.mics_per_array, cam.config.mic_radius
        ).T
        rel = abs_pos - np.mean(abs_pos, axis=0, keepdims=True)
        beams_t.append(cam._ds_beamform(t_all[i], rel, theta_deg))
        beams_d.append(cam._ds_beamform(d_all[i], rel, theta_deg))
    sir_in = rms(np.concatenate([s.ravel() for s in t_all])) / max(
        rms(np.concatenate([s.ravel() for s in d_all])), 1e-12
    )
    sir_out = rms(np.mean(beams_t, axis=0)) / max(rms(np.mean(beams_d, axis=0)), 1e-12)
    return sir_out / max(sir_in, 1e-12)


def compute_specs(
    cam: SoundCamera,
    mic_signals_list,
    modes=(0, 3, 1, 2),
    pipeline="realcoords",
    target_mics=None,
    interference_mics=None,
):
    """同一のマイク信号から各modeのスペクトログラム画像(ch0, float32)を生成

    pipeline="render"    : render()と同一経路（ピーク座標系の不整合バグ込み）
    pipeline="realcoords": 実機パイプラインと同じ、部屋座標系で一貫した経路
    戻り値: (specs, info)  infoにはmode0のSIRゲイン等の診断値
    """
    specs = {}
    info = {}
    music_results = None
    top_peaks = None

    def get_top_peaks():
        nonlocal music_results, top_peaks
        if top_peaks is not None:
            return top_peaks
        if music_results is None:
            music_results = [
                cam.estimate_music(mic_signals_list[i], i)
                for i in range(len(mic_signals_list))
            ]
        if pipeline == "render":
            maps = [
                cam._generate_soundmap_from_doa(music_results[i], cam.mic_positions[i])
                for i in range(len(music_results))
            ]
            combined = cam.build_combined_sound_map(maps)
            top_peaks = cam.compute_top_peaks(combined)
        else:
            xc, yc = workspace_coords(cam)
            maps = [
                cam._generate_soundmap_from_doa(
                    music_results[i], cam.mic_positions[i], x_coords=xc, y_coords=yc
                )
                for i in range(len(music_results))
            ]
            combined = cam.build_combined_sound_map(maps)
            # local_maxモードはx_coordsを無視してroom_size=10のピクセル座標系に
            # 落ちるため、num_peaks=1と等価なargmaxモードで部屋座標のピークを得る
            top_peaks = cam.compute_top_peaks(
                combined, x_coords=xc, y_coords=yc, selection_mode="argmax"
            )
        if top_peaks:
            gain = linear_sir_gain(cam, top_peaks, target_mics, interference_mics)
            if gain is not None:
                info["sir_gain"] = round(gain, 4)
            info["peak"] = [round(float(v), 3) for v in top_peaks[0]]
        return top_peaks

    for mode in modes:
        cam.config.spectrogram_mode = mode
        if mode == 0:
            peaks = get_top_peaks()
            cam.reset_nmf_state()
            rec = cam.reconstruct_primary_peak(mic_signals_list, peaks)
            img = None if rec is None else cam.create_spectrogram_image_from_audio(rec)
        elif mode == 3:
            # 操舵DSビームフォーミング + アレイ平均のみ（NMFマスクなし）
            peaks = get_top_peaks()
            if not peaks:
                img = None
            else:
                peak_x, peak_y = peaks[0]
                beams = []
                for i, sigs in enumerate(mic_signals_list):
                    theta_deg = cam._pixel_to_azimuth(peak_x, peak_y, cam.mic_positions[i])
                    abs_pos = cam._generate_circular_array(
                        cam.mic_positions[i], cam.config.mics_per_array, cam.config.mic_radius
                    ).T
                    rel = abs_pos - np.mean(abs_pos, axis=0, keepdims=True)
                    beams.append(cam._ds_beamform(sigs, rel, theta_deg))
                n = min(len(b) for b in beams)
                audio = np.mean([b[:n] for b in beams], axis=0)[: cam.required_length]
                img = cam.create_spectrogram_image_from_audio(audio)
        else:
            img = cam._generate_spectrogram([], mic_signals_list)
        specs[mode] = None if img is None else img[:, :, 0].astype(np.float32)
    return specs, info


def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64).ravel()
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(zscore(a) * zscore(b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="workspace/spotforming_verify")
    parser.add_argument("--n-trials", type=int, default=12, help="テスト試行数 / (音種)")
    parser.add_argument("--n-refs", type=int, default=6, help="参照試行数 / (音種)")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument(
        "--conditions", nargs="+", default=["opposite", "white_point", "sensor"]
    )
    parser.add_argument("--distractor-radius", type=float, default=2.0)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument(
        "--pipeline",
        choices=["render", "realcoords"],
        default="realcoords",
        help="render: render()と同一経路（座標系バグ込み） / realcoords: 実機相当の一貫した座標系",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="1試行のみで動作・時間確認")
    parser.add_argument("--nmf-max-iter", type=int, default=None,
                        help="NMFの反復数上限を上書き（リアルタイム用軽量設定の影響を切り分ける）")
    parser.add_argument("--nmf-tol", type=float, default=None)
    args = parser.parse_args()

    if args.nmf_max_iter is not None:
        import env.tasks.sound_camera as sc_module
        sc_module.REALTIME_NMF_INIT_MAX_ITER = args.nmf_max_iter
        sc_module.REALTIME_NMF_WARM_MAX_ITER = args.nmf_max_iter
    if args.nmf_tol is not None:
        import env.tasks.sound_camera as sc_module
        sc_module.REALTIME_NMF_TOL = args.nmf_tol

    if args.smoke:
        args.n_trials, args.n_refs, args.alphas = 1, 1, [1.0]

    os.makedirs(args.out_dir, exist_ok=True)
    sample_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cam = make_camera(args.height)
    fs = cam.config.fs
    n_samples = cam.required_length

    audio = {"A": load_audio(SOUND_A_PATH, fs), "B": load_audio(SOUND_B_PATH, fs)}

    t0 = time.time()

    # 全スペクトログラムを112x112 float16で保存（オフラインで読み出し方式を差し替え可能に）
    spec_store = []

    def store_spec(spec):
        if spec is None:
            return -1
        small = cv2.resize(spec, (112, 112), interpolation=cv2.INTER_AREA)
        spec_store.append(small.astype(np.float16))
        return len(spec_store) - 1

    # ---- 参照スペクトログラム（クリーン条件のセントロイド） ----
    print("[refs] building clean references...", flush=True)
    refs = {mode: {} for mode in MODES}
    ref_meta = []
    for sound in ["A", "B"]:
        acc = {mode: [] for mode in MODES}
        for k in range(args.n_refs):
            pos = sample_target_pos(rng)
            sig = random_chunk(audio[sound], n_samples, rng)
            mics = simulate_mics(cam, pos, sig)
            specs, _ = compute_specs(cam, mics, pipeline=args.pipeline)
            for mode, spec in specs.items():
                if spec is not None:
                    acc[mode].append(zscore(spec))
                ref_meta.append(
                    {"sound": sound, "mode": mode, "mode_name": MODES[mode], "spec_idx": store_spec(spec)}
                )
            print(f"  ref {sound} {k + 1}/{args.n_refs} ({time.time() - t0:.1f}s)", flush=True)
        for mode in MODES:
            refs[mode][sound] = np.mean(acc[mode], axis=0)

    # ---- テスト試行 ----
    records = []
    for trial in range(args.n_trials * 2):
        sound = "A" if trial % 2 == 0 else "B"
        other = "B" if sound == "A" else "A"
        target_pos = sample_target_pos(rng)
        distractor_pos = sample_distractor_pos(rng, args.distractor_radius)
        target_sig = random_chunk(audio[sound], n_samples, rng)

        target_mics = simulate_mics(cam, target_pos, target_sig)

        interference_mics = {}
        if "opposite" in args.conditions:
            opp_sig = random_chunk(audio[other], n_samples, rng)
            interference_mics["opposite"] = simulate_mics(cam, distractor_pos, opp_sig)
        if "white_point" in args.conditions:
            wn = rng.standard_normal(n_samples).astype(np.float32)
            interference_mics["white_point"] = simulate_mics(cam, distractor_pos, wn)

        # クリーンベースライン (alpha=0)
        combos = [("clean", 0.0, target_mics)]
        for cond in args.conditions:
            for alpha in args.alphas:
                if cond == "sensor":
                    mixed = add_sensor_noise(target_mics, alpha, rng)
                else:
                    mixed = combine(target_mics, interference_mics[cond], alpha)
                combos.append((cond, alpha, mixed))

        for cond, alpha, mixed in combos:
            specs, info = compute_specs(
                cam,
                mixed,
                pipeline=args.pipeline,
                target_mics=target_mics,
                interference_mics=interference_mics.get(cond),
            )
            for mode, spec in specs.items():
                rec = {
                    "trial": trial,
                    "condition": cond,
                    "alpha": alpha,
                    "mode": mode,
                    "mode_name": MODES[mode],
                    "true_sound": sound,
                    "target_pos": [round(float(v), 3) for v in target_pos],
                    "distractor_pos": [round(float(v), 3) for v in distractor_pos],
                }
                rec["spec_idx"] = store_spec(spec)
                if mode == 0:
                    rec.update(info)
                if spec is None:
                    rec.update({"pred": None, "corr_A": None, "corr_B": None})
                else:
                    corr_a = correlation(spec, refs[mode]["A"])
                    corr_b = correlation(spec, refs[mode]["B"])
                    rec.update(
                        {
                            "pred": "A" if corr_a >= corr_b else "B",
                            "corr_A": round(corr_a, 4),
                            "corr_B": round(corr_b, 4),
                        }
                    )
                    if trial < 2:
                        np.save(
                            os.path.join(
                                sample_dir,
                                f"{cond}_a{alpha:g}_{MODES[mode]}_{sound}_t{trial}.npy",
                            ),
                            spec.astype(np.float32),
                        )
                records.append(rec)
        print(
            f"[trial {trial + 1}/{args.n_trials * 2}] sound={sound} "
            f"({time.time() - t0:.1f}s elapsed)",
            flush=True,
        )
        # 逐次保存（中断しても途中結果が残るように）
        with open(os.path.join(args.out_dir, "records.json"), "w") as f:
            json.dump(
                {"args": vars(args), "ref_meta": ref_meta, "records": records}, f, indent=1
            )

    np.save(os.path.join(args.out_dir, "specs.npy"), np.stack(spec_store, axis=0))
    print(f"done in {time.time() - t0:.1f}s -> {args.out_dir}/records.json", flush=True)


if __name__ == "__main__":
    main()
