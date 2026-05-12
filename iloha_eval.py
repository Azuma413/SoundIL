#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
実機 Iloha で soundReal-m4-f10-s2-p0 の Policy 評価を行うスクリプト。

- 右腕のみ使用
- front / side カメラのみ使用
- 観測 state / action は 7 次元 joint
- 音観測は sound0 / sound1 / spec を 10 FPS のバックグラウンド更新で取得
- ACT / Diffusion のみ対応
"""

import argparse
import asyncio
import time
from pathlib import Path
from typing import Optional

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.robots.iloha import Iloha, IlohaConfig
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from soundreal_utils import (
    CAMERA_FPS,
    LoopingStereoPlayer,
    OBSERVATION_HEIGHT,
    OBSERVATION_WIDTH,
    RIGHT_ARM_DIM,
    RIGHT_ARM_FEATURE_NAMES,
    SOUNDREAL_TASK_NAME,
    RealSoundObservationSource,
    build_soundreal_dataset_features,
    full_action_to_right_feature_dict,
    get_camera_configs,
    make_full_action_from_right,
    preprocess_camera_frame,
    right_array_to_feature_dict,
    sample_sound_episode_condition,
)


CAMERA_MAX_FRAME_AGE_MS = 250
RELATIVE_WARMUP_SECONDS = 3.0
ABSOLUTE_MODE_DELTA_THRESHOLD = 0.2
EVAL_DATASET_PREFIX = f"eval_{SOUNDREAL_TASK_NAME}"
SUPPORTED_POLICY_TYPES = {"act", "diffusion"}


async def reset_robot_to_home(robot: Iloha, init: bool = True) -> None:
    print("ロボットを初期位置に戻しています...")

    home_action = robot.old_action.copy()
    if not init:
        home_action[0] = 0.0
        home_action[7] = 0.0
        home_action[1] = -np.pi / 6
        home_action[2] = -np.pi / 6
        home_action[8] = -np.pi / 6
        home_action[9] = -np.pi / 6
        await robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
        await asyncio.sleep(1.0)

    home_action[3:7] = 0.0
    home_action[10:14] = 0.0
    await robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
    await asyncio.sleep(2.0)

    home_action = np.zeros_like(home_action)
    await robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
    await asyncio.sleep(1.0)

    print("初期位置復帰完了")


def initialize_cameras() -> dict:
    try:
        camera_configs = {
            name: RealSenseCameraConfig(**config_dict)
            for name, config_dict in get_camera_configs(SOUNDREAL_TASK_NAME).items()
        }
        cameras = make_cameras_from_configs(camera_configs)
        for name, camera in cameras.items():
            print(f"{name} を接続中...")
            camera.connect(warmup=True)
            time.sleep(0.5)
        print(f"{len(cameras)}台のカメラを初期化しました")
        return cameras
    except Exception as exc:
        print(f"カメラ初期化エラー: {exc}")
        return {}


def get_next_dataset_number(root: Path, prefix: str) -> int:
    if not root.exists():
        return 0

    existing_nums = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith(f"{prefix}_"):
            try:
                existing_nums.append(int(path.name.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    return max(existing_nums) + 1 if existing_nums else 0


def parse_device_ids(raw: Optional[str]) -> Optional[list[int]]:
    if raw is None or not raw.strip():
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _normalize_feature_spec(feature: dict) -> tuple[str, tuple, Optional[tuple]]:
    return (
        feature["dtype"],
        tuple(feature["shape"]),
        None if feature["names"] is None else tuple(feature["names"]),
    )


def validate_soundreal_dataset_features(features: dict) -> None:
    expected = build_soundreal_dataset_features()
    for key, expected_feature in expected.items():
        if key not in features:
            raise ValueError(f"Dataset is missing required feature: {key}")
        actual = _normalize_feature_spec(features[key])
        expected_norm = _normalize_feature_spec(expected_feature)
        if actual != expected_norm:
            raise ValueError(
                f"Dataset feature mismatch for {key}: expected {expected_norm}, got {actual}"
            )


def capture_soundreal_observation(robot: Iloha, cameras: dict, sound_source: RealSoundObservationSource) -> dict:
    obs = {}
    for name, camera in cameras.items():
        try:
            frame = camera.read_latest(max_age_ms=CAMERA_MAX_FRAME_AGE_MS)
        except Exception:
            frame = camera.async_read(timeout_ms=CAMERA_MAX_FRAME_AGE_MS)
        obs[name] = preprocess_camera_frame(frame)

    obs.update(sound_source.get_latest_images())
    obs.update(full_action_to_right_feature_dict(robot.old_action))
    return obs


def policy_output_to_right_array(action_output) -> np.ndarray:
    if isinstance(action_output, dict):
        if "action" in action_output:
            action_output = action_output["action"]
        elif all(name in action_output for name in RIGHT_ARM_FEATURE_NAMES):
            values = []
            for name in RIGHT_ARM_FEATURE_NAMES:
                value = action_output[name]
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                values.append(float(np.asarray(value).reshape(-1)[0]))
            array = np.asarray(values, dtype=np.float32)
            if array.shape != (RIGHT_ARM_DIM,):
                raise ValueError(f"Unexpected action shape from dict output: {array.shape}")
            return array
        else:
            raise ValueError(f"Unsupported action dict keys: {sorted(action_output.keys())}")

    if hasattr(action_output, "detach"):
        action_output = action_output.detach().cpu().numpy()

    array = np.asarray(action_output, dtype=np.float32).reshape(-1)
    if array.size != RIGHT_ARM_DIM:
        raise ValueError(f"Expected {RIGHT_ARM_DIM}-D action, but got shape {array.shape}")
    return array


def load_local_dataset(dataset_path: Path) -> LeRobotDataset:
    repo_id = f"local/{dataset_path.name}"
    return LeRobotDataset(repo_id=repo_id, root=dataset_path, video_backend="pyav")


async def evaluation_loop(
    robot: Iloha,
    cameras: dict,
    sound_source: RealSoundObservationSource,
    policy,
    preprocessor,
    postprocessor,
    device,
    episode_time_s: float,
    task: str,
    policy_features: dict,
    dataset: Optional[LeRobotDataset] = None,
    display_data: bool = False,
) -> int:
    print(f"評価ループ開始（{episode_time_s}秒間）")

    frame_count = 0
    start_episode_t = time.perf_counter()
    first_action_time: Optional[float] = None

    while True:
        loop_start_t = time.perf_counter()
        elapsed = time.perf_counter() - start_episode_t
        if elapsed >= episode_time_s:
            print(f"エピソード時間（{episode_time_s}秒）に達しました")
            break

        obs = capture_soundreal_observation(robot, cameras, sound_source)
        observation_frame = build_dataset_frame(policy_features, obs, prefix="observation")

        try:
            action_output = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task,
                robot_type="iloha_single_arm",
            )
            right_action = policy_output_to_right_array(action_output)
        except Exception as exc:
            print(f"アクション予測エラー: {exc}")
            raise

        if first_action_time is None:
            first_action_time = time.time()

        current_right_state = np.array(
            [obs[name] for name in RIGHT_ARM_FEATURE_NAMES], dtype=np.float32
        )
        max_delta = float(np.max(np.abs(right_action - current_right_state)))
        elapsed_since_first_action = time.time() - first_action_time
        use_relative = (
            elapsed_since_first_action < RELATIVE_WARMUP_SECONDS
            or max_delta > ABSOLUTE_MODE_DELTA_THRESHOLD
        )

        await robot.async_send_action(
            make_full_action_from_right(right_action),
            use_relative=use_relative,
            use_filter=not use_relative,
        )

        if dataset is not None:
            save_observation_frame = build_dataset_frame(dataset.features, obs, prefix="observation")
            save_action_frame = build_dataset_frame(
                dataset.features,
                right_array_to_feature_dict(right_action),
                prefix="action",
            )
            dataset.add_frame({**save_observation_frame, **save_action_frame, "task": task})

        if display_data:
            log_rerun_data(observation=obs, action=right_array_to_feature_dict(right_action))

        frame_count += 1
        if frame_count % CAMERA_FPS == 0:
            print(f"フレーム: {frame_count}, 経過時間: {elapsed:.1f}秒")

        dt_s = time.perf_counter() - loop_start_t
        sleep_duration = 1.0 / CAMERA_FPS - dt_s
        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)

    print(f"評価ループ終了（合計{frame_count}フレーム）")
    return frame_count


async def main(args) -> None:
    init_logging()

    task = SOUNDREAL_TASK_NAME
    dataset_path = Path(args.dataset_path).resolve()
    policy_path = str(Path(args.policy_path).resolve())

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    dataset_for_stats = load_local_dataset(dataset_path)
    validate_soundreal_dataset_features(dataset_for_stats.features)

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    if policy_cfg.type not in SUPPORTED_POLICY_TYPES:
        raise ValueError(
            f"Unsupported policy type: {policy_cfg.type}. "
            f"Only {sorted(SUPPORTED_POLICY_TYPES)} are supported."
        )

    device = get_safe_torch_device(args.device)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = str(device)
    policy = make_policy(policy_cfg, ds_meta=dataset_for_stats.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=dataset_for_stats.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    robot = None
    cameras = {}
    sound_source = None
    audio_player = None
    dataset = None
    video_encoding_manager = None

    print("=" * 60)
    print("ロボットを初期化中...")
    robot = Iloha(
        IlohaConfig(
            right_dynamixel_port="/dev/ttyUSB_RightDynamixel",
            right_robstride_port="/dev/ttyUSB0",
            left_robstride_port="/dev/ttyUSB1",
            left_dynamixel_port="/dev/ttyUSB_LeftDynamixel",
            max_relative_target_1=0.03,
            max_relative_target_2=0.01,
            max_relative_target_3=0.01,
            max_relative_target_4=0.03,
            max_relative_target_5=0.01,
            max_relative_target_6=0.03,
            current_limit_gripper_R=0.3,
            current_limit_gripper_L=0.3,
        ),
        debug=False,
    )
    try:
        await robot.connect()
        print("ロボット接続完了")
        await reset_robot_to_home(robot)

        print("=" * 60)
        print("カメラを初期化中...")
        cameras = initialize_cameras()
        if not cameras:
            return
        robot.cameras = cameras

        print("=" * 60)
        print("音観測系を初期化中...")
        sound_source = RealSoundObservationSource(
            explicit_device_ids=parse_device_ids(args.input_device_ids),
            observation_height=OBSERVATION_HEIGHT,
            observation_width=OBSERVATION_WIDTH,
        )
        sound_source.start()
        if not sound_source.wait_until_ready(timeout_s=5.0):
            print("[soundreal] Audio buffers are still warming up. Initial frames may contain zeros.")
        audio_player = LoopingStereoPlayer(output_device=args.output_device)

        if args.save_data:
            print("=" * 60)
            print("評価データセット作成中...")
            output_root = Path(args.output_root)
            output_root.mkdir(parents=True, exist_ok=True)
            dataset_num = get_next_dataset_number(output_root, EVAL_DATASET_PREFIX)
            dataset_name = f"{EVAL_DATASET_PREFIX}_{dataset_num}"
            repo_id = f"local/{dataset_name}"
            save_path = output_root / dataset_name
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=CAMERA_FPS,
                root=save_path,
                robot_type="iloha_single_arm",
                features=build_soundreal_dataset_features(),
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=5,
                video_backend="pyav",
            )
            video_encoding_manager = VideoEncodingManager(dataset)
            video_encoding_manager.__enter__()
            print(f"データセット作成完了: {repo_id}")

        if args.display_data:
            init_rerun(session_name="iloha_soundreal_eval")

        rng = np.random.default_rng(args.seed)

        print("=" * 60)
        print(f"{args.num_episodes}エピソードの評価を開始します")
        for episode_idx in range(args.num_episodes):
            print(f"\n--- エピソード {episode_idx + 1}/{args.num_episodes} ---")

            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            condition = sample_sound_episode_condition(rng)
            print(
                "[soundreal] Episode stimulus: "
                f"speaker={condition.speaker}, sound={condition.sound_label}"
            )
            audio_player.start(condition)
            await asyncio.sleep(args.audio_preroll_s)

            frame_count = await evaluation_loop(
                robot=robot,
                cameras=cameras,
                sound_source=sound_source,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                episode_time_s=args.episode_time_s,
                task=task,
                policy_features=dataset_for_stats.features,
                dataset=dataset,
                display_data=args.display_data,
            )

            audio_player.stop()

            if dataset is not None:
                if frame_count > 0:
                    dataset.save_episode()
                    print(f"エピソード {episode_idx + 1} を保存しました")
                else:
                    dataset.clear_episode_buffer()
                    print(f"エピソード {episode_idx + 1} は空だったため保存をスキップしました")

            if episode_idx < args.num_episodes - 1:
                await reset_robot_to_home(robot, init=False)
                await asyncio.sleep(2.0)

    finally:
        print("=" * 60)
        print("クリーンアップ中...")

        if audio_player is not None:
            audio_player.stop()
        if sound_source is not None:
            sound_source.stop()

        if video_encoding_manager is not None:
            video_encoding_manager.__exit__(None, None, None)
        elif dataset is not None:
            dataset.finalize()

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
                print(f"初期位置復帰エラー: {exc}")
            try:
                await robot.disconnect()
                print("ロボット切断完了")
            except Exception as exc:
                print(f"ロボット切断エラー: {exc}")
        print("=" * 60)
        print("評価スクリプト終了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="soundReal 実機 Iloha Policy 評価")
    parser.add_argument("--policy_path", type=str, required=True, help="学習済み policy のパス")
    parser.add_argument("--dataset_path", type=str, required=True, help="正規化統計を使う dataset のパス")
    parser.add_argument("--output_root", type=str, default="datasets", help="保存先ルート")
    parser.add_argument("--save_data", action="store_true", help="評価時の観測と action を保存する")
    parser.add_argument("--episode_time_s", type=float, default=60.0, help="1 エピソードの長さ（秒）")
    parser.add_argument("--num_episodes", type=int, default=1, help="評価エピソード数")
    parser.add_argument("--display_data", action="store_true", help="rerun で可視化する")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="推論デバイス")
    parser.add_argument("--seed", type=int, default=0, help="音条件サンプリング用 seed")
    parser.add_argument(
        "--audio_preroll_s",
        type=float,
        default=0.5,
        help="音再生開始から rollout 開始までの待機時間（秒）",
    )
    parser.add_argument(
        "--output_device",
        type=int,
        default=None,
        help="sounddevice の出力デバイス ID。未指定時はデフォルト出力を使う",
    )
    parser.add_argument(
        "--input_device_ids",
        type=str,
        default=None,
        help="TAMAGO 入力デバイス ID をカンマ区切りで指定。未指定時は自動検出",
    )
    asyncio.run(main(parser.parse_args()))
