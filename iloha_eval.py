#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np
from serial.tools import list_ports

from iloha_calibration import (
    DEFAULT_CALIBRATION_PATH,
    dataset_to_hardware_action,
    hardware_to_dataset_state,
    load_joint_offsets,
    offsets_summary,
)
from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.robots.iloha import Iloha, IlohaConfig
from lerobot.robots.iloha.iloha_controller.robstride.src.robstride import (
    RobStride,
    RobStrideController,
)
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
    SOUND_FILES,
    SOUND_LABELS,
    SOUNDREAL_TASK_NAME,
    RealSoundObservationSource,
    SoundEpisodeCondition,
    build_soundreal_dataset_features,
    full_action_to_right_feature_dict,
    get_camera_configs,
    make_full_action_from_right,
    preprocess_soundreal_camera_frame,
    right_array_to_feature_dict,
)


CAMERA_MAX_FRAME_AGE_MS = 80
RELATIVE_WARMUP_SECONDS = 3.0
ABSOLUTE_MODE_DELTA_THRESHOLD = 0.2
EVAL_DATASET_PREFIX = f"eval_{SOUNDREAL_TASK_NAME}"
SUPPORTED_POLICY_TYPES = {"act", "diffusion"}
SOUND_SPEAKER_CHOICES = ("left", "right")


def _resolved_device_path(device: str) -> Path:
    return Path(device).resolve(strict=False)


async def detect_robstride_port(
    motor_id: int,
    excluded_ports: set[Path],
) -> str:
    candidates = sorted(
        port.device
        for port in list_ports.comports()
        if _resolved_device_path(port.device) not in excluded_ports
    )
    if not candidates:
        raise RuntimeError("RobStride候補のシリアルポートが見つかりません")

    for candidate in candidates:
        probe = RobStrideController(
            port=candidate,
            motors=[RobStride(id=motor_id, offset=0.0)],
        )
        try:
            if await probe.connect():
                return candidate
        finally:
            await probe.disconnect()

    raise RuntimeError(
        f"RobStride motor ID {motor_id} に応答するシリアルポートが"
        f"見つかりませんでした（候補: {candidates}）"
    )


async def resolve_auto_robstride_ports(config: IlohaConfig) -> None:
    excluded_ports = {
        _resolved_device_path(config.right_dynamixel_port),
        _resolved_device_path(config.left_dynamixel_port),
    }

    arm_specs = (
        ("right", config.enable_right_arm, 4),
        ("left", config.enable_left_arm, 1),
    )
    for arm, enabled, probe_motor_id in arm_specs:
        if not enabled:
            continue

        attribute = f"{arm}_robstride_port"
        configured_port = getattr(config, attribute)
        if configured_port != "auto":
            excluded_ports.add(_resolved_device_path(configured_port))
            continue

        detected_port = await detect_robstride_port(probe_motor_id, excluded_ports)
        setattr(config, attribute, detected_port)
        excluded_ports.add(_resolved_device_path(detected_port))
        print(f"{arm} RobStrideポートを自動検出しました: {detected_port}")


async def reset_robot_to_home(
    robot: Iloha,
    init: bool = True,
    right_joint_offsets: Optional[np.ndarray] = None,
) -> None:
    print("ロボットを初期位置に戻しています...")

    offsets = (
        np.zeros(RIGHT_ARM_DIM, dtype=np.float32)
        if right_joint_offsets is None
        else np.asarray(right_joint_offsets, dtype=np.float32)
    )
    if offsets.shape != (RIGHT_ARM_DIM,):
        raise ValueError(f"Expected {RIGHT_ARM_DIM} right joint offsets, got {offsets.shape}")
    calibrated_home = make_full_action_from_right(offsets)

    # Keep the current RobStride targets for the first command.  In particular,
    # replacing a negative motor target with zero immediately can make a
    # multi-turn motor take the long route to the origin on startup.
    home_action = np.asarray(robot.old_action, dtype=np.float32).copy()
    home_action[3:7] = calibrated_home[3:7]
    home_action[10:14] = calibrated_home[10:14]
    await robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
    await asyncio.sleep(2.0)

    home_action = calibrated_home
    await robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
    await asyncio.sleep(1.0)

    print("初期位置復帰完了")


def make_eval_sound_episode_condition(
    rng: np.random.Generator,
    sound_index: Optional[int] = None,
    speaker: Optional[str] = None,
) -> SoundEpisodeCondition:
    if sound_index is None:
        sound_index = int(rng.integers(0, len(SOUND_FILES)))
    if sound_index not in SOUND_FILES:
        raise ValueError(
            f"Invalid sound_index={sound_index}. "
            f"Available sound indices: {sorted(SOUND_FILES.keys())}"
        )

    if speaker is None:
        speaker = SOUND_SPEAKER_CHOICES[int(rng.integers(0, len(SOUND_SPEAKER_CHOICES)))]
    if speaker not in SOUND_SPEAKER_CHOICES:
        raise ValueError(
            f"Invalid speaker={speaker!r}. "
            f"Available speakers: {list(SOUND_SPEAKER_CHOICES)}"
        )

    return SoundEpisodeCondition(
        sound_index=sound_index,
        sound_label=SOUND_LABELS[sound_index],
        sound_path=SOUND_FILES[sound_index],
        speaker=speaker,
    )


def append_sound_condition_csv(
    dataset_path: Path,
    episode_index: int,
    condition: SoundEpisodeCondition,
    frame_count: int,
) -> None:
    csv_path = dataset_path / "sound_conditions.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode_index",
                "sound_index",
                "sound_label",
                "sound_path",
                "speaker",
                "frame_count",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "episode_index": episode_index,
                "sound_index": condition.sound_index,
                "sound_label": condition.sound_label,
                "sound_path": str(condition.sound_path),
                "speaker": condition.speaker,
                "frame_count": frame_count,
            }
        )


def prompt_episode_success(episode_number: int, total_episodes: int) -> bool:
    while True:
        result = input(
            f"エピソード {episode_number}/{total_episodes} の結果を入力してください "
            "(1: 成功, 2: 失敗): "
        ).strip()
        if result == "1":
            return True
        if result == "2":
            return False
        print("無効な入力です。1（成功）または 2（失敗）を入力してください。")


def append_episode_success_csv(dataset_path: Path, result: dict) -> Path:
    csv_path = dataset_path / "episode_success.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "episode_number",
                "episode_index",
                "success",
                "frame_count",
                "sound_index",
                "sound_label",
                "speaker",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    return csv_path


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


def is_sound_observation_feature(key: str) -> bool:
    return key in {
        "observation.images.sound0",
        "observation.images.sound1",
        "observation.images.spec",
    }


def validate_soundreal_dataset_features(features: dict, required_keys: Optional[set[str]] = None) -> None:
    required_keys = required_keys or set()
    expected = build_soundreal_dataset_features()
    for key, expected_feature in expected.items():
        if key not in features:
            if is_sound_observation_feature(key) and key not in required_keys:
                continue
            raise ValueError(f"Dataset is missing required feature: {key}")
        actual = _normalize_feature_spec(features[key])
        expected_norm = _normalize_feature_spec(expected_feature)
        if actual != expected_norm:
            raise ValueError(
                f"Dataset feature mismatch for {key}: expected {expected_norm}, got {actual}"
            )


def validate_policy_dataset_features(features: dict, policy_cfg: PreTrainedConfig) -> None:
    required_keys = set(policy_cfg.input_features) | set(policy_cfg.output_features)
    missing_keys = sorted(key for key in required_keys if key not in features)
    if missing_keys:
        raise ValueError(
            "Dataset is missing feature(s) required by the policy: "
            + ", ".join(missing_keys)
        )


def build_eval_dataset_features(reference_features: dict) -> dict:
    expected = build_soundreal_dataset_features()
    return {
        key: expected_feature
        for key, expected_feature in expected.items()
        if key in reference_features
    }


def requires_sound_observations(features: dict) -> bool:
    return any(is_sound_observation_feature(key) for key in features)


def capture_soundreal_observation(
    robot: Iloha,
    cameras: dict,
    sound_source: Optional[RealSoundObservationSource],
    right_joint_offsets: Optional[np.ndarray] = None,
) -> dict:
    obs = {}
    for name, camera in cameras.items():
        try:
            frame = camera.read_latest(max_age_ms=CAMERA_MAX_FRAME_AGE_MS)
        except Exception:
            frame = camera.async_read(timeout_ms=CAMERA_MAX_FRAME_AGE_MS)
        obs[name] = preprocess_soundreal_camera_frame(name, frame)

    if sound_source is not None:
        obs.update(sound_source.get_latest_images())
    if right_joint_offsets is None:
        obs.update(full_action_to_right_feature_dict(robot.old_action))
    else:
        dataset_state = hardware_to_dataset_state(
            np.asarray(robot.old_action, dtype=np.float32)[7:14],
            right_joint_offsets,
        )
        obs.update(right_array_to_feature_dict(dataset_state))
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


def override_act_eval_config(policy) -> None:
    policy.config.n_action_steps = 30
    # policy.config.temporal_ensemble_coeff = 0.1 # 0.01
    # policy.temporal_ensembler = ACTTemporalEnsembler(
    #     policy.config.temporal_ensemble_coeff,
    #     policy.config.chunk_size,
    # )
    policy.reset()
    print(
        "Overriding ACT eval config: "
        f"n_action_steps={policy.config.n_action_steps}, "
        f"temporal_ensemble_coeff={policy.config.temporal_ensemble_coeff}"
    )


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
    right_joint_offsets: np.ndarray,
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

        obs = capture_soundreal_observation(
            robot,
            cameras,
            sound_source,
            right_joint_offsets=right_joint_offsets,
        )
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

        hardware_right_action = dataset_to_hardware_action(
            right_action,
            right_joint_offsets,
        )
        await robot.async_send_action(
            make_full_action_from_right(hardware_right_action),
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
    calibration_path = Path(args.calibration_path).resolve() if args.calibration_path else None
    right_joint_offsets = load_joint_offsets(
        calibration_path,
        required=args.require_calibration,
    )
    if calibration_path is not None and calibration_path.is_file():
        print(f"キャリブレーションを読み込みました: {calibration_path}")
        print(offsets_summary(right_joint_offsets))
    else:
        print("キャリブレーションファイルなし: ゼロオフセットで評価します")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    dataset_for_stats = load_local_dataset(dataset_path)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    if policy_cfg.type not in SUPPORTED_POLICY_TYPES:
        raise ValueError(
            f"Unsupported policy type: {policy_cfg.type}. "
            f"Only {sorted(SUPPORTED_POLICY_TYPES)} are supported."
        )
    required_feature_keys = set(policy_cfg.input_features) | set(policy_cfg.output_features)
    validate_soundreal_dataset_features(dataset_for_stats.features, required_feature_keys)
    validate_policy_dataset_features(dataset_for_stats.features, policy_cfg)
    eval_dataset_features = build_eval_dataset_features(dataset_for_stats.features)
    needs_sound_observations = (
        requires_sound_observations(eval_dataset_features)
        or requires_sound_observations(policy_cfg.input_features)
    )

    device = get_safe_torch_device(args.device)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = str(device)
    policy = make_policy(policy_cfg, ds_meta=dataset_for_stats.meta)
    if policy_cfg.type == "act":
        override_act_eval_config(policy)
        policy_cfg.n_action_steps = policy.config.n_action_steps
        policy_cfg.temporal_ensemble_coeff = policy.config.temporal_ensemble_coeff
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
    dataset_save_path = None
    video_encoding_manager = None
    episode_success_count = 0
    episode_success_total = 0
    robot_is_home = False

    print("=" * 60)
    print("ロボットを初期化中...")
    robot_config = IlohaConfig(
        right_dynamixel_port="/dev/ttyUSB_RightDynamixel",
        right_robstride_port="auto",
        left_robstride_port="auto",
        left_dynamixel_port="/dev/ttyUSB_LeftDynamixel",
        enable_left_arm=False,
        enable_right_arm=True,
        max_relative_target_1=0.03,
        max_relative_target_2=0.01,
        max_relative_target_3=0.01,
        max_relative_target_4=0.03,
        max_relative_target_5=0.01,
        max_relative_target_6=0.03,
        current_limit_gripper_R=0.2 if args.is_sound_shake else 0.3,
        current_limit_gripper_L=0.3,
    )
    await resolve_auto_robstride_ports(robot_config)
    robot = Iloha(robot_config, debug=False)
    try:
        await robot.connect()
        print("ロボット接続完了")
        await reset_robot_to_home(robot, right_joint_offsets=right_joint_offsets)
        robot_is_home = True

        print("=" * 60)
        print("カメラを初期化中...")
        cameras = initialize_cameras()
        if not cameras:
            return
        robot.cameras = cameras

        print("=" * 60)
        if needs_sound_observations:
            print("音観測系を初期化中...")
            sound_source = RealSoundObservationSource(
                explicit_device_ids=parse_device_ids(args.input_device_ids),
                observation_height=OBSERVATION_HEIGHT,
                observation_width=OBSERVATION_WIDTH,
                remap_soundmap_channels=args.remap_soundmap_channels,
            )
            sound_source.start()
            if not sound_source.wait_until_ready(timeout_s=5.0):
                print("[soundreal] Audio buffers are still warming up. Initial frames may contain zeros.")
        else:
            print("音観測を使わない policy/dataset のため、音観測系の初期化をスキップします")
        if not args.is_sound_shake:
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
            dataset_save_path = save_path
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=CAMERA_FPS,
                root=save_path,
                robot_type="iloha_single_arm",
                features=eval_dataset_features,
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

            condition = make_eval_sound_episode_condition(
                rng,
                sound_index=args.sound_index,
                speaker=args.speaker,
            )
            print(
                "[soundreal] Episode stimulus: "
                f"speaker={condition.speaker}, sound={condition.sound_label}"
            )
            if audio_player is not None:
                audio_player.start(condition)
                print(
                    "[soundreal] Playback started: "
                    f"speaker={condition.speaker}, sound={condition.sound_label}, file={condition.sound_path}"
                )
                if sound_source is not None:
                    sound_source.reset_nmf_state()
                await asyncio.sleep(args.audio_preroll_s)
            else:
                print("[soundreal] Playback disabled by --is-sound-shake.")
                if sound_source is not None:
                    sound_source.reset_nmf_state()

            robot_is_home = False
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
                right_joint_offsets=right_joint_offsets,
                dataset=dataset,
                display_data=args.display_data,
            )

            if audio_player is not None:
                audio_player.stop()

            saved_episode_index: Optional[int] = None

            if dataset is not None:
                if frame_count > 0:
                    dataset.save_episode()
                    saved_episode_index = (
                        int(getattr(dataset, "num_episodes", episode_idx + 1)) - 1
                    )
                    if dataset_save_path is not None:
                        append_sound_condition_csv(
                            dataset_save_path,
                            saved_episode_index,
                            condition,
                            frame_count,
                        )
                    print(f"エピソード {episode_idx + 1} を保存しました")
                else:
                    dataset.clear_episode_buffer()
                    print(f"エピソード {episode_idx + 1} は空だったため保存をスキップしました")

            await reset_robot_to_home(
                robot,
                init=False,
                right_joint_offsets=right_joint_offsets,
            )
            robot_is_home = True
            if episode_idx < args.num_episodes - 1:
                await asyncio.sleep(2.0)

            episode_success = prompt_episode_success(episode_idx + 1, args.num_episodes)

            episode_success_result = {
                "episode_number": episode_idx + 1,
                "episode_index": "" if saved_episode_index is None else saved_episode_index,
                "success": int(episode_success),
                "frame_count": frame_count,
                "sound_index": condition.sound_index,
                "sound_label": condition.sound_label,
                "speaker": condition.speaker,
            }
            result_dataset_path = dataset_save_path if dataset_save_path is not None else dataset_path
            success_csv_path = append_episode_success_csv(
                result_dataset_path,
                episode_success_result,
            )
            print(f"エピソード成功判定をCSVに追記しました: {success_csv_path}")

            episode_success_total += 1
            episode_success_count += int(episode_success)

        if episode_success_total:
            success_rate = episode_success_count / episode_success_total * 100.0
            print("=" * 60)
            print(
                "成功率: "
                f"{episode_success_count}/{episode_success_total} ({success_rate:.2f}%)"
            )

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
            if not robot_is_home:
                try:
                    await reset_robot_to_home(
                        robot,
                        init=False,
                        right_joint_offsets=right_joint_offsets,
                    )
                    robot_is_home = True
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
    parser.add_argument(
        "--calibration-path",
        "--calibration_path",
        default=str(DEFAULT_CALIBRATION_PATH),
        help="iloha_calib.py が生成した calibration JSON（未作成時はゼロ補正）",
    )
    parser.add_argument(
        "--require-calibration",
        action="store_true",
        help="calibration JSON が存在しない場合にエラーにする",
    )
    parser.add_argument("--output_root", type=str, default="datasets", help="保存先ルート")
    parser.add_argument("--save_data", action="store_true", help="評価時の観測と action を保存する")
    parser.add_argument("--episode_time_s", type=float, default=60.0, help="1 エピソードの長さ（秒）")
    parser.add_argument("--num_episodes", type=int, default=1, help="評価エピソード数")
    parser.add_argument("--display_data", action="store_true", help="rerun で可視化する")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="推論デバイス")
    parser.add_argument("--seed", type=int, default=0, help="音条件サンプリング用 seed")
    parser.add_argument(
        "--is-sound-shake",
        "--is_sound_shake",
        dest="is_sound_shake",
        action="store_true",
        help="soundShake 評価として音声再生を無効化する",
    )
    parser.add_argument(
        "--sound_index",
        type=int,
        choices=sorted(SOUND_FILES.keys()),
        default=None,
        help="鳴らす音の種類。未指定時はエピソードごとにランダム",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        choices=SOUND_SPEAKER_CHOICES,
        default=None,
        help="音を鳴らすスピーカー（left/right）。未指定時はエピソードごとにランダム",
    )
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
    parser.add_argument(
        "--remap-soundmap-channels",
        "--remap_soundmap_channels",
        action="store_true",
        default=True,
        help="SoundMap チャンネルを右上から時計回りに G0,B0,R1,R0 へ入れ替える",
    )
    parser.add_argument(
        "--no-remap-soundmap-channels",
        dest="remap_soundmap_channels",
        action="store_false",
        help="SoundMap チャンネルを入れ替えずに保存する",
    )
    asyncio.run(main(parser.parse_args()))
