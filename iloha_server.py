#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import websockets
import socket
import json
import struct
import threading
import numpy as np
from typing import Optional
import time
from pathlib import Path
from lerobot.robots.iloha import Iloha, IlohaConfig, JOINT_NAMES
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras import make_cameras_from_configs
from soundreal_utils import (
    DEFAULT_CAMERA_CONFIGS,
    OBSERVATION_HEIGHT,
    OBSERVATION_WIDTH,
    SOUNDREAL_TASK_NAME,
    SOUND_FILES,
    SOUND_LABELS,
    LoopingStereoPlayer,
    RealSoundObservationSource,
    RIGHT_ARM_FEATURE_NAMES,
    SoundEpisodeCondition,
    build_soundreal_dataset_features,
    decode_right_arm_action_packet,
    full_action_to_right_feature_dict,
    get_camera_configs,
    is_soundreal_task,
    make_full_action_from_right,
    preprocess_cam_high_frame,
    preprocess_camera_frame,
    right_arm_full_slice,
)

USE_RIGHT_SPEAKER = False
FIXED_SOUND_INDEX: Optional[int] = 1 # 0(buzzer) or 1(piyopiyo) or None(0,1)
EPISODE_NUM = 50

TASK = SOUNDREAL_TASK_NAME

class RobotCommunicationNode:
    # データセット設定
    DATASET_ROOT = Path("datasets")
    DATASET_FPS = 30
    EPISODE_MAX_TIME_S = 180
    CAMERA_MAX_FRAME_AGE_MS = 250
    # カメラ設定
    CAMERA_CONFIGS = DEFAULT_CAMERA_CONFIGS

    def __init__(self):
        self.websocket_port = 8080
        self.task = TASK
        self.soundreal_enabled = is_soundreal_task(self.task)
        self.camera_configs = get_camera_configs(self.task)
        self.state_names = RIGHT_ARM_FEATURE_NAMES if self.soundreal_enabled else tuple(JOINT_NAMES)
        self.dataset_prefix = self.task if self.soundreal_enabled else "iloha"
        self.unity_joint_port: Optional[int] = None
        self.is_connected = False
        self.is_receiving_joints = False
        self.joint_thread: Optional[threading.Thread] = None
        self.stop_threads = False
        self.robot: Optional[Iloha] = None
        self.robot_connected = False
        self.robot_control_task: Optional[asyncio.Task] = None
        self.stop_event = asyncio.Event()  # タスク停止用Event
        self.robot_lock = asyncio.Lock()  # ロボット制御排他用Lock
        self.reset_in_progress = asyncio.Event()  # リセット処理中フラグ
        self.latest_action = None
        self.action_lock = threading.Lock()  # UDP受信スレッド用
        self.control_frequency = 60 # Hz
        self.relative_warmup_seconds = 3.0
        self.absolute_mode_delta_threshold = 0.2  # rad
        self.is_recording = False
        self.recording_ready = False  # 記録準備完了フラグ
        self.current_dataset: Optional[LeRobotDataset] = None
        self.recording_start_time: Optional[float] = None
        self.recording_task: Optional[asyncio.Task] = None
        self.cameras: dict = {}
        self.video_encoding_manager: Optional[VideoEncodingManager] = None
        self.sound_source: Optional[RealSoundObservationSource] = None
        self.audio_player: Optional[LoopingStereoPlayer] = None
        self.current_sound_condition = None
        self.awaiting_recording_trigger = False
        self.first_action_time: Optional[float] = None
        self.previous_recorded_action_state: Optional[np.ndarray] = None
        self.rng = np.random.default_rng()

    def _get_saved_episode_count(self) -> int:
        """現在のデータセットに保存済みのエピソード数を返す"""
        if self.current_dataset is None:
            return 0
        return int(getattr(self.current_dataset, "num_episodes", 0))

    @staticmethod
    def _make_right_arm_only_action(right_arm_action: np.ndarray) -> np.ndarray:
        full_action = np.zeros(14, dtype=np.float32)
        full_action[7:14] = np.asarray(right_arm_action[7:14], dtype=np.float32)
        return full_action

    def _make_right_arm_home_action(self) -> np.ndarray:
        home_action = np.zeros_like(self.robot.old_action, dtype=np.float32)
        home_action[7:10] = self.robot.old_action[7:10]
        return home_action

    def _get_buffered_frame_count(self) -> int:
        """現在のエピソードバッファに積まれているフレーム数を返す"""
        if self.current_dataset is None:
            return 0
        episode_buffer = getattr(self.current_dataset, "episode_buffer", None)
        if episode_buffer is None:
            return 0
        return int(episode_buffer.get("size", 0))

    def _get_next_dataset_number(self, prefix: Optional[str] = None) -> int:
        """既存のデータセット番号を確認し、次の番号を返す"""
        if not self.DATASET_ROOT.exists():
            return 0
        prefix = prefix or self.dataset_prefix
        existing_nums = []
        for path in self.DATASET_ROOT.iterdir():
            if path.is_dir() and path.name.startswith(f"{prefix}_"):
                try:
                    num = int(path.name.rsplit("_", 1)[-1])
                    existing_nums.append(num)
                except ValueError:
                    continue
        return max(existing_nums) + 1 if existing_nums else 0

    async def initialize_robot(self):
        try:
            config = IlohaConfig(
                left_dynamixel_port="/dev/ttyUSB_LeftDynamixel",
                left_robstride_port="/dev/ttyUSB3",
                right_robstride_port="/dev/ttyUSB0",
                right_dynamixel_port="/dev/ttyUSB_RightDynamixel",
                enable_left_arm=False,
                enable_right_arm=True,
                max_relative_target_1=0.03, # yaw
                max_relative_target_2=0.01, # pitch
                max_relative_target_3=0.01, # pitch
                max_relative_target_4=0.03, # yaw
                max_relative_target_5=0.01, # pitch
                max_relative_target_6=0.03, # yaw
                current_limit_gripper_R=0.3,
                current_limit_gripper_L=0.3,
            )
            self.robot = Iloha(config, debug=False)
            await self.robot.connect()
            self.robot_connected = True
            await asyncio.sleep(2.0)
            print("Ilohaロボット初期化・接続完了")
        except Exception as e:
            print(f"ロボット初期化エラー: {e}")
            self.robot_connected = False

    def _initialize_cameras(self) -> dict:
        """カメラを初期化して辞書で返す"""
        try:
            camera_configs = {}
            for name, config_dict in self.camera_configs.items():
                camera_configs[name] = RealSenseCameraConfig(**config_dict)
            cameras = make_cameras_from_configs(camera_configs)
            for name, camera in cameras.items():
                print(f"{name} を接続中...")
                camera.connect(warmup=True)
                time.sleep(1.0)
            print(f"{len(cameras)}台のカメラを初期化しました")
            return cameras
        except Exception as e:
            print(f"カメラ初期化エラー: {e}")
            return {}

    def _ensure_sound_runtime(self) -> None:
        if not self.soundreal_enabled:
            return
        if self.sound_source is None:
            self.sound_source = RealSoundObservationSource()
            self.sound_source.start()
            if not self.sound_source.wait_until_ready(timeout_s=5.0):
                print("[soundreal] Audio buffers are still warming up. Initial frames may contain zeros.")
        if self.audio_player is None:
            self.audio_player = LoopingStereoPlayer()

    def _set_next_sound_condition(self) -> None:
        if not self.soundreal_enabled:
            return
        self._ensure_sound_runtime()
        if self.sound_source is not None:
            self.sound_source.reset_nmf_state()
        next_episode_index = self._get_saved_episode_count()
        sound_index = (
            FIXED_SOUND_INDEX
            if FIXED_SOUND_INDEX is not None
            else 0 if next_episode_index < EPISODE_NUM // 2 else 1
        )
        if sound_index not in SOUND_FILES:
            raise ValueError(
                f"Invalid FIXED_SOUND_INDEX={sound_index}. "
                f"Available sound indices: {sorted(SOUND_FILES.keys())}"
            )
        speaker = "right" if USE_RIGHT_SPEAKER else "left"
        self.current_sound_condition = SoundEpisodeCondition(
            sound_index=sound_index,
            sound_label=SOUND_LABELS[sound_index],
            sound_path=SOUND_FILES[sound_index],
            speaker=speaker,
        )
        print(
            "[soundreal] Episode stimulus prepared: "
            f"episode={next_episode_index + 1}/{EPISODE_NUM}, "
            f"speaker={self.current_sound_condition.speaker}, "
            f"sound={self.current_sound_condition.sound_label}"
        )
        self.audio_player.start(self.current_sound_condition)

    def _stop_sound_runtime(self) -> None:
        if self.audio_player is not None:
            self.audio_player.stop()
            self.audio_player = None
        if self.sound_source is not None:
            self.sound_source.stop()
            self.sound_source = None
        self.current_sound_condition = None

    async def start_recording(self, websocket):
        """記録を開始"""
        try:
            if self.is_recording:
                response = {"status": "recording_error", "message": "既に記録中です"}
                await websocket.send(json.dumps(response))
                return
            if self.recording_ready or self.current_dataset is not None:
                response = {"status": "recording_error", "message": "既に記録セッションが準備されています"}
                await websocket.send(json.dumps(response))
                return
            if not self.robot_connected:
                response = {"status": "recording_error", "message": "ロボットが接続されていません"}
                await websocket.send(json.dumps(response))
                return
            self.cameras = self._initialize_cameras()
            if not self.cameras:
                response = {"status": "recording_error", "message": "カメラの初期化に失敗しました"}
                await websocket.send(json.dumps(response))
                return
            self.robot.cameras = self.cameras
            if self.soundreal_enabled:
                self._ensure_sound_runtime()
            dataset_num = self._get_next_dataset_number(self.dataset_prefix)
            dataset_name = f"{self.dataset_prefix}_{dataset_num}"
            repo_id = f"local/{dataset_name}"
            dataset_path = self.DATASET_ROOT / dataset_name
            print(f"データセットを作成中: {repo_id}")
            if self.soundreal_enabled:
                dataset_features = build_soundreal_dataset_features()
            else:
                dataset_features = {
                    "observation.state": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
                    "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
                }
                for key in self.camera_configs.keys():
                    image_shape = (
                        (OBSERVATION_HEIGHT, OBSERVATION_WIDTH, 3)
                        if key == "cam_high"
                        else (480, 640, 3)
                    )
                    dataset_features[f"observation.images.{key}"] = {
                        "dtype": "video",
                        "shape": image_shape,
                        "names": ("height", "width", "channels")
                    }
            image_feature_count = sum(
                1 for key in dataset_features.keys() if key.startswith("observation.images.")
            )
            self.current_dataset = LeRobotDataset.create(
                repo_id,
                self.DATASET_FPS,
                root=dataset_path,
                robot_type="iloha_single_arm" if self.soundreal_enabled else "aloha",
                features=dataset_features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=max(1, image_feature_count),
                video_backend="pyav",  # torchcodecのAV1デコード問題を回避
            )
            self.video_encoding_manager = VideoEncodingManager(self.current_dataset)
            self.video_encoding_manager.__enter__()
            print(f"データセット作成完了: {repo_id}")
            self.previous_recorded_action_state = None
            self.recording_ready = True
            self.awaiting_recording_trigger = True
            self.first_action_time = None
            if self.soundreal_enabled:
                self._set_next_sound_condition()
            self.recording_task = asyncio.create_task(self.record_episode())
            response = {"status": "recording_ready", "message": f"記録準備完了: {dataset_name}。初回アクション受信後に記録を開始します"}
            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"記録開始エラー: {e}")
            import traceback
            traceback.print_exc()
            try:
                await self._full_cleanup_recording()
            except Exception as cleanup_error:
                print(f"記録開始失敗後のクリーンアップエラー: {cleanup_error}")
            response = {"status": "recording_error", "message": f"記録開始エラー: {e}"}
            await websocket.send(json.dumps(response))

    async def stop_recording(self):
        """記録を停止（エピソードの記録のみ停止、リソースは保持）"""
        if self.is_recording:
            self.is_recording = False
            if self.recording_task and not self.recording_task.done():
                self.recording_task.cancel()
                try:
                    await self.recording_task
                except asyncio.CancelledError:
                    pass
            self.recording_task = None
            self.first_action_time = None
            self.recording_start_time = None
            self.awaiting_recording_trigger = False
            with self.action_lock:
                self.latest_action = None
            print("記録を停止しました（リソースは保持）")

    async def save_episode(self, websocket):
        """エピソードを保存（データセットとリソースは保持）"""
        try:
            await self.stop_recording()
            if self.current_dataset is None:
                response = {"status": "save_error", "message": "データセットが存在しません"}
                await websocket.send(json.dumps(response))
                return
            if self._get_buffered_frame_count() == 0:
                if getattr(self.current_dataset, "episode_buffer", None) is not None:
                    self.current_dataset.clear_episode_buffer()
                await self._prepare_next_episode()
                response = {
                    "status": "save_skipped",
                    "message": "保存対象のフレームがまだありません。初回アクション受信後に記録が始まります。",
                }
                await websocket.send(json.dumps(response))
                return
            self.current_dataset.save_episode()
            saved_episode_count = self._get_saved_episode_count()
            print(f"エピソード保存完了 ({saved_episode_count}/{EPISODE_NUM})")
            if saved_episode_count >= EPISODE_NUM:
                await self._finish_dataset_and_shutdown_robot()
                response = {
                    "status": "recording_complete",
                    "message": f"{EPISODE_NUM}エピソードを保存しました。データセットを保存し、ロボットを終了しました。",
                }
                await websocket.send(json.dumps(response))
                return
            await self._prepare_next_episode()
            response = {"status": "save_complete", "message": "エピソードを保存しました。次のエピソードの準備ができています"}
            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"エピソード保存エラー: {e}")
            import traceback
            traceback.print_exc()
            response = {"status": "save_error", "message": f"保存エラー: {e}"}
            await websocket.send(json.dumps(response))

    async def discard_episode(self, websocket):
        """エピソードを破棄（データセットとリソースは保持）"""
        try:
            await self.stop_recording()
            if self.current_dataset is None:
                response = {"status": "discard_error", "message": "データセットが存在しません"}
                await websocket.send(json.dumps(response))
                return
            if getattr(self.current_dataset, "episode_buffer", None) is not None:
                self.current_dataset.clear_episode_buffer()
            await self._prepare_next_episode()
            print("エピソード破棄完了")
            response = {"status": "discard_complete", "message": "エピソードを破棄しました。次のエピソードの準備ができています"}
            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"エピソード破棄エラー: {e}")
            import traceback
            traceback.print_exc()
            response = {"status": "discard_error", "message": f"破棄エラー: {e}"}
            await websocket.send(json.dumps(response))

    async def _prepare_next_episode(self):
        """次のエピソード記録の準備"""
        if (
            self.recording_ready
            and self.current_dataset is not None
            and (self.recording_task is None or self.recording_task.done())
        ):
            self.awaiting_recording_trigger = True
            self.first_action_time = None
            self.recording_start_time = None
            self.previous_recorded_action_state = None
            if self.soundreal_enabled:
                self._set_next_sound_condition()
            self.recording_task = asyncio.create_task(self.record_episode())
            print("次のエピソードの記録準備完了")

    async def _full_cleanup_recording(self):
        """記録関連のリソースを完全にクリーンアップ"""
        await self.stop_recording()
        if self.video_encoding_manager:
            try:
                self.video_encoding_manager.__exit__(None, None, None)
            except Exception as e:
                print(f"VideoEncodingManager終了エラー: {e}")
            self.video_encoding_manager = None
        elif self.current_dataset:
            try:
                self.current_dataset.finalize()
            except Exception as e:
                print(f"データセット終了エラー: {e}")
        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except Exception as e:
                print(f"カメラ切断エラー: {e}")
        self.cameras = {}
        self._stop_sound_runtime()
        if self.robot:
            self.robot.cameras = {}
        self.current_dataset = None
        self.recording_start_time = None
        self.previous_recorded_action_state = None
        self.recording_ready = False

    async def _finish_dataset_and_shutdown_robot(self):
        """規定エピソード数に到達したあと、データセットとロボットを終了する"""
        print(f"{EPISODE_NUM}エピソードに到達しました。データセットを確定し、ロボットを終了します。")
        await self._full_cleanup_recording()
        await self.cleanup_connection()
        if self.robot_connected and self.robot:
            try:
                await self.robot.disconnect()
                print("ロボット接続切断")
            except Exception as e:
                print(f"ロボット切断エラー: {e}")
            finally:
                self.robot_connected = False

    def _capture_latest_observation(self) -> dict:
        """カメラの最新フレームを非同期制御を止めずに取得する"""
        if self.robot is None:
            raise RuntimeError("ロボットが初期化されていません")

        obs = {}
        joint_state = self.robot.old_action.copy()
        for name, camera in self.cameras.items():
            try:
                frame = camera.read_latest(max_age_ms=self.CAMERA_MAX_FRAME_AGE_MS)
            except Exception:
                # 最新フレームがまだ無い場合のみ、新規フレーム待ちにフォールバックする
                frame = camera.async_read(timeout_ms=self.CAMERA_MAX_FRAME_AGE_MS)
            if self.soundreal_enabled:
                obs[name] = preprocess_camera_frame(frame)
            elif name == "cam_high":
                obs[name] = preprocess_cam_high_frame(frame)
            else:
                obs[name] = frame

        if self.soundreal_enabled:
            if self.sound_source is None:
                raise RuntimeError("Sound runtime is not initialized.")
            obs.update(self.sound_source.get_latest_images())
            obs.update(full_action_to_right_feature_dict(joint_state))
        else:
            for i, joint_name in enumerate(JOINT_NAMES):
                obs[joint_name] = joint_state[i]
        return obs

    def _get_observation_state_array(self, obs: dict) -> np.ndarray:
        """obs 内の observation.state 相当の関節値を配列で返す"""
        return np.array([obs[joint_name] for joint_name in self.state_names], dtype=np.float32)

    def _overwrite_observation_state(self, obs: dict, state: np.ndarray) -> None:
        """obs 内の observation.state 相当の関節値を指定値で置き換える"""
        for idx, joint_name in enumerate(self.state_names):
            obs[joint_name] = float(state[idx])

    def _record_frame_sync(self) -> None:
        """1フレーム分の観測構築とデータセット書き込みをワーカースレッドで行う"""
        if self.current_dataset is None:
            raise RuntimeError("データセットが初期化されていません")

        obs = self._capture_latest_observation()
        current_state = self._get_observation_state_array(obs)
        previous_action_state = (
            self.previous_recorded_action_state
            if self.previous_recorded_action_state is not None
            else current_state
        )
        self._overwrite_observation_state(obs, previous_action_state)
        action_data = {
            joint_name: float(current_state[idx]) for idx, joint_name in enumerate(self.state_names)
        }
        observation_frame = build_dataset_frame(
            self.current_dataset.features, obs, prefix="observation"
        )
        action_frame = build_dataset_frame(
            self.current_dataset.features, action_data, prefix="action"
        )
        frame = {**observation_frame, **action_frame, "task": self.task}
        self.current_dataset.add_frame(frame)
        self.previous_recorded_action_state = current_state.copy()

    async def record_episode(self):
        """30FPSで画像と関節角度を記録"""
        print("エピソード記録ループ準備完了。初回アクション受信待機中...")
        frame_count = 0
        while not self.is_recording and self.recording_ready:
            await asyncio.sleep(0.01)
        if not self.is_recording:
            print("記録がキャンセルされました")
            return
        print("記録開始！")
        try:
            while self.is_recording:
                start_time = time.perf_counter()
                await asyncio.to_thread(self._record_frame_sync)
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - self.recording_start_time
                    print(f"記録中... {frame_count}フレーム ({elapsed_time:.1f}秒)")
                if time.time() - self.recording_start_time >= self.EPISODE_MAX_TIME_S:
                    print(f"最大記録時間({self.EPISODE_MAX_TIME_S}秒)に達しました")
                    break
                elapsed = time.perf_counter() - start_time
                if elapsed > (1.0 / self.control_frequency):
                    print(f"記録フレーム処理が重いです: {elapsed * 1000:.1f} ms")
                sleep_duration = 1.0 / self.DATASET_FPS - elapsed
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            print("記録ループがキャンセルされました")
        except Exception as e:
            print(f"記録ループエラー: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"エピソード記録ループ終了 (合計{frame_count}フレーム)")

    async def websocket_handler(self, websocket):
        print(f"WebSocket接続: {websocket.remote_address}")
        # 新しい接続でロボットが切断されている場合は再接続を試みる
        if not self.robot_connected:
            print("ロボット再接続を試行中...")
            await self.initialize_robot()
        try:
            print("Unity側からのメッセージを待機中...")
            message = await websocket.recv()
            data = json.loads(message)
            self.unity_joint_port = data.get('joint_send_port')
            self.is_connected = True
            self.start_udp_communication()
            await asyncio.sleep(0.5)
            response = {"status": "connected", "message": "接続情報受信完了"}
            await websocket.send(json.dumps(response))
            await self.handle_websocket_messages(websocket)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket接続がクライアント側から閉じられました")
        except Exception as e:
            print(f"WebSocketエラー: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("WebSocket接続終了")
            await self.cleanup_connection()

    async def handle_websocket_messages(self, websocket):
        try:
            async for message in websocket:
                print(f"WebSocketメッセージ受信: {message}")
                try:
                    data = json.loads(message)
                    command = data.get('command')
                    if command == 'reset_robot':
                        print("ロボットリセット要求を受信しました")
                        await self.handle_reset_request(websocket)
                    elif command == 'save_data':
                        print("データ保存要求を受信しました")
                        await self.save_episode(websocket)
                    elif command == 'discard_data':
                        print("データ破棄要求を受信しました")
                        await self.discard_episode(websocket)
                    elif command == 'recording':
                        print("recording要求を受信しました")
                        await self.start_recording(websocket)
                    elif command == 'teleoperation':
                        print("teleoperation要求を受信しました")
                        if self.is_recording or self.recording_ready:
                            await self._full_cleanup_recording()
                            print("記録を停止し、テレオペレーションモードに切り替えました")
                        response = {"status": "teleoperation_mode", "message": "テレオペレーションモードに切り替えました"}
                        await websocket.send(json.dumps(response))
                    else:
                        print(f"不明なコマンド: {command}")
                except json.JSONDecodeError:
                    print(f"JSON解析エラー: {message}")
                except Exception as e:
                    print(f"メッセージ処理エラー: {e}")
                    import traceback
                    traceback.print_exc()
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket接続が閉じられました（メッセージ受信中）")
        except Exception as e:
            print(f"WebSocketメッセージ受信エラー: {e}")

    async def handle_reset_request(self, websocket):
        try:
            print("ロボットリセット処理開始...")
            if not self.robot_connected:
                response = {"status": "reset_error", "message": "ロボットが接続されていません"}
                await websocket.send(json.dumps(response))
                return
            self.reset_in_progress.set()
            async with self.robot_lock:  # ロボット制御を排他制御
                try:
                    print("制御タスク停止シグナル送信中...")
                    self.stop_event.set()
                    self.stop_threads = True
                    if self.robot_control_task and not self.robot_control_task.done():
                        print("ロボット制御タスクの停止を待機中...")
                        try:
                            await asyncio.wait_for(self.robot_control_task, timeout=2.0)
                        except asyncio.TimeoutError:
                            print("警告: ロボット制御タスクが停止できませんでした")
                            self.robot_control_task.cancel()
                    home_action = self._make_right_arm_home_action()
                    await self.robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
                    await asyncio.sleep(2.0)
                    home_action = self._make_right_arm_only_action(np.zeros_like(self.robot.old_action))
                    await self.robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
                    await asyncio.sleep(1.0)
                    print("ホームポジション移動完了")
                finally:
                    with self.action_lock:
                        self.latest_action = None
                    self.stop_event.clear()
                    self.stop_threads = False
                    if self.is_receiving_joints:
                        print("UDP受信スレッドとロボット制御タスクを再開中...")
                        if not self.joint_thread or not self.joint_thread.is_alive():
                            print("UDP受信スレッドを再起動中...")
                            self.joint_thread = threading.Thread(target=self.joint_receiver_thread)
                            self.joint_thread.daemon = True
                            self.joint_thread.start()
                        await self.start_robot_control_task()
                    if self.recording_ready and not self.is_recording:
                        await self._prepare_next_episode()
                    self.reset_in_progress.clear()
            print("ロボットリセット処理完了")
            response = {"status": "reset_complete", "message": "ロボットリセットが完了しました"}
            await websocket.send(json.dumps(response))
        except Exception as e:
            self.reset_in_progress.clear()
            print(f"リセット処理エラー: {e}")
            import traceback
            traceback.print_exc()
            error_response = {"status": "reset_error", "message": f"リセット処理でエラーが発生しました: {e}"}
            await websocket.send(json.dumps(error_response))

    def start_udp_communication(self):
        if self.unity_joint_port:
            self.stop_threads = False
            self.is_receiving_joints = True
            self.joint_thread = threading.Thread(target=self.joint_receiver_thread)
            self.joint_thread.daemon = True
            self.joint_thread.start()
            asyncio.create_task(self.start_robot_control_task())
            print("UDP通信スレッドと非同期ロボット制御タスクを開始しました")

    async def start_robot_control_task(self):
        if self.robot_connected:
            if self.robot_control_task and not self.robot_control_task.done():
                print("ロボット制御タスクは既に動作中です")
                return
            self.robot_control_task = asyncio.create_task(self.robot_control_worker())
            print("新しいロボット制御タスクを開始しました")

    async def robot_control_worker(self):
        print("ロボット制御ワーカー開始")
        try:
            current_latest_action = None
            while self.robot_connected and not self.stop_threads and not self.stop_event.is_set():
                start_time = time.perf_counter()
                with self.action_lock:
                    latest_action = None if self.latest_action is None else self.latest_action.copy()
                if latest_action is None:
                    current_latest_action = None
                    await asyncio.sleep(0.01)
                    continue
                current_latest_action = latest_action
                if self.reset_in_progress.is_set():
                    await asyncio.sleep(0.1)
                    continue
                if self.first_action_time is None:
                    self.first_action_time = time.time()
                    print("初回アクション受信を記録しました。安定化するまで相対制限付きで制御します。")
                if self.recording_ready and not self.is_recording and self.awaiting_recording_trigger:
                    self.awaiting_recording_trigger = False
                    self.first_action_time = time.time()
                    self.is_recording = True
                    self.recording_start_time = time.time()
                elapsed_since_first_action = time.time() - self.first_action_time
                previous_action = (
                    right_arm_full_slice(self.robot.old_action)
                    if self.soundreal_enabled
                    else self.robot.old_action.copy()
                )
                delta_from_previous = np.abs(current_latest_action - previous_action)
                max_delta = float(np.max(delta_from_previous))
                use_relative = (
                    elapsed_since_first_action < self.relative_warmup_seconds
                    or max_delta > self.absolute_mode_delta_threshold
                )
                # if max_delta > self.absolute_mode_delta_threshold:
                    # print(
                    #     f"急激な目標変化を検出したため、相対制限を維持します "
                    #     f"(max_delta={max_delta:.3f} rad)"
                    # )
                async with self.robot_lock:
                    if not self.reset_in_progress.is_set() and self.robot_connected:
                        action_to_send = (
                            make_full_action_from_right(current_latest_action)
                            if self.soundreal_enabled
                            else current_latest_action
                        )
                        await self.robot.async_send_action(
                            action_to_send,
                            use_relative=use_relative,
                            use_filter=not use_relative,
                        )
                elapsed_time = time.perf_counter() - start_time
                sleep_duration = 1.0 / self.control_frequency - elapsed_time
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            print("ロボット制御ワーカーがキャンセルされました")
        finally:
            print("ロボット制御ワーカー終了")

    def joint_receiver_thread(self):
        print(f"関節角度受信開始: ポート{self.unity_joint_port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        sock.bind(('0.0.0.0', self.unity_joint_port))
        sock.settimeout(0.1)
        try:
            while self.is_receiving_joints and not self.stop_threads:
                try:
                    data, addr = sock.recvfrom(256)
                    if self.soundreal_enabled:
                        mode, right_action = decode_right_arm_action_packet(data)
                        if mode == 1 and right_action is not None and self.robot_connected:
                            with self.action_lock:
                                self.latest_action = right_action
                    elif len(data) >= 57:  # 1バイト（モード） + 56バイト（14個のfloat32）
                        mode = data[0]
                        joint_angles = [
                            struct.unpack('<f', data[1 + i * 4:5 + i * 4])[0]
                            for i in range(14)
                        ]
                        joint_angles[0] += np.pi / 2
                        joint_angles[1] -= np.pi / 2
                        joint_angles[2] = -joint_angles[2] - np.pi / 2
                        joint_angles[3] = -joint_angles[3]
                        joint_angles[4] = -joint_angles[4]
                        joint_angles[5] = -joint_angles[5]
                        joint_angles[7] -= np.pi / 2
                        joint_angles[8] -= np.pi / 2
                        joint_angles[9] = -joint_angles[9] - np.pi / 2
                        joint_angles[10] = -joint_angles[10]
                        joint_angles[11] = -joint_angles[11]
                        joint_angles[12] = -joint_angles[12]
                        if mode == 1 and self.robot_connected:
                            with self.action_lock:
                                self.latest_action = np.array(joint_angles, dtype=np.float32)
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.is_receiving_joints:
                        print(f"関節角度受信エラー: {e}")
                    break
        finally:
            sock.close()
            print("関節角度受信終了")

    async def cleanup_connection(self):
        print("WebSocket接続クリーンアップ開始...")
        self.is_connected = False
        self.is_receiving_joints = False
        self.stop_threads = True
        self.stop_event.set()
        if self.robot_control_task and not self.robot_control_task.done():
            print("ロボット制御タスクを終了中...")
            self.robot_control_task.cancel()
            try:
                await asyncio.wait_for(self.robot_control_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self.robot_control_task = None
        if self.joint_thread and self.joint_thread.is_alive():
            print("UDP受信スレッドを終了中...")
            self.joint_thread.join(timeout=2)
        self.joint_thread = None
        with self.action_lock:
            self.latest_action = None
        if self.robot_connected and self.robot:
            try:
                print("ロボットを初期位置に戻しています...")
                home_action = self._make_right_arm_home_action()
                await self.robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
                await asyncio.sleep(2.0)
                home_action = self._make_right_arm_only_action(np.zeros_like(self.robot.old_action))
                await self.robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
                await asyncio.sleep(1.0)
                print("ロボット初期位置復帰完了")
            except Exception as e:
                print(f"初期位置復帰エラー: {e}")
        print("WebSocket接続クリーンアップ完了")

    async def cleanup(self):
        if self.is_recording or self.recording_ready or self.current_dataset is not None:
            print("記録中のデータセットを終了します...")
            await self._full_cleanup_recording()
        await self.cleanup_connection()
        if self.robot_connected and self.robot:
            try:
                await self.robot.disconnect()
                print("ロボット接続切断")
                self.robot_connected = False
            except Exception as e:
                print(f"ロボット切断エラー: {e}")
        print("完全クリーンアップ完了")

    async def start_server(self):
        print(f"WebSocketサーバーを開始: ポート{self.websocket_port}")
        await self.initialize_robot()
        try:
            async with websockets.serve(self.websocket_handler, "0.0.0.0", self.websocket_port):
                print("サーバー起動完了。Unityからの接続を待機中...")
                await asyncio.Future()
        except KeyboardInterrupt:
            print("\nサーバー停止中...")
        except Exception as e:
            print(f"サーバーエラー: {e}")
        finally:
            await self.cleanup()

if __name__ == "__main__":
    node = RobotCommunicationNode()
    print("Unity-Iloha通信サーバーを起動します...")
    asyncio.run(node.start_server())

# uv run iloha_server.py
