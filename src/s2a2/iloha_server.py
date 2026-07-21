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
from s2a2.soundreal_utils import (
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
    preprocess_soundreal_camera_frame,
    right_arm_full_slice,
)

USE_RIGHT_SPEAKER = True
FIXED_SOUND_INDEX: Optional[int] = 1 # 0(buzzer) or 1(piyopiyo) or None(0,1)
IS_SOUND_SHAKE = True
EPISODE_NUM = 50

TASK = SOUNDREAL_TASK_NAME
if IS_SOUND_SHAKE:
    TASK = "Shake the cans. Pick up the one that makes sound and place it in the box."

class RobotCommunicationNode:
    # Dataset settings
    DATASET_ROOT = Path("datasets")
    DATASET_FPS = 30
    EPISODE_MAX_TIME_S = 180
    CAMERA_MAX_FRAME_AGE_MS = 80
    # Camera settings
    CAMERA_CONFIGS = DEFAULT_CAMERA_CONFIGS

    def __init__(self):
        self.websocket_port = 8080
        self.task = TASK
        self.soundreal_enabled = is_soundreal_task(self.task) or IS_SOUND_SHAKE
        self.sound_playback_enabled = self.soundreal_enabled and not IS_SOUND_SHAKE
        soundreal_camera_task = SOUNDREAL_TASK_NAME if self.soundreal_enabled else self.task
        self.camera_configs = get_camera_configs(soundreal_camera_task)
        self.state_names = RIGHT_ARM_FEATURE_NAMES if self.soundreal_enabled else tuple(JOINT_NAMES)
        self.dataset_prefix = SOUNDREAL_TASK_NAME if self.soundreal_enabled else "iloha"
        self.unity_joint_port: Optional[int] = None
        self.is_connected = False
        self.is_receiving_joints = False
        self.joint_thread: Optional[threading.Thread] = None
        self.stop_threads = False
        self.robot: Optional[Iloha] = None
        self.robot_connected = False
        self.robot_control_task: Optional[asyncio.Task] = None
        self.stop_event = asyncio.Event()  # Event for stopping tasks
        self.robot_lock = asyncio.Lock()  # Lock for exclusive robot control
        self.reset_in_progress = asyncio.Event()  # Reset-in-progress flag
        self.latest_action = None
        self.action_lock = threading.Lock()  # For UDP receiver thread
        self.control_frequency = 60 # Hz
        self.relative_warmup_seconds = 3.0
        self.absolute_mode_delta_threshold = 0.2  # rad
        self.is_recording = False
        self.recording_ready = False  # Recording-ready flag
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
        """Return the number of episodes already saved in the current dataset."""
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
        """Return the number of frames buffered in the current episode."""
        if self.current_dataset is None:
            return 0
        episode_buffer = getattr(self.current_dataset, "episode_buffer", None)
        if episode_buffer is None:
            return 0
        return int(episode_buffer.get("size", 0))

    def _get_next_dataset_number(self, prefix: Optional[str] = None) -> int:
        """Check existing dataset numbers and return the next one."""
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
                current_limit_gripper_R=0.02 if IS_SOUND_SHAKE else 0.3,
                current_limit_gripper_L=0.3,
            )
            self.robot = Iloha(config, debug=False)
            await self.robot.connect()
            self.robot_connected = True
            await asyncio.sleep(2.0)
            print("Iloha robot initialized and connected")
        except Exception as e:
            print(f"Robot initialization error: {e}")
            self.robot_connected = False

    def _initialize_cameras(self) -> dict:
        """Initialize the cameras and return them as a dict."""
        try:
            camera_configs = {}
            for name, config_dict in self.camera_configs.items():
                camera_configs[name] = RealSenseCameraConfig(**config_dict)
            cameras = make_cameras_from_configs(camera_configs)
            for name, camera in cameras.items():
                print(f"Connecting {name}...")
                camera.connect(warmup=True)
                time.sleep(1.0)
            print(f"Initialized {len(cameras)} camera(s)")
            return cameras
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return {}

    def _ensure_sound_runtime(self) -> None:
        if not self.soundreal_enabled:
            return
        if self.sound_source is None:
            self.sound_source = RealSoundObservationSource(remap_soundmap_channels=True)
            self.sound_source.start()
            if not self.sound_source.wait_until_ready(timeout_s=5.0):
                print("[soundreal] Audio buffers are still warming up. Initial frames may contain zeros.")
        if self.sound_playback_enabled and self.audio_player is None:
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
        if self.sound_playback_enabled and self.audio_player is not None:
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
        """Start recording."""
        try:
            if self.is_recording:
                response = {"status": "recording_error", "message": "Already recording"}
                await websocket.send(json.dumps(response))
                return
            if self.recording_ready or self.current_dataset is not None:
                response = {"status": "recording_error", "message": "A recording session is already prepared"}
                await websocket.send(json.dumps(response))
                return
            if not self.robot_connected:
                response = {"status": "recording_error", "message": "Robot is not connected"}
                await websocket.send(json.dumps(response))
                return
            self.cameras = self._initialize_cameras()
            if not self.cameras:
                response = {"status": "recording_error", "message": "Failed to initialize cameras"}
                await websocket.send(json.dumps(response))
                return
            self.robot.cameras = self.cameras
            if self.soundreal_enabled:
                self._ensure_sound_runtime()
            dataset_num = self._get_next_dataset_number(self.dataset_prefix)
            dataset_name = f"{self.dataset_prefix}_{dataset_num}"
            repo_id = f"local/{dataset_name}"
            dataset_path = self.DATASET_ROOT / dataset_name
            print(f"Creating dataset: {repo_id}")
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
                video_backend="pyav",  # Avoid torchcodec AV1 decoding issue
            )
            self.video_encoding_manager = VideoEncodingManager(self.current_dataset)
            self.video_encoding_manager.__enter__()
            print(f"Dataset created: {repo_id}")
            self.previous_recorded_action_state = None
            self.recording_ready = True
            self.awaiting_recording_trigger = True
            self.first_action_time = None
            if self.soundreal_enabled:
                self._set_next_sound_condition()
            self.recording_task = asyncio.create_task(self.record_episode())
            response = {"status": "recording_ready", "message": f"Ready to record: {dataset_name}. Recording will start after the first action is received"}
            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"Recording start error: {e}")
            import traceback
            traceback.print_exc()
            try:
                await self._full_cleanup_recording()
            except Exception as cleanup_error:
                print(f"Cleanup error after failed recording start: {cleanup_error}")
            response = {"status": "recording_error", "message": f"Recording start error: {e}"}
            await websocket.send(json.dumps(response))

    async def stop_recording(self):
        """Stop recording (stops only episode recording; keeps resources)."""
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
            print("Recording stopped (resources kept)")

    async def save_episode(self, websocket):
        """Save the episode (keeps the dataset and resources)."""
        try:
            await self.stop_recording()
            if self.current_dataset is None:
                response = {"status": "save_error", "message": "No dataset exists"}
                await websocket.send(json.dumps(response))
                return
            if self._get_buffered_frame_count() == 0:
                if getattr(self.current_dataset, "episode_buffer", None) is not None:
                    self.current_dataset.clear_episode_buffer()
                await self._prepare_next_episode()
                response = {
                    "status": "save_skipped",
                    "message": "No frames to save yet. Recording starts after the first action is received.",
                }
                await websocket.send(json.dumps(response))
                return
            self.current_dataset.save_episode()
            saved_episode_count = self._get_saved_episode_count()
            print(f"Episode saved ({saved_episode_count}/{EPISODE_NUM})")
            if saved_episode_count >= EPISODE_NUM:
                await self._finish_dataset_and_shutdown_robot()
                response = {
                    "status": "recording_complete",
                    "message": f"Saved {EPISODE_NUM} episodes. Dataset saved and robot shut down.",
                }
                await websocket.send(json.dumps(response))
                return
            await self._prepare_next_episode()
            response = {"status": "save_complete", "message": "Episode saved. Ready for the next episode"}
            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"Episode save error: {e}")
            import traceback
            traceback.print_exc()
            response = {"status": "save_error", "message": f"Save error: {e}"}
            await websocket.send(json.dumps(response))

    async def discard_episode(self, websocket):
        """Discard the episode (keeps the dataset and resources)."""
        try:
            await self.stop_recording()
            if self.current_dataset is None:
                response = {"status": "discard_error", "message": "No dataset exists"}
                await websocket.send(json.dumps(response))
                return
            if getattr(self.current_dataset, "episode_buffer", None) is not None:
                self.current_dataset.clear_episode_buffer()
            await self._prepare_next_episode()
            print("Episode discarded")
            response = {"status": "discard_complete", "message": "Episode discarded. Ready for the next episode"}
            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"Episode discard error: {e}")
            import traceback
            traceback.print_exc()
            response = {"status": "discard_error", "message": f"Discard error: {e}"}
            await websocket.send(json.dumps(response))

    async def _prepare_next_episode(self):
        """Prepare for recording the next episode."""
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
            print("Ready to record the next episode")

    async def _full_cleanup_recording(self):
        """Fully clean up recording-related resources."""
        await self.stop_recording()
        if self.video_encoding_manager:
            try:
                self.video_encoding_manager.__exit__(None, None, None)
            except Exception as e:
                print(f"VideoEncodingManager exit error: {e}")
            self.video_encoding_manager = None
        elif self.current_dataset:
            try:
                self.current_dataset.finalize()
            except Exception as e:
                print(f"Dataset finalize error: {e}")
        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except Exception as e:
                print(f"Camera disconnect error: {e}")
        self.cameras = {}
        self._stop_sound_runtime()
        if self.robot:
            self.robot.cameras = {}
        self.current_dataset = None
        self.recording_start_time = None
        self.previous_recorded_action_state = None
        self.recording_ready = False

    async def _finish_dataset_and_shutdown_robot(self):
        """After reaching the target episode count, finalize the dataset and shut down the robot."""
        print(f"Reached {EPISODE_NUM} episodes. Finalizing the dataset and shutting down the robot.")
        await self._full_cleanup_recording()
        await self.cleanup_connection()
        if self.robot_connected and self.robot:
            try:
                await self.robot.disconnect()
                print("Robot disconnected")
            except Exception as e:
                print(f"Robot disconnect error: {e}")
            finally:
                self.robot_connected = False

    def _capture_latest_observation(self) -> dict:
        """Grab the latest camera frames without stopping async control."""
        if self.robot is None:
            raise RuntimeError("Robot is not initialized")

        obs = {}
        joint_state = self.robot.old_action.copy()
        for name, camera in self.cameras.items():
            try:
                frame = camera.read_latest(max_age_ms=self.CAMERA_MAX_FRAME_AGE_MS)
            except Exception:
                # Only if no latest frame yet, fall back to waiting for a new frame
                frame = camera.async_read(timeout_ms=self.CAMERA_MAX_FRAME_AGE_MS)
            if self.soundreal_enabled:
                obs[name] = preprocess_soundreal_camera_frame(name, frame)
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
        """Return the observation.state joint values from obs as an array."""
        return np.array([obs[joint_name] for joint_name in self.state_names], dtype=np.float32)

    def _overwrite_observation_state(self, obs: dict, state: np.ndarray) -> None:
        """Overwrite the observation.state joint values in obs with the given values."""
        for idx, joint_name in enumerate(self.state_names):
            obs[joint_name] = float(state[idx])

    def _record_frame_sync(self) -> None:
        """Build one frame of observations and write it to the dataset in a worker thread."""
        if self.current_dataset is None:
            raise RuntimeError("Dataset is not initialized")

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
        """Record images and joint angles at 30 FPS."""
        print("Episode recording loop ready. Waiting for the first action...")
        frame_count = 0
        while not self.is_recording and self.recording_ready:
            await asyncio.sleep(0.01)
        if not self.is_recording:
            print("Recording was cancelled")
            return
        print("Recording started!")
        try:
            while self.is_recording:
                start_time = time.perf_counter()
                await asyncio.to_thread(self._record_frame_sync)
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - self.recording_start_time
                    print(f"Recording... {frame_count} frames ({elapsed_time:.1f}s)")
                if time.time() - self.recording_start_time >= self.EPISODE_MAX_TIME_S:
                    print(f"Reached the maximum recording time ({self.EPISODE_MAX_TIME_S}s)")
                    break
                elapsed = time.perf_counter() - start_time
                if elapsed > (1.0 / self.control_frequency):
                    print(f"Recording frame processing is slow: {elapsed * 1000:.1f} ms")
                sleep_duration = 1.0 / self.DATASET_FPS - elapsed
                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError:
            print("Recording loop was cancelled")
        except Exception as e:
            print(f"Recording loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"Episode recording loop ended (total {frame_count} frames)")

    async def websocket_handler(self, websocket):
        print(f"WebSocket connected: {websocket.remote_address}")
        # If the robot is disconnected on a new connection, try to reconnect
        if not self.robot_connected:
            print("Attempting to reconnect the robot...")
            await self.initialize_robot()
        try:
            print("Waiting for a message from Unity...")
            message = await websocket.recv()
            data = json.loads(message)
            self.unity_joint_port = data.get('joint_send_port')
            self.is_connected = True
            self.start_udp_communication()
            await asyncio.sleep(0.5)
            response = {"status": "connected", "message": "Connection info received"}
            await websocket.send(json.dumps(response))
            await self.handle_websocket_messages(websocket)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection was closed by the client")
        except Exception as e:
            print(f"WebSocket error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("WebSocket connection ended")
            await self.cleanup_connection()

    async def handle_websocket_messages(self, websocket):
        try:
            async for message in websocket:
                print(f"WebSocket message received: {message}")
                try:
                    data = json.loads(message)
                    command = data.get('command')
                    if command == 'reset_robot':
                        print("Received robot reset request")
                        await self.handle_reset_request(websocket)
                    elif command == 'save_data':
                        print("Received data save request")
                        await self.save_episode(websocket)
                    elif command == 'discard_data':
                        print("Received data discard request")
                        await self.discard_episode(websocket)
                    elif command == 'recording':
                        print("Received recording request")
                        await self.start_recording(websocket)
                    elif command == 'teleoperation':
                        print("Received teleoperation request")
                        if self.is_recording or self.recording_ready:
                            await self._full_cleanup_recording()
                            print("Stopped recording and switched to teleoperation mode")
                        response = {"status": "teleoperation_mode", "message": "Switched to teleoperation mode"}
                        await websocket.send(json.dumps(response))
                    else:
                        print(f"Unknown command: {command}")
                except json.JSONDecodeError:
                    print(f"JSON parse error: {message}")
                except Exception as e:
                    print(f"Message handling error: {e}")
                    import traceback
                    traceback.print_exc()
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection was closed (while receiving messages)")
        except Exception as e:
            print(f"WebSocket message receive error: {e}")

    async def handle_reset_request(self, websocket):
        try:
            print("Starting robot reset...")
            if not self.robot_connected:
                response = {"status": "reset_error", "message": "Robot is not connected"}
                await websocket.send(json.dumps(response))
                return
            self.reset_in_progress.set()
            async with self.robot_lock:  # Exclusive robot control
                try:
                    print("Sending stop signal to control task...")
                    self.stop_event.set()
                    self.stop_threads = True
                    if self.robot_control_task and not self.robot_control_task.done():
                        print("Waiting for the robot control task to stop...")
                        try:
                            await asyncio.wait_for(self.robot_control_task, timeout=2.0)
                        except asyncio.TimeoutError:
                            print("Warning: the robot control task could not be stopped")
                            self.robot_control_task.cancel()
                    home_action = self._make_right_arm_home_action()
                    await self.robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
                    await asyncio.sleep(2.0)
                    home_action = self._make_right_arm_only_action(np.zeros_like(self.robot.old_action))
                    await self.robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
                    await asyncio.sleep(1.0)
                    print("Moved to home position")
                finally:
                    with self.action_lock:
                        self.latest_action = None
                    self.stop_event.clear()
                    self.stop_threads = False
                    if self.is_receiving_joints:
                        print("Restarting the UDP receiver thread and robot control task...")
                        if not self.joint_thread or not self.joint_thread.is_alive():
                            print("Restarting the UDP receiver thread...")
                            self.joint_thread = threading.Thread(target=self.joint_receiver_thread)
                            self.joint_thread.daemon = True
                            self.joint_thread.start()
                        await self.start_robot_control_task()
                    if self.recording_ready and not self.is_recording:
                        await self._prepare_next_episode()
                    self.reset_in_progress.clear()
            print("Robot reset completed")
            response = {"status": "reset_complete", "message": "Robot reset completed"}
            await websocket.send(json.dumps(response))
        except Exception as e:
            self.reset_in_progress.clear()
            print(f"Reset error: {e}")
            import traceback
            traceback.print_exc()
            error_response = {"status": "reset_error", "message": f"An error occurred during reset: {e}"}
            await websocket.send(json.dumps(error_response))

    def start_udp_communication(self):
        if self.unity_joint_port:
            self.stop_threads = False
            self.is_receiving_joints = True
            self.joint_thread = threading.Thread(target=self.joint_receiver_thread)
            self.joint_thread.daemon = True
            self.joint_thread.start()
            asyncio.create_task(self.start_robot_control_task())
            print("Started the UDP communication thread and async robot control task")

    async def start_robot_control_task(self):
        if self.robot_connected:
            if self.robot_control_task and not self.robot_control_task.done():
                print("The robot control task is already running")
                return
            self.robot_control_task = asyncio.create_task(self.robot_control_worker())
            print("Started a new robot control task")

    async def robot_control_worker(self):
        print("Robot control worker started")
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
                    print("Recorded the first action. Controlling with relative limits until it stabilizes.")
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
                    #     f"Detected a sudden target change, keeping relative limits "
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
            print("Robot control worker was cancelled")
        finally:
            print("Robot control worker ended")

    def joint_receiver_thread(self):
        print(f"Joint angle receiver started: port {self.unity_joint_port}")
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
                    elif len(data) >= 57:  # 1 byte (mode) + 56 bytes (14 float32 values)
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
                        print(f"Joint angle receive error: {e}")
                    break
        finally:
            sock.close()
            print("Joint angle receiver ended")

    async def cleanup_connection(self):
        print("Starting WebSocket connection cleanup...")
        self.is_connected = False
        self.is_receiving_joints = False
        self.stop_threads = True
        self.stop_event.set()
        if self.robot_control_task and not self.robot_control_task.done():
            print("Shutting down the robot control task...")
            self.robot_control_task.cancel()
            try:
                await asyncio.wait_for(self.robot_control_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self.robot_control_task = None
        if self.joint_thread and self.joint_thread.is_alive():
            print("Shutting down the UDP receiver thread...")
            self.joint_thread.join(timeout=2)
        self.joint_thread = None
        with self.action_lock:
            self.latest_action = None
        if self.robot_connected and self.robot:
            try:
                print("Returning the robot to its home position...")
                home_action = self._make_right_arm_home_action()
                await self.robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
                await asyncio.sleep(2.0)
                home_action = self._make_right_arm_only_action(np.zeros_like(self.robot.old_action))
                await self.robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
                await asyncio.sleep(1.0)
                print("Robot returned to home position")
            except Exception as e:
                print(f"Home position return error: {e}")
        print("WebSocket connection cleanup completed")

    async def cleanup(self):
        if self.is_recording or self.recording_ready or self.current_dataset is not None:
            print("Finalizing the dataset being recorded...")
            await self._full_cleanup_recording()
        await self.cleanup_connection()
        if self.robot_connected and self.robot:
            try:
                await self.robot.disconnect()
                print("Robot disconnected")
                self.robot_connected = False
            except Exception as e:
                print(f"Robot disconnect error: {e}")
        print("Full cleanup completed")

    async def start_server(self):
        print(f"Starting WebSocket server: port {self.websocket_port}")
        await self.initialize_robot()
        try:
            async with websockets.serve(self.websocket_handler, "0.0.0.0", self.websocket_port):
                print("Server started. Waiting for a connection from Unity...")
                await asyncio.Future()
        except KeyboardInterrupt:
            print("\nStopping the server...")
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            await self.cleanup()

if __name__ == "__main__":
    node = RobotCommunicationNode()
    print("Starting the Unity-Iloha communication server...")
    asyncio.run(node.start_server())

# uv run src/iloha_server.py
