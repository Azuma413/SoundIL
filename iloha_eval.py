#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習済みのPolicyを読み込んでIlohaロボットを動かすスクリプト

使用例:
python src/lerobot/my_aloha_eval.py \
    --policy_path outputs/train/act-aloha-dataset-0/checkpoints/last/pretrained_model \
    --dataset_path datasets/aloha-dataset-0 \
    --episode_time_s 60 \
    --num_episodes 1 \
    --save_data
"""

import argparse
import asyncio
import time
from pathlib import Path
from typing import Optional
import numpy as np

from lerobot.robots.iloha import Iloha, IlohaConfig, JOINT_NAMES
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras import make_cameras_from_configs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.processor.rename_processor import rename_stats


# カメラ設定（my_aloha_server.pyと同じ）
CAMERA_CONFIGS = {
    "cam_high": {"serial_number_or_name": "029522250086", "width": 640, "height": 480, "fps": 30},
    "cam_left_wrist": {"serial_number_or_name": "341522301205", "width": 640, "height": 480, "fps": 30},
    "cam_right_wrist": {"serial_number_or_name": "146222252104", "width": 640, "height": 480, "fps": 30}
}


async def reset_robot_to_home(robot: Iloha, init=True):
    """
    ロボットを初期位置に戻す（my_aloha_server.pyのhandle_reset_requestと同じロジック）
    """
    print("ロボットを初期位置に戻しています...")
    
    # 1. グリッパーを開く（0.0に設定）
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
    home_action[3:7] = 0.0   # 左グリッパー関連
    home_action[10:14] = 0.0  # 右グリッパー関連
    await robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
    await asyncio.sleep(2.0)
    
    # 2. 全関節を0に戻す
    home_action = np.zeros_like(home_action)
    await robot.async_send_action(home_action, use_relative=False, use_filter=False, use_unwrap=False)
    await asyncio.sleep(1.0)
    
    print("初期位置復帰完了")


def initialize_cameras() -> dict:
    """カメラを初期化して辞書で返す"""
    try:
        camera_configs = {}
        for name, config_dict in CAMERA_CONFIGS.items():
            camera_configs[name] = RealSenseCameraConfig(**config_dict)
        
        cameras = make_cameras_from_configs(camera_configs)
        
        for name, camera in cameras.items():
            print(f"{name} を接続中...")
            camera.connect(warmup=True)
            time.sleep(0.2)
        
        print(f"{len(cameras)}台のカメラを初期化しました")
        return cameras
    except Exception as e:
        print(f"カメラ初期化エラー: {e}")
        return {}


def get_next_dataset_number(root: Path, prefix: str = "aloha-eval-") -> int:
    """既存のデータセット番号を確認し、次の番号を返す"""
    if not root.exists():
        return 0
    
    existing_nums = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith(prefix):
            try:
                num = int(path.name.split("-")[-1])
                existing_nums.append(num)
            except ValueError:
                continue
    
    return max(existing_nums) + 1 if existing_nums else 0


async def evaluation_loop(
    robot: Iloha,
    policy,
    preprocessor,
    postprocessor,
    device,
    fps: int,
    episode_time_s: float,
    task: str,
    ds_features: dict,
    dataset: Optional[LeRobotDataset] = None,
    display_data: bool = False,
):
    """
    評価ループ: ポリシーからアクションを予測してロボットを制御
    """
    print(f"評価ループ開始（{episode_time_s}秒間）")
    
    frame_count = 0
    start_episode_t = time.perf_counter()
    
    while True:
        start_loop_t = time.perf_counter()
        
        # 時間チェック
        elapsed = time.perf_counter() - start_episode_t
        if elapsed >= episode_time_s:
            print(f"エピソード時間（{episode_time_s}秒）に達しました")
            break
        
        # 1. ロボットから観測を取得
        obs = robot.get_observation()
        
        if frame_count == 0:
            print(f"観測データのキー: {list(obs.keys())}")
        
        # 2. データセット形式のフレームを構築
        observation_frame = build_dataset_frame(
            ds_features, 
            obs, 
            prefix="observation"
        )
        
        if frame_count == 0:
            print(f"observation_frameのキー: {list(observation_frame.keys())}")
        
        # 3. ポリシーでアクションを予測
        try:
            action_tensor = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task,
                robot_type=robot.name,
            )
            
            # 4. Tensorをnumpy配列に変換し、バッチ次元を削除
            if isinstance(action_tensor, dict):
                # 辞書形式の場合（一部のポリシー）
                action_array = np.array([action_tensor[name] for name in JOINT_NAMES], dtype=np.float32)
                action_values = action_tensor
            else:
                # Tensor形式の場合
                action_array = action_tensor.squeeze(0).cpu().numpy()  # (1, 14) -> (14,)
                # 辞書形式に変換（後の処理のため）
                action_values = {name: float(action_array[i]) for i, name in enumerate(JOINT_NAMES)}
            
            if frame_count == 0:
                print(f"予測されたアクション形状: {action_array.shape}")
                print(f"アクション値（最初の3要素）: {action_array[:3]}")
        
        except Exception as e:
            print(f"アクション予測エラー: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # 6. ロボットにアクションを送信
        await robot.async_send_action(action_array)
        
        # 7. データセットに保存（オプション）
        if dataset is not None:
            action_frame = build_dataset_frame(
                ds_features, 
                {name: action_values[name] for name in JOINT_NAMES}, 
                prefix="action"
            )
            frame = {**observation_frame, **action_frame, "task": task}
            dataset.add_frame(frame)
        
        # 8. 可視化（オプション）
        if display_data:
            log_rerun_data(observation=obs, action=action_values)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"フレーム: {frame_count}, 経過時間: {elapsed:.1f}秒")
        
        # 9. FPS制御
        dt_s = time.perf_counter() - start_loop_t
        sleep_duration = 1.0 / fps - dt_s
        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)
    
    print(f"評価ループ終了（合計{frame_count}フレーム）")


async def main(args):
    init_logging()
    
    # 1. ロボットの初期化と接続
    print("=" * 60)
    print("ロボットを初期化中...")
    config = IlohaConfig(
        right_dynamixel_port="/dev/ttyUSB1",
        right_robstride_port="/dev/ttyUSB0",
        left_robstride_port="/dev/ttyUSB2",
        left_dynamixel_port="/dev/ttyUSB3",
        max_relative_target_1=0.03,
        max_relative_target_2=0.01,
        max_relative_target_3=0.01,
        max_relative_target_4=0.03,
        max_relative_target_5=0.01,
        max_relative_target_6=0.03,
        current_limit_gripper_R=0.3,
        current_limit_gripper_L=0.3,
    )
    robot = Iloha(config, debug=False)
    await robot.connect()
    print("ロボット接続完了")
    
    # 2. 初期位置に戻す
    await reset_robot_to_home(robot)
    
    # 3. カメラの初期化
    print("=" * 60)
    print("カメラを初期化中...")
    cameras = initialize_cameras()
    if not cameras:
        print("エラー: カメラの初期化に失敗しました")
        await robot.disconnect()
        return
    robot.cameras = cameras
    
    # 4. ポリシーとプロセッサの読み込み
    print("=" * 60)
    print(f"ポリシーを読み込み中: {args.policy_path}")
    
    # データセットからメタデータとstatsを読み込み
    print(f"データセット読み込み中: {args.dataset_path}")
    dataset_for_stats = LeRobotDataset(args.dataset_path, root=args.dataset_path)
    
    # ポリシー設定を読み込み
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = "cuda" if args.device == "cuda" else "cpu"
    
    # ポリシーを作成
    policy = make_policy(policy_cfg, ds_meta=dataset_for_stats.meta)
    
    # プロセッサを作成
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        dataset_stats=rename_stats(dataset_for_stats.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
        },
    )
    
    device = get_safe_torch_device(policy_cfg.device)
    print(f"ポリシー読み込み完了（デバイス: {device}）")
    
    # 5. データセットの作成（保存する場合）
    dataset = None
    video_encoding_manager = None
    
    if args.save_data:
        print("=" * 60)
        print("データセット作成中...")
        
        dataset_root = Path(args.output_root)
        dataset_num = get_next_dataset_number(dataset_root, prefix="aloha-eval-")
        dataset_name = f"aloha-eval-{dataset_num}"
        repo_id = f"local/{dataset_name}"
        dataset_path = dataset_root / dataset_name
        
        # データセット特徴量の定義
        dataset_features = {
            "observation.state": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
            "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        }
        for key in CAMERA_CONFIGS.keys():
            dataset_features[f"observation.images.{key}"] = {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ("height", "width", "channels")
            }
        
        dataset = LeRobotDataset.create(
            repo_id,
            args.fps,
            root=dataset_path,
            robot_type="my_aloha",
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * len(cameras),
            video_backend="pyav",
        )
        
        video_encoding_manager = VideoEncodingManager(dataset)
        video_encoding_manager.__enter__()
        
        print(f"データセット作成完了: {repo_id}")
    
    # 6. 可視化の初期化（オプション）
    if args.display_data:
        init_rerun(session_name="evaluation")
    
    # 7. エピソードループ
    print("=" * 60)
    print(f"{args.num_episodes}エピソードの評価を開始します")
    
    try:
        for episode_idx in range(args.num_episodes):
            print(f"\n--- エピソード {episode_idx + 1}/{args.num_episodes} ---")
            
            # ポリシーとプロセッサをリセット
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            
            # 評価ループを実行
            await evaluation_loop(
                robot=robot,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=device,
                fps=args.fps,
                episode_time_s=args.episode_time_s,
                task=args.task,
                ds_features=dataset_for_stats.features,
                dataset=dataset,
                display_data=args.display_data,
            )
            
            # エピソードを保存
            if dataset is not None:
                dataset.save_episode()
                print(f"エピソード {episode_idx + 1} を保存しました")
            
            # 次のエピソードのためにロボットを初期位置に戻す
            if episode_idx < args.num_episodes - 1:
                print(f"\n次のエピソードのためにロボットをリセットします...")
                await reset_robot_to_home(robot, init=False)
                await asyncio.sleep(2.0)  # リセット後の待機時間
    
    except KeyboardInterrupt:
        print("\n中断されました")
    
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 8. クリーンアップ
        print("=" * 60)
        print("クリーンアップ中...")
        
        # データセットのクリーンアップ
        if video_encoding_manager:
            video_encoding_manager.__exit__(None, None, None)
        if dataset:
            dataset.finalize()
            print("データセットを終了しました")
        
        # カメラの切断
        for name, camera in cameras.items():
            try:
                camera.disconnect()
                print(f"{name} を切断しました")
            except Exception as e:
                print(f"{name} 切断エラー: {e}")
        
        # ロボットを初期位置に戻して切断
        await reset_robot_to_home(robot, init=False)
        await robot.disconnect()
        print("ロボット切断完了")
        
        print("=" * 60)
        print("評価スクリプト終了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="学習済みPolicyでIlohaロボットを評価")
    
    # 必須引数
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="学習済みポリシーのパス（例: outputs/train/act-aloha-dataset-0/checkpoints/last/pretrained_model）"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="統計情報を取得するデータセットのパス（例: datasets/aloha-dataset-0）"
    )
    
    # オプション引数
    parser.add_argument(
        "--output_root",
        type=str,
        default="datasets",
        help="保存先ルートディレクトリ（デフォルト: datasets）"
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="観測データを保存する"
    )
    parser.add_argument(
        "--episode_time_s",
        type=float,
        default=60.0,
        help="1エピソードの実行時間（秒）（デフォルト: 60）"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="実行するエピソード数（デフォルト: 1）"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="制御周波数（Hz）（デフォルト: 30）"
    )
    parser.add_argument(
        "--display_data",
        action="store_true",
        help="rerunでリアルタイム可視化"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="do something",
        help="タスク名（デフォルト: do something）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="推論デバイス（デフォルト: cuda）"
    )
    
    args = parser.parse_args()
    
    # 非同期でメイン関数を実行
    asyncio.run(main(args))

# 使用例:
# uv run src/lerobot/my_aloha_eval.py \
#     --policy_path outputs/train/act-aloha-dataset-0/checkpoints/last/pretrained_model \
#     --dataset_path datasets/aloha-dataset-0 \
#     --episode_time_s 60 \
#     --num_episodes 1 \
#     --save_data
