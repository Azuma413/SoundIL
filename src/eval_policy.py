from pathlib import Path
import imageio
import numpy as np
import torch
import genesis as gs
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# 環境変数を読み込み
load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.genesis_env import GenesisEnv
from src.notify import NotificationSystem

def process_image_for_video(image_array, target_height, target_width):
    """Process an image array for video recording, ensuring it's HWC, RGB, uint8."""
    if image_array is None:
        # Return a black frame if the image is missing
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Ensure image is HWC
    if image_array.ndim == 2:  # Grayscale (H, W)
        image_array = np.stack((image_array,) * 3, axis=-1) # Convert to (H, W, 3)
    elif image_array.ndim == 3 and image_array.shape[0] == 3: # CHW
        image_array = image_array.transpose(1, 2, 0) # Convert to (H, W, C)

    # Ensure 3 channels (RGB)
    if image_array.shape[2] == 1:  # Grayscale with channel dim
        image_array = np.concatenate([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = image_array[..., :3] # Keep only RGB

    # Ensure uint8
    if image_array.dtype != np.uint8:
        if np.issubdtype(image_array.dtype, np.floating):
            image_array = (image_array * 255).clip(0, 255)
        image_array = image_array.astype(np.uint8)
    if image_array.shape[0] != target_height or image_array.shape[1] != target_width:
        print(f"Warning: Image shape {image_array.shape} mismatch with target {target_height}x{target_width}. Using black frame.")
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
    return image_array

def main(training_name, observation_height, observation_width, episode_num, show_viewer, checkpoint_step="last"):
    notifier = NotificationSystem()
    start_time = time.time()
    
    policy_list = ["act", "diffusion", "pi0", "tdmpc", "vqbet"]
    task_list = ["test", "sound", "marker_sound", "weighted_sound", "2sound", "marker_2sound", "weighted_2sound", "test_sound", "test_no_brank", "dummy"]
    output_directory = Path(f"outputs/eval/{training_name}_{checkpoint_step}")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    pretrained_policy_path = Path(f"outputs/train/{training_name}/checkpoints/{checkpoint_step}/pretrained_model")
    if not pretrained_policy_path.exists():
        print(f"Error: Pretrained model not found at {pretrained_policy_path}")
        return
    print(f"Loading policy from: {pretrained_policy_path}")
    model_type = None
    for p in policy_list:
        if p in training_name:
            model_type = p
            break
    if model_type is None:
        print(f"Error: Unknown model type in training name '{training_name}'. Expected one of {policy_list}.")
        return
    if model_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    elif model_type == "act":
        policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    elif model_type == "vqbet":
        policy = VQBeTPolicy.from_pretrained(pretrained_policy_path)
    elif model_type == "tdmpc":
        policy = TDMPCPolicy.from_pretrained(pretrained_policy_path)
    elif model_type == "pi0":
        policy = PI0Policy.from_pretrained(pretrained_policy_path)
    else:
        print(f"Error: Unknown model type: {model_type}")
        return
    policy.to(device)
    policy.eval()
    task_name = None
    for t in task_list:
        if t in training_name:
            task_name = t
    print(f"Detected task name: {task_name}")
    if task_name is None:
        print(f"Error: Unknown task name in training name '{training_name}'. Expected one of {task_list}.")
        return
    if task_name == "test_no_brank":
        env = GenesisEnv(task="test", observation_height=observation_height, observation_width=observation_width, show_viewer=show_viewer)
    else:
        env = GenesisEnv(task=task_name, observation_height=observation_height, observation_width=observation_width, show_viewer=show_viewer)
    print("Policy Input Features:", policy.config.input_features)
    print("Environment Observation Space:", env.observation_space)
    print("Policy Output Features:", policy.config.output_features)
    print("Environment Action Space:", env.action_space)
    success_num = 0

    combined_video_h = observation_height * 2
    combined_video_w = observation_width * 2

    for ep in range(episode_num):
        print(f"\n=== Episode {ep+1} ===")
        policy.reset()
        numpy_observation, _ = env.reset()
        rewards = []
        frames = [] # Stores combined frames

        # Process initial observation for video
        front_img_obs = numpy_observation.get("observation.images.front")
        side_img_obs = numpy_observation.get("observation.images.side")
        sound_img_obs = numpy_observation.get("observation.images.sound")

        front_video_img = process_image_for_video(front_img_obs, observation_height, observation_width)
        side_video_img = process_image_for_video(side_img_obs, observation_height, observation_width)
        sound_video_img = process_image_for_video(sound_img_obs, observation_height, observation_width)

        combined_frame = np.zeros((combined_video_h, combined_video_w, 3), dtype=np.uint8)
        combined_frame[0:observation_height, 0:observation_width] = front_video_img
        combined_frame[0:observation_height, observation_width:combined_video_w] = side_video_img
        sound_start_w = (combined_video_w - observation_width) // 2
        combined_frame[observation_height:combined_video_h, sound_start_w : sound_start_w + observation_width] = sound_video_img
        frames.append(combined_frame)

        step = 0
        done = False
        while not done:
            observation = {}
            for key in policy.config.input_features:
                # make_sim_dataset.pyの観測キーに合わせてマッピング
                if key == "observation.state":
                    data = numpy_observation["agent_pos"]
                    tensor_data = torch.from_numpy(data).to(torch.float32)
                    observation[key] = tensor_data.to(device).unsqueeze(0)
                elif key == "observation.images.front":
                    img = numpy_observation["observation.images.front"]
                    img = img.copy()  # 負のstride対策
                    tensor_img = torch.from_numpy(img).to(torch.float32) / 255.0
                    if tensor_img.ndim == 3 and tensor_img.shape[2] in [1, 3, 4]:
                        tensor_img = tensor_img.permute(2, 0, 1)
                    elif tensor_img.ndim == 2:
                        tensor_img = tensor_img.unsqueeze(0)
                    observation[key] = tensor_img.to(device).unsqueeze(0)
                elif key == "observation.images.side":
                    img = numpy_observation["observation.images.side"]
                    img = img.copy()  # 負のstride対策
                    tensor_img = torch.from_numpy(img).to(torch.float32) / 255.0
                    if tensor_img.ndim == 3 and tensor_img.shape[2] in [1, 3, 4]:
                        tensor_img = tensor_img.permute(2, 0, 1)
                    elif tensor_img.ndim == 2:
                        tensor_img = tensor_img.unsqueeze(0)
                    observation[key] = tensor_img.to(device).unsqueeze(0)
                elif key == "observation.images.sound":
                    img = numpy_observation["observation.images.sound"] # 画像データを取得
                    img = img.copy()  # 負のstride対策でコピーを作成
                    tensor_img = torch.from_numpy(img).to(torch.float32) / 255.0 # 0-1に正規化
                    if tensor_img.ndim == 3 and tensor_img.shape[2] in [1, 3, 4]: # HWC形式でチャンネルが1, 3, 4のいずれか
                        tensor_img = tensor_img.permute(2, 0, 1) # CHW形式に変換
                    elif tensor_img.ndim == 2:
                        tensor_img = tensor_img.unsqueeze(0) # (H, W) -> (1, H, W)に変換
                    observation[key] = tensor_img.to(device).unsqueeze(0) # バッチ次元を追加
                else:
                    print(f"Warning: Unsupported input feature '{key}'. Skipping.")
            with torch.inference_mode():
                action = policy.select_action(observation)
                if isinstance(action, dict):
                    action_tensor = action.get('action', None)
                    if action_tensor is None:
                        print("Error: Policy did not return 'action' key.")
                        break
                else:
                    action_tensor = action
            numpy_action = action_tensor.squeeze(0).cpu().numpy()
            numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
            print(f"Step: {step}, Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")
            rewards.append(reward)

            # Process current observation for video and add to frames list
            front_img_obs = numpy_observation.get("observation.images.front")
            side_img_obs = numpy_observation.get("observation.images.side")
            sound_img_obs = numpy_observation.get("observation.images.sound")

            front_video_img = process_image_for_video(front_img_obs, observation_height, observation_width)
            side_video_img = process_image_for_video(side_img_obs, observation_height, observation_width)
            sound_video_img = process_image_for_video(sound_img_obs, observation_height, observation_width)

            current_combined_frame = np.zeros((combined_video_h, combined_video_w, 3), dtype=np.uint8)
            current_combined_frame[0:observation_height, 0:observation_width] = front_video_img
            current_combined_frame[0:observation_height, observation_width:combined_video_w] = side_video_img
            sound_start_w = (combined_video_w - observation_width) // 2
            current_combined_frame[observation_height:combined_video_h, sound_start_w : sound_start_w + observation_width] = sound_video_img
            frames.append(current_combined_frame)
            done = terminated or truncated
            step += 1
            # 評価用（必要なければ消す
            if reward > 0:
                done = True
        total_reward = sum(rewards)
        print(f"Evaluation finished after {step} steps. Total reward: {total_reward:.4f}")
        if total_reward > 0:
            print("Result: Success!")
            success_num += 1
        else:
            print("Result: Failure.")
        valid_frames = [f for f in frames if f is not None and isinstance(f, np.ndarray)]
        if valid_frames:
            fps = env.metadata.get("render_fps", 30)
            video_path = output_directory / f"rollout_ep{ep+1}.mp4"
            # Ensure all frames are uint8, HWC, and have the correct combined shape
            processed_valid_frames = []
            for f_val in valid_frames:
                if f_val.shape != (combined_video_h, combined_video_w, 3):
                    print(f"Warning: Frame shape is {f_val.shape}, expected {(combined_video_h, combined_video_w, 3)}. Skipping this frame for video.")
                    continue
                if f_val.dtype != np.uint8:
                    f_val = f_val.astype(np.uint8) # Should be handled by process_image_for_video already
                processed_valid_frames.append(f_val)

            valid_frames = processed_valid_frames

            if not valid_frames:
                print("No valid frames with correct dimensions found after processing, skipping video saving.")
            else:
                first_shape = valid_frames[0].shape
                if not all(f.shape == first_shape for f in valid_frames):
                    # This check might be redundant if the loop above filters correctly, but good for safety
                    print(f"Warning: Frame shapes are inconsistent after processing. First frame: {first_shape}. Video may be corrupted.")

                try:
                    imageio.mimsave(str(video_path), np.stack(valid_frames), fps=fps, output_params=['-pix_fmt', 'yuv420p'])
                except Exception as e1:
                    print(f"Error saving with pyav plugin: {e1}. Trying default imageio plugin.")
                    try:
                        imageio.mimsave(str(video_path), np.stack(valid_frames), fps=fps)
                    except Exception as e2:
                        print(f"Error saving video with default plugin: {e2}. Video saving failed for episode {ep+1}.")
                else:
                    print(f"Video saved: {video_path}")
        else:
            print("No valid frames recorded, skipping video saving.")
    env.close()
    
    # 実行完了
    end_time = time.time()
    duration = end_time - start_time
    success_rate = (success_num / episode_num) * 100
    
    print(f"Success rate: {success_num}/{episode_num} ({success_rate:.2f}%)")
    
    # 成功率をtextファイルに保存
    success_rate_file = output_directory / "success_rate.txt"
    with open(success_rate_file, "w") as f:
        f.write(f"Success rate: {success_num}/{episode_num} ({success_rate:.2f}%)\n")
    
    # 完了通知
    if notifier:
        result_info = f"成功率: {success_num}/{episode_num} ({success_rate:.2f}%)\n"
        notifier.send_discord_message(f"お兄ちゃん！　{training_name}のポリシー評価が完了したよ！\n```\n{result_info}```")

if __name__ == "__main__":
    training_name = "act-sound-ep100_0"
    observation_height = 480
    observation_width = 640
    episode_num = 50
    show_viewer = False
    checkpoint_step = "120000"
    main(
        training_name=training_name,
        observation_height=observation_height,
        observation_width=observation_width,
        episode_num=episode_num,
        show_viewer=show_viewer,
        checkpoint_step=checkpoint_step,
    )
