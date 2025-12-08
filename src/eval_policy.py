from pathlib import Path
import imageio
import numpy as np
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.control_utils import predict_action
from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.constants import OBS_STR
from lerobot.utils.utils import get_safe_torch_device
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.genesis_env import GenesisEnv

def process_image_for_video(image_array, target_height, target_width):
    """Process an image array for video recording, ensuring it's HWC, RGB, uint8."""
    if image_array is None:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    if image_array.ndim == 2:  # Grayscale (H, W)
        image_array = np.stack((image_array,) * 3, axis=-1) # Convert to (H, W, 3)
    elif image_array.ndim == 3 and image_array.shape[0] == 3: # CHW
        image_array = image_array.transpose(1, 2, 0) # Convert to (H, W, C)
    if image_array.shape[2] == 1:  # Grayscale with channel dim
        image_array = np.concatenate([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = image_array[..., :3] # Keep only RGB
    if image_array.dtype != np.uint8:
        if np.issubdtype(image_array.dtype, np.floating):
            image_array = (image_array * 255).clip(0, 255)
        image_array = image_array.astype(np.uint8)
    if image_array.shape[0] != target_height or image_array.shape[1] != target_width:
        print(f"Warning: Image shape {image_array.shape} mismatch with target {target_height}x{target_width}. Using black frame.")
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    return image_array

def combine_frames(np_obs, observation_height, observation_width):
    front_img = np_obs.get("observation.images.front")
    side_img = np_obs.get("observation.images.side")
    sound_img1 = np_obs.get("observation.images.sound0")
    sound_img2 = np_obs.get("observation.images.sound1")
    front_video_img = process_image_for_video(front_img, observation_height, observation_width)
    side_video_img = process_image_for_video(side_img, observation_height, observation_width)
    sound_video_img1 = process_image_for_video(sound_img1, observation_height, observation_width)
    sound_video_img2 = process_image_for_video(sound_img2, observation_height, observation_width)
    combined_h = observation_height * 2
    combined_w = observation_width * 2
    combined_frame = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    combined_frame[0:observation_height, 0:observation_width] = front_video_img
    combined_frame[0:observation_height, observation_width:combined_w] = side_video_img
    combined_frame[observation_height:combined_h, 0:observation_width] = sound_video_img1
    combined_frame[observation_height:combined_h, observation_width:combined_w] = sound_video_img2
    return combined_frame

def main(training_name, observation_height, observation_width, episode_num, show_viewer, checkpoint_step="last"):
    output_directory = Path(f"outputs/eval/{training_name}_{checkpoint_step}")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    pretrained_policy_path = Path(f"outputs/train/{training_name}/checkpoints/{checkpoint_step}/pretrained_model")
    if not pretrained_policy_path.exists():
        print(f"Error: Pretrained model not found at {pretrained_policy_path}")
        return
    print(f"Loading policy from: {pretrained_policy_path}")
    model_type = training_name.split("_")[0]
    if model_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    elif model_type == "act":
        policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    elif model_type == "pi0":
        policy = PI0Policy.from_pretrained(pretrained_policy_path)
    else:
        print(f"Error: Unknown model type: {model_type}")
        return
    policy.to(device)
    policy.eval()
    task_name = training_name.split("_")[1]
    dataset_name = f"{task_name}_{training_name.split('_')[-1]}"
    print(f"Detected task name: {task_name}")
    # Load dataset to get statistics for normalization
    dataset_path = Path(f"datasets/{dataset_name}")
    dataset_path = dataset_path.resolve()
    print(f"Loading dataset from: {dataset_path}")
    dataset = LeRobotDataset(str(dataset_path))
    
    # Create preprocessor and postprocessor
    print("Creating preprocessor and postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(pretrained_policy_path),
        dataset_stats=dataset.meta.stats,
    )
    
    env = GenesisEnv(task=task_name, observation_height=observation_height, observation_width=observation_width, show_viewer=show_viewer)
    print("Policy Input Features:", policy.config.input_features)
    print("Environment Observation Space:", env.observation_space)
    print("Policy Output Features:", policy.config.output_features)
    print("Environment Action Space:", env.action_space)
    success_num = 0
    all_actions = []  # 全エピソードのactionを保存
    combined_video_h = observation_height * 2
    combined_video_w = observation_width * 2
    for ep in range(episode_num):
        print(f"\n=== Episode {ep+1} ===")
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        numpy_observation, _ = env.reset()
        rewards = []
        frames = []  # Stores combined frames
        combined_frame = combine_frames(numpy_observation, observation_height, observation_width)
        frames.append(combined_frame)
        step = 0
        done = False
        while not done:
            # Convert observation keys to format expected by build_dataset_frame
            # Environment returns keys like "observation.images.front"
            # build_dataset_frame expects short keys like "front" for images
            converted_obs = {}
            for key, value in numpy_observation.items():
                if key.startswith("observation.images."):
                    # Extract short name (e.g., "front" from "observation.images.front")
                    short_key = key.replace("observation.images.", "")
                    # Make a copy to ensure contiguous memory layout (avoid negative stride issues)
                    converted_obs[short_key] = value.copy() if isinstance(value, np.ndarray) else value
                elif key == "observation.state":
                    # For state, extract individual joint values
                    # build_dataset_frame expects individual joint names as keys
                    if "observation.state" in dataset.features:
                        for i, name in enumerate(dataset.features["observation.state"]["names"]):
                            converted_obs[name] = value[i]
                else:
                    converted_obs[key] = value.copy() if isinstance(value, np.ndarray) else value
            
            # Build observation frame in dataset format
            observation_frame = build_dataset_frame(dataset.features, converted_obs, prefix=OBS_STR)
            
            # Use predict_action to apply preprocessor, policy, and postprocessor
            action_dict = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=task_name,
                robot_type=None,
            )
            
            # Extract action tensor from the returned dictionary
            if isinstance(action_dict, dict) and 'action' in action_dict:
                action_tensor = action_dict['action']
            else:
                action_tensor = action_dict
            
            # Convert to numpy for environment
            numpy_action = action_tensor.squeeze(0).cpu().numpy()
            all_actions.append(numpy_action)  # actionを記録
            numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
            # print(f"Step: {step}, Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")
            rewards.append(reward)
            current_combined_frame = combine_frames(numpy_observation, observation_height, observation_width)
            frames.append(current_combined_frame)
            done = terminated or truncated
            step += 1
            if reward > 0: # 成功したら早期終了
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
    success_rate = (success_num / episode_num) * 100
    print(f"Success rate: {success_num}/{episode_num} ({success_rate:.2f}%)")
    
    # actionの統計情報を計算
    action_stats = None
    if all_actions:
        all_actions_array = np.array(all_actions)  # shape: (total_steps, action_dim)
        action_stats = {
            'min': np.min(all_actions_array, axis=0),
            'max': np.max(all_actions_array, axis=0),
            'mean': np.mean(all_actions_array, axis=0),
            'std': np.std(all_actions_array, axis=0)
        }
        print("\nAction Statistics:")
        print(f"  Min:  {action_stats['min']}")
        print(f"  Max:  {action_stats['max']}")
        print(f"  Mean: {action_stats['mean']}")
        print(f"  Std:  {action_stats['std']}")
    
    # success_rate.txtに書き込み
    success_rate_file = output_directory / "success_rate.txt"
    with open(success_rate_file, "w") as f:
        f.write(f"Success rate: {success_num}/{episode_num} ({success_rate:.2f}%)\n")
        if action_stats is not None:
            f.write(f"\nAction Statistics:\n")
            f.write(f"  Min:  {action_stats['min']}\n")
            f.write(f"  Max:  {action_stats['max']}\n")
            f.write(f"  Mean: {action_stats['mean']}\n")
            f.write(f"  Std:  {action_stats['std']}\n")

if __name__ == "__main__":
    # 評価したい学習済みモデルの名前を指定
    # outputs/train/<training_name>/checkpoints/<checkpoint_step>
    training_name = "act_soundDiff-m3-fo-so_0"
    checkpoint_step = "100000"
    main(
        training_name=training_name,
        observation_height=224,
        observation_width=224,
        episode_num=1,
        show_viewer=False,
        checkpoint_step=checkpoint_step,
    )
