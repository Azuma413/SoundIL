import argparse
from pathlib import Path
import imageio
import numpy as np
import torch
import cv2
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.control_utils import predict_action
from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.constants import OBS_STR
from lerobot.utils.utils import get_safe_torch_device
import time
from s2a2.env.genesis_env import GenesisEnv

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

def normalize_to_uint8(image_array):
    """Normalize a 2D or 3D array to uint8 for visualization."""
    if image_array is None:
        return None
    image_array = np.asarray(image_array, dtype=np.float32)
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    if max_val > min_val:
        normalized = (image_array - min_val) / (max_val - min_val) * 255.0
    else:
        normalized = np.zeros_like(image_array, dtype=np.float32)
    return np.clip(normalized, 0, 255).astype(np.uint8)

def render_combined_soundmap_heatmap(sound_img0, sound_img1, target_height, target_width):
    """Merge sound map channels and render a heatmap from their sum."""
    if sound_img0 is None and sound_img1 is None:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    sound0 = process_image_for_video(sound_img0, target_height, target_width)
    sound1 = process_image_for_video(sound_img1, target_height, target_width)
    full_sound_map = np.concatenate([sound0, sound1], axis=2).astype(np.float32)
    summed_map = np.sum(full_sound_map, axis=2)
    summed_map_uint8 = normalize_to_uint8(summed_map)
    heatmap_bgr = cv2.applyColorMap(summed_map_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

def render_grayscale_spectrogram(spec_img, target_height, target_width):
    """Render a spectrogram image in grayscale."""
    if spec_img is None:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    spec_video_img = process_image_for_video(spec_img, target_height, target_width)
    grayscale_spec = np.mean(spec_video_img.astype(np.float32), axis=2)
    grayscale_spec_uint8 = normalize_to_uint8(grayscale_spec)
    return np.stack([grayscale_spec_uint8] * 3, axis=-1)

def combine_frames(np_obs, observation_height, observation_width):
    front_img = np_obs.get("observation.images.front")
    side_img = np_obs.get("observation.images.side")
    sound_img0 = np_obs.get("observation.images.sound0")
    sound_img1 = np_obs.get("observation.images.sound1")
    spec_img = np_obs.get("observation.images.spec")
    front_video_img = process_image_for_video(front_img, observation_height, observation_width)
    side_video_img = process_image_for_video(side_img, observation_height, observation_width)
    sound_heatmap_img = render_combined_soundmap_heatmap(sound_img0, sound_img1, observation_height, observation_width)
    spectrogram_img = render_grayscale_spectrogram(spec_img, observation_height, observation_width)
    combined_h = observation_height * 2
    combined_w = observation_width * 2
    combined_frame = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    combined_frame[0:observation_height, 0:observation_width] = front_video_img
    combined_frame[0:observation_height, observation_width:combined_w] = side_video_img
    combined_frame[observation_height:combined_h, 0:observation_width] = sound_heatmap_img
    combined_frame[observation_height:combined_h, observation_width:combined_w] = spectrogram_img
    return combined_frame

def fill_missing_image_observations(converted_obs, dataset_features):
    """Fill missing image inputs expected by the dataset with zero images."""
    for key, feature in dataset_features.items():
        if not key.startswith("observation.images."):
            continue
        if feature["dtype"] not in ["image", "video"]:
            continue

        short_key = key.removeprefix("observation.images.")
        if short_key in converted_obs:
            continue

        shape = tuple(feature["shape"])
        converted_obs[short_key] = np.zeros(shape, dtype=np.uint8)

def normalize_checkpoint_step(checkpoint_step):
    """Pad numeric checkpoint steps to six digits while preserving names like 'last'."""
    checkpoint_step_str = str(checkpoint_step)
    if checkpoint_step_str.isdigit():
        return checkpoint_step_str.zfill(6)
    return checkpoint_step_str

def infer_dataset_name(training_name):
    """Infer dataset name from a training run name.

    Supported patterns:
    - <policy>_<dataset>
    - <policy>_<dataset>_seed<seed>
    """
    parts = training_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Could not infer dataset name from training name: {training_name}")

    dataset_parts = parts[1:]
    if dataset_parts and dataset_parts[-1].startswith("seed"):
        dataset_parts = dataset_parts[:-1]

    if not dataset_parts:
        raise ValueError(f"Could not infer dataset name from training name: {training_name}")

    return "_".join(dataset_parts)

def infer_task_name(dataset_name):
    """Infer Genesis task name from dataset name."""
    if "_" not in dataset_name:
        return dataset_name
    return dataset_name.rsplit("_", 1)[0]

def main(training_name, observation_height, observation_width, episode_num, show_viewer, checkpoint_step="last", dataset_name=None):
    checkpoint_step = normalize_checkpoint_step(checkpoint_step)
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
        policy.config.n_action_steps = 1
        policy.config.temporal_ensemble_coeff = 0.01
        policy.temporal_ensembler = ACTTemporalEnsembler(
            policy.config.temporal_ensemble_coeff, policy.config.chunk_size
        )
        policy.reset()
        print(
            "Overriding ACT eval config: "
            f"n_action_steps={policy.config.n_action_steps}, "
            f"temporal_ensemble_coeff={policy.config.temporal_ensemble_coeff}"
        )
    elif model_type == "pi0":
        policy = PI0Policy.from_pretrained(pretrained_policy_path)
    elif model_type == "vqbet":
        policy = VQBeTPolicy.from_pretrained(pretrained_policy_path)
    else:
        print(f"Error: Unknown model type: {model_type}")
        return
    policy.to(device)
    policy.eval()
    dataset_name = dataset_name or infer_dataset_name(training_name)
    task_name = infer_task_name(dataset_name)
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
    
    env = GenesisEnv(
        task=task_name,
        observation_height=observation_height,
        observation_width=observation_width,
        show_viewer=show_viewer,
        use_legacy_sound_config=True,
    )
    print("Policy Input Features:", policy.config.input_features)
    print("Environment Observation Space:", env.observation_space)
    print("Policy Output Features:", policy.config.output_features)
    print("Environment Action Space:", env.action_space)
    success_num = 0
    all_actions = []  # Store actions from all episodes
    combined_video_h = observation_height * 2
    combined_video_w = observation_width * 2
    ep = 0
    while ep < episode_num:
        try:
            print(f"\n=== Episode {ep+1} ===")
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            numpy_observation, _ = env.reset()
            episode_task_description = env.get_task_description()
            print(f"Task description: {episode_task_description}")
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

                # Some evaluation environments intentionally omit sound images
                # (e.g. s0 settings), while the training dataset schema may still
                # expect them. Match training-time behavior by padding zeros.
                fill_missing_image_observations(converted_obs, dataset.features)
                
                # Build observation frame in dataset format
                observation_frame = build_dataset_frame(dataset.features, converted_obs, prefix=OBS_STR)
                task_description = env.get_task_description()
                
                # Use predict_action to apply preprocessor, policy, and postprocessor
                action_dict = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=get_safe_torch_device(policy.config.device),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=task_description,
                    robot_type=None,
                )
                
                # Extract action tensor from the returned dictionary
                if isinstance(action_dict, dict) and 'action' in action_dict:
                    action_tensor = action_dict['action']
                else:
                    action_tensor = action_dict
                
                # Convert to numpy for environment
                numpy_action = action_tensor.squeeze(0).cpu().numpy()
                all_actions.append(numpy_action)  # Record the action
                numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
                # print(f"Step: {step}, Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")
                rewards.append(reward)
                current_combined_frame = combine_frames(numpy_observation, observation_height, observation_width)
                frames.append(current_combined_frame)
                done = terminated or truncated
                step += 1
                if reward > 0: # Early termination on success
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
            ep += 1
        except Exception as e:
            print(f"⚠️ Error occurred during episode {ep+1}: {e}")
            print("🔄 Retrying episode...")
            env.close()
            env = GenesisEnv(
                task=task_name,
                observation_height=observation_height,
                observation_width=observation_width,
                show_viewer=show_viewer,
                use_legacy_sound_config=True,
            )
            continue
    env.close()
    success_rate = (success_num / episode_num) * 100
    print(f"Success rate: {success_num}/{episode_num} ({success_rate:.2f}%)")
    
    # Compute action statistics
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
    
    # Write to success_rate.txt
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
    parser = argparse.ArgumentParser(description="Evaluate a trained policy checkpoint.")
    parser.add_argument(
        "--training-name",
        default="act_soundDiff-m4-f10-s2-p0_0",
        help="Model directory name under outputs/train/.",
    )
    parser.add_argument(
        "--checkpoint-step",
        default="100000",
        help='Checkpoint step under outputs/train/<training_name>/checkpoints/. Use "last" if needed.',
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset directory name under datasets/. If omitted, it is inferred from training-name.",
    )
    parser.add_argument("--observation-height", type=int, default=224)
    parser.add_argument("--observation-width", type=int, default=224)
    parser.add_argument("--episode-num", type=int, default=100)
    parser.add_argument("--show-viewer", action="store_true")
    args = parser.parse_args()

    main(
        training_name=args.training_name,
        observation_height=args.observation_height,
        observation_width=args.observation_width,
        episode_num=args.episode_num,
        show_viewer=args.show_viewer,
        checkpoint_step=str(args.checkpoint_step),
        dataset_name=args.dataset_name,
    )
