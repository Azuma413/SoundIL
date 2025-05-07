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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.genesis_env import GenesisEnv

def main(training_name, observation_height, observation_width, episode_num, show_viewer, checkpoint_step="last"):
    policy_list = ["act", "diffusion", "pi0", "tdmpc", "vqbet"]
    task_list = ["test", "sound"]
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
            break
    if task_name is None:
        print(f"Error: Unknown task name in training name '{training_name}'. Expected one of {task_list}.")
        return
    gs.init(backend=gs.cpu, precision="32")
    env = GenesisEnv(task=task_name, observation_height=observation_height, observation_width=observation_width, show_viewer=show_viewer)
    print("Policy Input Features:", policy.config.input_features)
    print("Environment Observation Space:", env.observation_space)
    print("Policy Output Features:", policy.config.output_features)
    print("Environment Action Space:", env.action_space)
    success_num = 0
    for ep in range(episode_num):
        print(f"\n=== Episode {ep+1} ===")
        policy.reset()
        numpy_observation, _ = env.reset()
        rewards = []
        frames = []
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        step = 0
        done = False
        limit = 600
        while not done:
            observation = {}
            for key in policy.config.input_features:
                # make_sim_dataset.pyの観測キーに合わせてマッピング
                if key == "observation.state":
                    data = numpy_observation["agent_pos"]
                    tensor_data = torch.from_numpy(data).to(torch.float32)
                    observation[key] = tensor_data.to(device).unsqueeze(0)
                elif key == "observation.images.front":
                    img = numpy_observation["front"]
                    img = img.copy()  # 負のstride対策
                    tensor_img = torch.from_numpy(img).to(torch.float32) / 255.0
                    if tensor_img.ndim == 3 and tensor_img.shape[2] in [1, 3, 4]:
                        tensor_img = tensor_img.permute(2, 0, 1)
                    elif tensor_img.ndim == 2:
                        tensor_img = tensor_img.unsqueeze(0)
                    observation[key] = tensor_img.to(device).unsqueeze(0)
                elif key == "observation.images.side":
                    img = numpy_observation["side"]
                    img = img.copy()  # 負のstride対策
                    tensor_img = torch.from_numpy(img).to(torch.float32) / 255.0
                    if tensor_img.ndim == 3 and tensor_img.shape[2] in [1, 3, 4]:
                        tensor_img = tensor_img.permute(2, 0, 1)
                    elif tensor_img.ndim == 2:
                        tensor_img = tensor_img.unsqueeze(0)
                    observation[key] = tensor_img.to(device).unsqueeze(0)
                elif key == "observation.images.sound":
                    img = numpy_observation["sound"]
                    img = img.copy()  # 負のstride対策
                    tensor_img = torch.from_numpy(img).to(torch.float32) / 255.0
                    if tensor_img.ndim == 3 and tensor_img.shape[2] in [1, 3, 4]:
                        tensor_img = tensor_img.permute(2, 0, 1)
                    elif tensor_img.ndim == 2:
                        tensor_img = tensor_img.unsqueeze(0)
                    observation[key] = tensor_img.to(device).unsqueeze(0)
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
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            done = terminated or truncated
            step += 1
            if step >= limit:
                print(f"Reached step limit ({limit}).")
                break
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
            if not all(f.dtype == np.uint8 for f in valid_frames):
                valid_frames = [(f * 255).clip(0, 255).astype(np.uint8) if np.issubdtype(f.dtype, np.floating) else f.astype(np.uint8) for f in valid_frames]
            first_shape = valid_frames[0].shape
            if not all(f.shape == first_shape for f in valid_frames):
                print("Warning: Frame shapes are inconsistent. Video may be corrupted.")
            try:
                imageio.mimsave(str(video_path), np.stack(valid_frames), fps=fps, plugin='pyav', output_params=['-pix_fmt', 'yuv420p'])
            except Exception:
                imageio.mimsave(str(video_path), np.stack(valid_frames), fps=fps)
            print(f"Video saved: {video_path}")
        else:
            print("No valid frames recorded, skipping video saving.")
    env.close()        
    print(f"Success rate: {success_num}/{episode_num} ({(success_num / episode_num) * 100:.2f}%)")
    # 成功率をtextファイルに保存
    success_rate_file = output_directory / "success_rate.txt"
    with open(success_rate_file, "w") as f:
        f.write(f"Success rate: {success_num}/{episode_num} ({(success_num / episode_num) * 100:.2f}%)\n")

if __name__ == "__main__":
    training_name = "act-test_0"
    observation_height = 480
    observation_width = 640
    episode_num = 20
    show_viewer = False
    checkpoint_step = "last"
    main(
        training_name=training_name,
        observation_height=observation_height,
        observation_width=observation_width,
        episode_num=episode_num,
        show_viewer=show_viewer,
        checkpoint_step=checkpoint_step,
    )