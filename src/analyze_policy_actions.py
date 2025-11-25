from pathlib import Path
import numpy as np
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.genesis_env import GenesisEnv

def analyze_policy_actions(training_name, observation_height, observation_width, episode_num, checkpoint_step="last"):
    """
    Policyが出力するaction値の統計情報を収集
    """
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
    print(f"Detected task name: {task_name}")
    
    env = GenesisEnv(task=task_name, observation_height=observation_height, observation_width=observation_width, show_viewer=False)
    
    print("\n" + "=" * 80)
    print(f"Collecting actions from {episode_num} episodes...")
    print("=" * 80)
    
    all_actions = []
    
    for ep in range(episode_num):
        print(f"\nEpisode {ep+1}/{episode_num}")
        policy.reset()
        numpy_observation, _ = env.reset()
        
        step = 0
        done = False
        
        while not done:
            observation = {}
            for key in policy.config.input_features:
                if key == "observation.state":
                    data = numpy_observation[key]
                    tensor_data = torch.from_numpy(data).to(torch.float32)
                    observation[key] = tensor_data.to(device).unsqueeze(0)
                else:
                    img = numpy_observation[key]
                    img = img.copy()
                    tensor_img = torch.from_numpy(img).to(torch.float32) / 255.0
                    if tensor_img.ndim == 3 and tensor_img.shape[2] in [1, 3, 4]:
                        tensor_img = tensor_img.permute(2, 0, 1)
                    elif tensor_img.ndim == 2:
                        tensor_img = tensor_img.unsqueeze(0)
                    observation[key] = tensor_img.to(device).unsqueeze(0)
            
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
            all_actions.append(numpy_action)
            
            numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
            done = terminated or truncated or reward > 0
            step += 1
            
            if step >= 100:  # 最大100ステップに制限
                break
    
    env.close()
    
    # 統計情報を計算
    all_actions = np.array(all_actions)
    
    print("\n" + "=" * 80)
    print(f"Analysis Results: {len(all_actions)} actions collected")
    print("=" * 80)
    
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper"
    ]
    
    print("\nPolicy Action Statistics per Joint:")
    print("-" * 80)
    print(f"{'Joint':<20} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 80)
    
    for i, joint_name in enumerate(joint_names):
        joint_actions = all_actions[:, i]
        print(f"{joint_name:<20} {joint_actions.min():>12.6f} {joint_actions.max():>12.6f} "
              f"{joint_actions.mean():>12.6f} {joint_actions.std():>12.6f}")
    
    print("=" * 80)
    
    # 全体の統計
    print(f"\nOverall Statistics:")
    print(f"  Min: {all_actions.min():.6f}")
    print(f"  Max: {all_actions.max():.6f}")
    print(f"  Mean: {all_actions.mean():.6f}")
    print(f"  Std: {all_actions.std():.6f}")
    
    # URDF定義との比較
    print("\n" + "=" * 80)
    print("Comparison with URDF Joint Limits:")
    print("-" * 80)
    
    urdf_limits = {
        "shoulder_pan": (-1.91986, 1.91986),
        "shoulder_lift": (-1.74533, 1.74533),
        "elbow_flex": (-1.69, 1.69),
        "wrist_flex": (-1.65806, 1.65806),
        "wrist_roll": (-2.74385, 2.84121),
        "gripper": (-0.17453, 1.74533)
    }
    
    print(f"{'Joint':<20} {'Policy Min':>12} {'URDF Min':>12} {'Policy Max':>12} {'URDF Max':>12} {'Within Limits':>15}")
    print("-" * 80)
    
    for i, joint_name in enumerate(joint_names):
        joint_actions = all_actions[:, i]
        urdf_min, urdf_max = urdf_limits[joint_name]
        within_limits = (joint_actions.min() >= urdf_min) and (joint_actions.max() <= urdf_max)
        
        print(f"{joint_name:<20} {joint_actions.min():>12.6f} {urdf_min:>12.6f} "
              f"{joint_actions.max():>12.6f} {urdf_max:>12.6f} {'✓' if within_limits else '✗':>15}")
    
    print("=" * 80)
    
    # データセットとの比較用の統計を表示
    print("\n" + "=" * 80)
    print("Expected Dataset Statistics (from previous analysis):")
    print("-" * 80)
    dataset_stats = {
        "shoulder_pan": (-1.032878, 1.049928, -0.037778, 0.467255),
        "shoulder_lift": (-0.990089, 0.985590, 0.239595, 0.389767),
        "elbow_flex": (-1.580934, 1.346996, -0.207260, 0.609407),
        "wrist_flex": (0.772381, 1.658063, 1.468754, 0.276596),
        "wrist_roll": (-0.984198, 1.098607, 0.010901, 0.467255),
        "gripper": (0.000000, 1.047198, 0.594355, 0.518796)
    }
    
    print(f"{'Joint':<20} {'DS Min':>10} {'P Min':>10} {'DS Max':>10} {'P Max':>10} {'DS Mean':>10} {'P Mean':>10}")
    print("-" * 80)
    
    for i, joint_name in enumerate(joint_names):
        joint_actions = all_actions[:, i]
        ds_min, ds_max, ds_mean, _ = dataset_stats[joint_name]
        
        print(f"{joint_name:<20} {ds_min:>10.4f} {joint_actions.min():>10.4f} "
              f"{ds_max:>10.4f} {joint_actions.max():>10.4f} "
              f"{ds_mean:>10.4f} {joint_actions.mean():>10.4f}")
    
    print("=" * 80)
    
    return all_actions

if __name__ == "__main__":
    training_name = "act_sound-m3-fx-sx_0"
    checkpoint_step = "100000"
    
    analyze_policy_actions(
        training_name=training_name,
        observation_height=224,
        observation_width=224,
        episode_num=5,  # 5エピソード分のデータを収集
        checkpoint_step=checkpoint_step,
    )
