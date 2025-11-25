"""
Policyの出力を詳細にデバッグするスクリプト
"""
from pathlib import Path
import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.genesis_env import GenesisEnv

def main():
    training_name = "act_sound-m3-fx-sx_0"
    checkpoint_step = "100000"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_policy_path = Path(f"outputs/train/{training_name}/checkpoints/{checkpoint_step}/pretrained_model")
    
    print("=" * 80)
    print("Loading Policy and Checking Configuration")
    print("=" * 80)
    
    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    policy.to(device)
    policy.eval()
    
    # Policyの設定を確認
    print("\nPolicy Configuration:")
    print(f"  output_features: {policy.config.output_features}")
    print(f"  normalization_mapping: {policy.config.normalization_mapping}")
    
    # Postprocessorを確認
    if hasattr(policy, 'policy_postprocessor'):
        print(f"\nPolicy Postprocessor: {policy.policy_postprocessor}")
        print(f"  Number of steps: {len(policy.policy_postprocessor.steps)}")
        for i, step in enumerate(policy.policy_postprocessor.steps):
            print(f"  Step {i}: {step}")
    
    # 環境を作成
    task_name = training_name.split("_")[1]
    env = GenesisEnv(task=task_name, observation_height=224, observation_width=224, show_viewer=False)
    
    print("\n" + "=" * 80)
    print("Testing Policy Output")
    print("=" * 80)
    
    # リセット
    policy.reset()
    numpy_observation, _ = env.reset()
    
    # 1ステップだけ実行して詳細にログ
    observation = {}
    for key in policy.config.input_features:
        if key == "observation.state":
            data = numpy_observation[key]
            tensor_data = torch.from_numpy(data).to(torch.float32)
            observation[key] = tensor_data.to(device).unsqueeze(0)
            print(f"\nInput {key}: {tensor_data.numpy()}")
        else:
            img = numpy_observation[key]
            img = img.copy()
            tensor_img = torch.from_numpy(img).to(torch.float32) / 255.0
            if tensor_img.ndim == 3 and tensor_img.shape[2] in [1, 3, 4]:
                tensor_img = tensor_img.permute(2, 0, 1)
            elif tensor_img.ndim == 2:
                tensor_img = tensor_img.unsqueeze(0)
            observation[key] = tensor_img.to(device).unsqueeze(0)
            print(f"\nInput {key}: shape={tensor_img.shape}, min={tensor_img.min():.4f}, max={tensor_img.max():.4f}")
    
    print("\n" + "-" * 80)
    print("Calling policy.select_action()...")
    print("-" * 80)
    
    with torch.inference_mode():
        action_output = policy.select_action(observation)
    
    print(f"\nRaw policy output type: {type(action_output)}")
    
    if isinstance(action_output, dict):
        print("Policy output is a dictionary:")
        for k, v in action_output.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, device={v.device}, dtype={v.dtype}")
                print(f"       values={v.squeeze(0).cpu().numpy()}")
            else:
                print(f"  {k}: {v}")
        
        action_tensor = action_output.get('action', None)
    else:
        action_tensor = action_output
        print(f"Policy output is a tensor: shape={action_tensor.shape}")
    
    if action_tensor is not None:
        numpy_action = action_tensor.squeeze(0).cpu().numpy()
        
        print("\n" + "=" * 80)
        print("Final Action Values:")
        print("=" * 80)
        
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper"
        ]
        
        for i, name in enumerate(joint_names):
            print(f"  {name:<20}: {numpy_action[i]:>10.6f}")
        
        print("\n" + "=" * 80)
        print("Expected Dataset Range (for comparison):")
        print("=" * 80)
        
        dataset_ranges = {
            "shoulder_pan": (-1.032878, 1.049928),
            "shoulder_lift": (-0.990089, 0.985590),
            "elbow_flex": (-1.580934, 1.346996),
            "wrist_flex": (0.772381, 1.658063),
            "wrist_roll": (-0.984198, 1.098607),
            "gripper": (0.000000, 1.047198)
        }
        
        for i, name in enumerate(joint_names):
            min_val, max_val = dataset_ranges[name]
            in_range = min_val <= numpy_action[i] <= max_val
            status = "✓" if in_range else "✗ OUT OF RANGE"
            print(f"  {name:<20}: [{min_val:>8.4f}, {max_val:>8.4f}] - {status}")
    
    env.close()
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
