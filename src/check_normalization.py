from safetensors import safe_open
import torch

# 逆正規化パラメータを読み込む
unnorm_path = "outputs/train/act_sound-m3-fx-sx_0/checkpoints/100000/pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors"

print("=" * 80)
print("Unnormalizer (Postprocessor) Parameters:")
print("=" * 80)

with safe_open(unnorm_path, framework='pt', device='cpu') as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"\n{key}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Values: {tensor}")

# 正規化パラメータも読み込む
norm_path = "outputs/train/act_sound-m3-fx-sx_0/checkpoints/100000/pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors"

print("\n" + "=" * 80)
print("Normalizer (Preprocessor) Parameters:")
print("=" * 80)

with safe_open(norm_path, framework='pt', device='cpu') as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"\n{key}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Values: {tensor}")

# データセットの統計と比較
print("\n" + "=" * 80)
print("Expected Dataset Statistics vs Normalization Parameters:")
print("=" * 80)

dataset_stats = {
    "shoulder_pan": (-0.037778, 0.467255),
    "shoulder_lift": (0.239595, 0.389767),
    "elbow_flex": (-0.207260, 0.609407),
    "wrist_flex": (1.468754, 0.276596),
    "wrist_roll": (0.010901, 0.467255),
    "gripper": (0.594355, 0.518796)
}

joint_names = [
    "shoulder_pan",
    "shoulder_lift", 
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]

print("\nDataset Mean/Std:")
for i, name in enumerate(joint_names):
    mean, std = dataset_stats[name]
    print(f"  {name:<20}: mean={mean:>8.4f}, std={std:>8.4f}")

# 逆正規化パラメータから mean と std を取得
with safe_open(unnorm_path, framework='pt', device='cpu') as f:
    action_mean = f.get_tensor('action.mean')
    action_std = f.get_tensor('action.std')

print("\nUnnormalizer Mean/Std (should match dataset):")
for i, name in enumerate(joint_names):
    print(f"  {name:<20}: mean={action_mean[i].item():>8.4f}, std={action_std[i].item():>8.4f}")

print("\n" + "=" * 80)
