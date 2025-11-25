"""
データセット作成時とPolicy評価時の関節順序を確認
"""
import numpy as np

print("=" * 80)
print("Joint Order Check")
print("=" * 80)

# make_sim_dataset.pyで定義されている関節名
dataset_joints_name = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

print("\n1. Dataset Joint Names (from env.tasks.normal):")
print(f"   {dataset_joints_name}")

print("\n2. Dataset Feature Names (from meta/info.json):")
feature_names = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]
print(f"   {tuple(feature_names)}")

# データセット統計
print("\n" + "=" * 80)
print("Dataset Statistics (from analyze_dataset_actions.py):")
print("=" * 80)
dataset_stats = {
    "shoulder_pan": (-1.032878, 1.049928, -0.037778, 0.467255),
    "shoulder_lift": (-0.990089, 0.985590, 0.239595, 0.389767),
    "elbow_flex": (-1.580934, 1.346996, -0.207260, 0.609407),
    "wrist_flex": (0.772381, 1.658063, 1.468754, 0.276596),
    "wrist_roll": (-0.984198, 1.098607, 0.010901, 0.467255),
    "gripper": (0.000000, 1.047198, 0.594355, 0.518796)
}

print(f"{'Joint':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
print("-" * 80)
for name in feature_names:
    min_val, max_val, mean_val, std_val = dataset_stats[name]
    print(f"{name:<20} {min_val:>10.4f} {max_val:>10.4f} {mean_val:>10.4f} {std_val:>10.4f}")

# Policy出力統計
print("\n" + "=" * 80)
print("Policy Output Statistics (from analyze_policy_actions.py):")
print("=" * 80)
policy_stats = {
    "shoulder_pan": (0.580817, 1.616554, 1.112774, 0.323771),
    "shoulder_lift": (-0.393708, 0.672732, 0.187500, 0.307562),
    "elbow_flex": (-0.340063, 0.807519, 0.341764, 0.321769),
    "wrist_flex": (-1.717979, 0.555558, -0.778879, 0.789430),
    "wrist_roll": (0.585462, 1.621280, 1.113093, 0.323289),
    "gripper": (-1.162611, 0.890850, 0.040610, 0.975558)
}

print(f"{'Joint':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
print("-" * 80)
for name in feature_names:
    min_val, max_val, mean_val, std_val = policy_stats[name]
    print(f"{name:<20} {min_val:>10.4f} {max_val:>10.4f} {mean_val:>10.4f} {std_val:>10.4f}")

# 相関を確認するために、統計の類似度をチェック
print("\n" + "=" * 80)
print("Checking for potential joint order mismatch:")
print("=" * 80)
print("Looking for similar mean/std patterns between dataset and policy output...")
print()

# データセットの各関節について、Policyのどの関節と統計が似ているか確認
for ds_name in feature_names:
    ds_min, ds_max, ds_mean, ds_std = dataset_stats[ds_name]
    print(f"\nDataset {ds_name} (mean={ds_mean:.4f}, std={ds_std:.4f}):")
    
    # Policy出力の各関節との差を計算
    similarities = []
    for policy_name in feature_names:
        p_min, p_max, p_mean, p_std = policy_stats[policy_name]
        
        # mean と std の差の絶対値を計算
        mean_diff = abs(ds_mean - p_mean)
        std_diff = abs(ds_std - p_std)
        total_diff = mean_diff + std_diff
        
        similarities.append((policy_name, total_diff, mean_diff, std_diff))
    
    # 類似度でソート
    similarities.sort(key=lambda x: x[1])
    
    # 最も似ている3つを表示
    for i, (policy_name, total_diff, mean_diff, std_diff) in enumerate(similarities[:3]):
        marker = "<<< MATCH!" if i == 0 and total_diff < 0.3 else ""
        print(f"  {i+1}. Policy {policy_name:<15} (mean_diff={mean_diff:.4f}, std_diff={std_diff:.4f}, total={total_diff:.4f}) {marker}")

print("\n" + "=" * 80)
