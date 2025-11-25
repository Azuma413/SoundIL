"""
データセットの実際の保存内容を詳細に確認
"""
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    dataset_path = Path("datasets/sound-m3-fx-sx_0")
    
    print("=" * 80)
    print("Inspecting Raw Dataset Content")
    print("=" * 80)
    
    # 最初のparquetファイルを読み込む
    first_parquet = dataset_path / "data" / "chunk-000" / "file-000.parquet"
    
    if not first_parquet.exists():
        print(f"Error: File not found: {first_parquet}")
        return
    
    print(f"\nReading: {first_parquet}")
    df = pd.read_parquet(first_parquet)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 最初の10フレームのactionを表示
    print("\n" + "=" * 80)
    print("First 10 frames - Action values:")
    print("=" * 80)
    
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper"
    ]
    
    print(f"\n{'Frame':<8} {' '.join([f'{name:<12}' for name in joint_names])}")
    print("-" * 90)
    
    for idx in range(min(10, len(df))):
        action = np.array(df.loc[idx, 'action'])
        action_str = ' '.join([f'{val:<12.6f}' for val in action])
        print(f"{idx:<8} {action_str}")
    
    # observation.stateも確認
    print("\n" + "=" * 80)
    print("First 5 frames - observation.state values:")
    print("=" * 80)
    
    state_names = [
        "eef_pos_x", "eef_pos_y", "eef_pos_z",
        "eef_quat_w", "eef_quat_x", "eef_quat_y", "eef_quat_z",
        "grip_angle"
    ]
    
    for idx in range(min(5, len(df))):
        state = np.array(df.loc[idx, 'observation.state'])
        print(f"\nFrame {idx}:")
        for i, name in enumerate(state_names):
            print(f"  {name:<15}: {state[i]:.6f}")
    
    # エピソード0の全フレームの統計
    print("\n" + "=" * 80)
    print("Episode 0 - Action statistics:")
    print("=" * 80)
    
    episode_0_mask = df['episode_index'] == 0
    episode_0_actions = np.array([np.array(a) for a in df.loc[episode_0_mask, 'action'].values])
    
    print(f"\nTotal frames in episode 0: {len(episode_0_actions)}")
    print(f"\n{'Joint':<20} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 80)
    
    for i, name in enumerate(joint_names):
        joint_values = episode_0_actions[:, i]
        print(f"{name:<20} {joint_values.min():>12.6f} {joint_values.max():>12.6f} "
              f"{joint_values.mean():>12.6f} {joint_values.std():>12.6f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
