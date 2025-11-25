import numpy as np
import pandas as pd
from pathlib import Path

def analyze_dataset_actions(dataset_path):
    """
    データセット内のaction値の統計情報を出力
    """
    dataset_dir = Path(dataset_path)
    
    # データファイルを読み込む
    data_dir = dataset_dir / "data"
    all_actions = []
    
    print(f"Analyzing dataset: {dataset_path}")
    print("=" * 80)
    
    # すべてのparquetファイルを読み込む
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            df = pd.read_parquet(parquet_file)
            # actionカラムから値を取得（リスト形式で保存されている）
            actions = np.array([np.array(a) for a in df['action'].values])
            all_actions.append(actions)
    
    # すべてのactionを結合
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"\nTotal frames: {len(all_actions)}")
    print(f"Action shape: {all_actions.shape}")
    print("\n" + "=" * 80)
    
    # 各関節の統計情報
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper"
    ]
    
    print("\nAction Statistics per Joint:")
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
    
    # 各関節の値の範囲をURDF定義と比較
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
    
    print(f"{'Joint':<20} {'Dataset Min':>12} {'URDF Min':>12} {'Dataset Max':>12} {'URDF Max':>12} {'Within Limits':>15}")
    print("-" * 80)
    
    for i, joint_name in enumerate(joint_names):
        joint_actions = all_actions[:, i]
        urdf_min, urdf_max = urdf_limits[joint_name]
        within_limits = (joint_actions.min() >= urdf_min) and (joint_actions.max() <= urdf_max)
        
        print(f"{joint_name:<20} {joint_actions.min():>12.6f} {urdf_min:>12.6f} "
              f"{joint_actions.max():>12.6f} {urdf_max:>12.6f} {'✓' if within_limits else '✗':>15}")
    
    print("=" * 80)
    
    return all_actions

if __name__ == "__main__":
    # データセットパスを指定
    dataset_path = "datasets/sound-m3-fx-sx_0"
    analyze_dataset_actions(dataset_path)
