"""
inverse_kinematicsが返す関節の順序を確認するスクリプト
"""
import numpy as np
import torch
import genesis as gs
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.genesis_env import GenesisEnv

def main():
    # 環境を作成
    env = GenesisEnv(task="sound-m3-fx-sx", observation_height=224, observation_width=224, show_viewer=False)
    task = env._env
    
    print("=" * 80)
    print("Checking inverse_kinematics output order")
    print("=" * 80)
    
    # 初期位置を取得
    print("\n1. Initial robot state:")
    initial_qpos = task.so_arm.get_dofs_position().cpu().numpy()
    print(f"   Total DOFs: {len(initial_qpos)}")
    print(f"   Initial qpos: {initial_qpos}")
    
    # inverse_kinematicsを実行
    eef = task.eef
    target_pos = np.array([0.3, 0.0, 0.15])
    quat = np.array([1, 0, 0, 0], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    
    print(f"\n2. Running inverse_kinematics:")
    print(f"   Target position: {target_pos}")
    print(f"   Target orientation: {quat}")
    
    ik_result = task.so_arm.inverse_kinematics(
        link=eef,
        pos=target_pos,
        quat=quat,
    ).cpu().numpy()
    
    print(f"\n3. IK result:")
    print(f"   Length: {len(ik_result)}")
    print(f"   Values: {ik_result}")
    
    # データセットのfeature定義
    joints_name_from_dataset = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    
    print(f"\n4. Expected feature names (from dataset):")
    for i, name in enumerate(joints_name_from_dataset):
        print(f"   [{i}] {name}")
    
    # URDFから関節名を取得
    print(f"\n5. Checking URDF joint order:")
    print("   From so101_new_calib.xml actuator section:")
    urdf_joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper"
    ]
    for i, name in enumerate(urdf_joint_names):
        print(f"   [{i}] {name}")
    
    # 各関節を個別に動かして確認
    print(f"\n6. Testing each joint individually:")
    
    # 初期姿勢に戻す
    task.so_arm.set_qpos(torch.tensor(initial_qpos, dtype=torch.float32, device=gs.device), zero_velocity=True)
    task.scene.step()
    
    for joint_idx in range(6):
        # 初期姿勢
        test_qpos = initial_qpos.copy()
        # 1つの関節だけ変更
        test_qpos[joint_idx] += 0.3  # 0.3ラジアン動かす
        
        # 設定
        task.so_arm.set_qpos(torch.tensor(test_qpos, dtype=torch.float32, device=gs.device), zero_velocity=True)
        task.scene.step()
        
        # 現在のEEF位置を取得
        eef_pos = task.eef.get_pos().cpu().numpy()
        
        print(f"\n   Joint {joint_idx} (expecting {urdf_joint_names[joint_idx]}):")
        print(f"      qpos change: [{joint_idx}] += 0.3")
        print(f"      EEF position: {eef_pos}")
        
        # 初期姿勢に戻す
        task.so_arm.set_qpos(torch.tensor(initial_qpos, dtype=torch.float32, device=gs.device), zero_velocity=True)
        task.scene.step()
    
    env.close()
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
