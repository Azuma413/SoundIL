import genesis as gs
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.genesis_env import GenesisEnv
from env.tasks.sound import joints_name
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os
import numpy as np
from PIL import Image

def expert_policy(env, stage):
    task = env._env
    cube_pos = task.cubeA.get_pos().cpu().numpy()
    box_pos = task.box.get_pos().cpu().numpy()
    # motors_dof = task.motors_dof
    # fingers_dof = task.fingers_dof
    finder_pos = -0.02  # tighter grip
    quat = np.array([0, 1, 0, 0]) # Changed from [[0, 1, 0, 0]] to [0, 1, 0, 0]
    eef = task.eef

    # === Stage definitions ===
    if stage == "hover":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.2])  # hover safely
        grip = np.array([0.04, 0.04])  # open
    elif stage == "stabilize":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.1])
        grip = np.array([0.04, 0.04])  # still open
    elif stage == "grasp":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.1])  # lower slightly
        grip = np.array([finder_pos, finder_pos])  # close grip
    elif stage == "lift":
        target_pos = np.array([cube_pos[0], cube_pos[1], 0.25])
        grip = np.array([finder_pos, finder_pos])  # keep closed
    elif stage == "to_box":
        target_pos = box_pos + np.array([0.0, 0.0, 0.25])
        grip = np.array([finder_pos, finder_pos])
    elif stage == "stabilize_box":
        target_pos = box_pos + np.array([0.0, 0.0, 0.25])
        grip = np.array([finder_pos, finder_pos])
    elif stage == "release":
        target_pos = box_pos + np.array([0.0, 0.0, 0.25])
        grip = np.array([0.04, 0.04])
    else:
        raise ValueError(f"Unknown stage: {stage}")
    # Use IK to compute joint positions for the arm
    qpos = task.franka.inverse_kinematics(
        link=eef,
        pos=target_pos,
        quat=quat,
    ).cpu().numpy()
    qpos_arm = qpos[:-2]
    action = np.concatenate([qpos_arm, grip]) # Shape (9)
    return action.astype(np.float32)

def initialize_dataset(task, height, width):
    # Initialize dataset
    dict_idx = 0
    dataset_path = f"datasets/{task}_{dict_idx}"
    while os.path.exists(f"datasets/{task}_{dict_idx}"):
        dict_idx += 1
        dataset_path = f"datasets/{task}_{dict_idx}"
    lerobot_dataset = LeRobotDataset.create(
        repo_id=None,
        fps=30,
        root=dataset_path,
        robot_type="franka",
        use_videos=True,
        features={
            "observation.state": {"dtype": "float32", "shape": (9,), "names": joints_name},
            "action": {"dtype": "float32", "shape": (9,), "names": joints_name},
            "observation.images.front": {"dtype": "video", "shape": (height, width, 3), "names": ("height", "width", "channels")},
            "observation.images.side": {"dtype": "video", "shape": (height, width, 3), "names": ("height", "width", "channels")},
            "observation.images.sound": {"dtype": "video", "shape": (height, width, 3), "names": ("height", "width", "channels")},
        },
    )
    return lerobot_dataset

def main(task, stage_dict, observation_height=480, observation_width=640, episode_num=1, show_viewer=False):
    gs.init(backend=gs.gpu, precision="32") # cpuの方が早い？
    env = None
    dataset = initialize_dataset(task, observation_height, observation_width)
    if task == "sound":
        dummy_dataset = initialize_dataset("dummy", observation_height, observation_width)
    ep = 0
    while ep < episode_num:
        print(f"\n🎬 Starting episode {ep+1}")
        if ep % 10 == 0:
            # メモリリークを避けるために、10エピソードごとに環境をリセット
            if env is not None:
                env.close()
            env = GenesisEnv(task=task, observation_height=observation_height, observation_width=observation_width, show_viewer=show_viewer)
        env.reset()
        states, images_front, images_side, images_sound, actions = [], [], [], [], []
        reward_greater_than_zero = False
        for stage in stage_dict.keys():
            print(f"  Stage: {stage}")
            for t in range(stage_dict[stage]):
                action = expert_policy(env, stage)
                obs, reward, _, _, _ = env.step(action)
                states.append(obs["agent_pos"])
                images_front.append(obs["front"])
                images_side.append(obs["side"])
                images_sound.append(obs["sound"])
                actions.append(action)
                if reward > 0:
                    reward_greater_than_zero = True
        # デバッグ用
        # env.save_video(file_name=f"video", fps=30)

        if not reward_greater_than_zero:
            print(f"🚫 Skipping episode {ep+1} — reward was always 0")
            continue
        print(f"✅ Saving episode {ep+1} — reward > 0 observed")
        ep += 1

        for i in range(len(states)):
            image_front = images_front[i]
            if isinstance(image_front, Image.Image):
                image_front = np.array(image_front)
            image_side = images_side[i]
            if isinstance(image_side, Image.Image):
                image_side = np.array(image_side)
            image_sound = images_sound[i]
            if isinstance(image_sound, Image.Image):
                image_sound = np.array(image_sound)

            dataset.add_frame({
                "observation.state": states[i].astype(np.float32),
                "action": actions[i].astype(np.float32),
                "observation.images.front": image_front,
                "observation.images.side": image_side,
                "observation.images.sound": image_sound,
                "task": "pick cube with sound",
            })
            if task == "sound":
                dummy_dataset.add_frame({
                    "observation.state": states[i].astype(np.float32),
                    "action": actions[i].astype(np.float32),
                    "observation.images.front": image_front,
                    "observation.images.side": image_side,
                    "observation.images.sound": np.zeros_like(image_sound),
                    "task": "pick cube without sound",
                })
        dataset.save_episode()
        if task == "sound":
            dummy_dataset.save_episode()
    env.close()

if __name__ == "__main__":
    task = "test" # "test" or "sound"
    # 20秒くらいのタスクを想定 → 合計600フレーム
    stage_dict = {
        "hover": 100, # cubeの上に手を持っていく
        "stabilize": 60, # cubeの上で手を安定させる
        "grasp": 20, # cubeを掴む
        "lift": 60, # cubeを持ち上げる
        "to_box": 100, # cubeを箱の上に持っていく
        "stabilize_box": 60, # cubeを箱の上で安定させる
        "release": 60, # cubeを離す
    }
    main(task=task, stage_dict=stage_dict, observation_height=480, observation_width=640, episode_num=1, show_viewer=False)