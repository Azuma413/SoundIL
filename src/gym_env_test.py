import genesis as gs
import numpy as np
from tqdm import trange
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.genesis_env import GenesisEnv
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import os
import torch
import numpy as np
from tqdm import trange
from PIL import Image

def expert_policy(env, stage, step_i):
    """
    A low-level, smooth grasping policy inspired by direct control:
    1. Move to pre-grasp hover
    2. Hold & stabilize
    3. Grasp
    4. Lift
    """
    task = env._env
    cube_pos = task.cubeA.get_pos().cpu().numpy()
    motors_dof = task.motors_dof
    fingers_dof = task.fingers_dof
    finder_pos = -0.02  # tighter grip
    quat = np.array([[0, 1, 0, 0]])
    eef = task.eef

    # === Stage definitions ===
    if stage == "hover":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.115])  # hover safely
        grip = np.array([[0.04, 0.04]])  # open

    elif stage == "stabilize":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.115])
        grip = np.array([[0.04, 0.04]])  # still open

    elif stage == "grasp":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.03])  # lower slightly
        grip = np.array([[finder_pos, finder_pos]])  # close grip

    elif stage == "lift":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.25])
        grip = np.array([[finder_pos, finder_pos]])  # keep closed

    else:
        raise ValueError(f"Unknown stage: {stage}")
    # Use IK to compute joint positions for the arm
    qpos = task.franka.inverse_kinematics(
        link=eef,
        pos=target_pos,
        quat=quat,
    ).cpu().numpy()
    # Combine arm + gripper, handle potential shape variations of qpos
    if qpos.ndim == 1:
        # qpos is (9,), reshape arm part to (1, 7)
        qpos_arm = qpos[:-2].reshape(1, -1)
    elif qpos.ndim == 2 and qpos.shape[0] == 1:
        # qpos is (1, 9), slice arm part to get (1, 7)
        qpos_arm = qpos[:, :-2]
    else:
        raise ValueError(f"Unexpected shape for qpos: {qpos.shape}")
    # grip is already (1, 2)
    action = np.concatenate([qpos_arm, grip], axis=1) # Shape (1, 9)
    # Ensure dtype is float32
    return action.astype(np.float32)

def initialize_dataset(task, height, width):
    # Initialize dataset
    dict_idx = 0
    dataset_path = f"data/{task}_{dict_idx}"
    while os.path.exists(f"data/{task}_{dict_idx}"):
        dict_idx += 1
        dataset_path = f"data/{task}_{dict_idx}"
    lerobot_dataset = LeRobotDataset.create(
        repo_id=None,
        fps=30,
        root=dataset_path,
        robot_type="franka",
        use_videos=True,
        features={
            "observation.state": {"dtype": "float32", "shape": (9,)},
            "action": {"dtype": "float32", "shape": (9,)},
            "observation.images.front": {"dtype": "video", "shape": (height, width, 3)},
            "observation.images.side": {"dtype": "video", "shape": (height, width, 3)},
            "observation.images.sound": {"dtype": "video", "shape": (height, width, 3)},
        },
    )
    return lerobot_dataset

def main(task, observation_height=480, observation_width=640, episode_num=1):
    gs.init(backend=gs.gpu, precision="32")
    env = GenesisEnv(task=task, observation_height=observation_height, observation_width=observation_width)
    dataset = initialize_dataset(task, observation_height, observation_width)
    for ep in range(episode_num):
        print(f"\n🎬 Starting episode {ep+1}")
        env.reset()
        states, images_front, images_side, images_sound, actions = [], [], [], [], []
        reward_greater_than_zero = False
        for stage in ["hover", "stabilize", "grasp", "grasp", "lift"]:
            for t in trange(40, leave=False):
                action = expert_policy(env, stage, t)
                obs, reward, done, _, info = env.step(action)
                states.append(obs["agent_pos"])
                images_front.append(obs["front"])
                images_side.append(obs["side"])
                images_sound.append(obs["sound"])
                actions.append(action)
                if reward > 0:
                    reward_greater_than_zero = True
        env.save_video(file_name=f"video_{ep+1}", fps=30)

        # if not reward_greater_than_zero:
        #     print(f"🚫 Skipping episode {ep+1} — reward was always 0")
        #     continue
        # print(f"✅ Saving episode {ep+1} — reward > 0 observed")

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
                "observation.state": states.astype(np.float32),
                "action": actions.astype(np.float32),
                "observation.images.front": image_front,
                "observation.images.side": image_side,
                "observation.images.sound": image_sound,
                "task": "pick cube",
            })
        dataset.save_episode()
    env.close()

if __name__ == "__main__":
    main(task="sound")