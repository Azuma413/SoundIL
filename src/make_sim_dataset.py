import numpy as np
from PIL import Image
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.genesis_env import GenesisEnv
from env.tasks.normal import joints_name, AGENT_DIM
from lerobot.datasets.lerobot_dataset import LeRobotDataset

saved_cube_pos = None
is_first_call = True

def expert_policy(env, stage):
    global saved_cube_pos, is_first_call
    task = env._env
    if task.color == "red":
        cube_pos = task.cubeR.get_pos().cpu().numpy()
    elif task.color == "blue":
        cube_pos = task.cubeB.get_pos().cpu().numpy()
    elif task.color == "green":
        cube_pos = task.cubeG.get_pos().cpu().numpy()
    box_pos = task.box.get_pos().cpu().numpy()
    grip_close = np.array([0.0])
    grip_open = np.array([np.pi/3])
    quat = np.array([1, 0, 0, 0], dtype=np.float32)
    quat /= np.linalg.norm(quat)
    eef = task.eef
    offset = np.array([-0.02, 0.0, 0.0])
    # === Stage definitions ===
    if stage == "hover":
        is_first_call = True
        target_pos = cube_pos + np.array([0.0, 0.0, 0.15]) + offset # hover safely
        grip = grip_open
    elif stage == "stabilize":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.10]) + offset
        grip = grip_open  # still open
    elif stage == "grasp":
        target_pos = cube_pos + np.array([0.0, 0.0, 0.07]) + offset  # lower slightly
        grip = grip_close  # close grip
    elif stage == "lift":
        if is_first_call:
            saved_cube_pos = cube_pos
            is_first_call = False
        target_pos = np.array([saved_cube_pos[0], saved_cube_pos[1], 0.15]) + offset
        grip = grip_close  # keep closed
    elif stage == "to_box":
        target_pos = box_pos + np.array([0.0, 0.0, 0.16]) + offset
        grip = grip_close
    elif stage == "stabilize_box":
        target_pos = box_pos + np.array([0.0, 0.0, 0.16]) + offset
        grip = grip_close
    elif stage == "release":
        target_pos = box_pos + np.array([0.0, 0.0, 0.16]) + offset
        grip = grip_open
    else:
        raise ValueError(f"Unknown stage: {stage}")
    qpos = task.so_arm.inverse_kinematics(
        link=eef,
        pos=target_pos,
        quat=quat,
    ).cpu().numpy()
    qpos_arm = qpos[:-1]
    action = np.concatenate([qpos_arm, grip])
    return action.astype(np.float32)

def initialize_dataset(env: GenesisEnv) -> LeRobotDataset:
    task = env.task
    height = env.observation_height
    width = env.observation_width
    dict_idx = 0
    dataset_path = f"datasets/{task}_{dict_idx}"
    while os.path.exists(f"datasets/{task}_{dict_idx}"):
        dict_idx += 1
        dataset_path = f"datasets/{task}_{dict_idx}"
    # env.observation_spaceの内容に基づいてfeaturesを定義
    features = {"action": {"dtype": "float32", "shape": (AGENT_DIM,), "names": joints_name}}
    for key, space in env.observation_space.spaces.items():
        if key == "observation.state":
            features[key] = {"dtype": "float32", "shape": (8,), "names": joints_name}
        elif key.startswith("observation.images"):
            # すべての画像は3チャンネル（sound0, sound1, specも含む）
            features[key] = {"dtype": "video", "shape": (height, width, 3), "names": ("height", "width", "channels")}
    lerobot_dataset = LeRobotDataset.create(
        repo_id=None,
        fps=30,
        root=dataset_path,
        robot_type="so-101",
        use_videos=True,
        features=features,
        # batch_encoding_size=10,
        batch_encoding_size=1,
    )
    return lerobot_dataset

def main(task, stage_dict, observation_height=480, observation_width=640, episode_num=1, show_viewer=False, sound_config=None):
    env = GenesisEnv(task=task, observation_height=observation_height, observation_width=observation_width, show_viewer=show_viewer, sound_config=sound_config)
    dataset = initialize_dataset(env)
    ep = 0
    while ep < episode_num:
        print(f"\n🎬 Starting episode {ep+1}")
        env.reset()
        obs_dict = {"action": []}
        for key in env.observation_space.spaces.keys():
            obs_dict[key] = []
        save_flag = False
        for stage in stage_dict.keys():
            print(f"  Stage: {stage}")
            for t in range(stage_dict[stage]):
                action = expert_policy(env, stage)
                obs, reward, _, _, _ = env.step(action)
                obs_dict["action"].append(action)
                for key in obs.keys():
                    if key in obs_dict.keys():
                        obs_dict[key].append(obs[key])
                if reward > 0:
                    save_flag = True
        # if not save_flag:
        #     print(f"🚫 Skipping episode {ep+1}")
        #     continue
        print(f"✅ Saving episode {ep+1}")
        ep += 1
        for i in range(len(obs_dict["action"])):
            obs = {"task": env.get_task_description()}
            for key in obs_dict.keys():
                if key.startswith("observation.images") and isinstance(obs_dict[key][i], Image.Image):
                    obs_dict[key][i] = np.array(obs_dict[key][i])
                obs[key] = obs_dict[key][i]
            dataset.add_frame(obs)
        dataset.save_episode()
    env.close()

if __name__ == "__main__":
    # datasetを作成したいタスクを指定
    task = "sound-m6-fx-so" # "normal"
    stage_dict = {
        "hover": 10, # cubeの上に手を持っていく
        # "hover": 100, # cubeの上に手を持っていく
        # "stabilize": 50, # cubeの上で手を安定させる
        # "grasp": 100, # cubeを掴む
        # "lift": 100, # cubeを持ち上げる
        # "to_box": 100, # cubeを箱の上に持っていく
        # "stabilize_box": 50, # cubeを箱の上で安定させる
        # "release": 100, # cubeを離す
    }
    # sound_config = SoundConfig()
    sound_config = None # Noneならタスクごとのデフォルト値が使われる．
    main(episode_num=1, task=task, stage_dict=stage_dict, observation_height=224, observation_width=224, show_viewer=False, sound_config=sound_config)

# normal: 音は関係なく，赤，青，緑のCubeから指定された色のCubeを箱に入れるタスク
# sound-m3-fo-sx: mはマイクロフォンアレイの数, fは特徴量マップを使うかどうか，sはスペクトログラムを使うかどうか
