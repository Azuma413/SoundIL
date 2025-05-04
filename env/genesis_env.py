import gymnasium as gym
import genesis as gs
import numpy as np
from gymnasium import spaces
import warnings
from env.tasks.sound import SoundTask

class GenesisEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
            self,
            task,
            observation_height = 480,
            observation_width = 640,
            render_mode=None,
    ):
        super().__init__()
        self.task = task
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.render_mode = render_mode
        self._env = self._make_env_task(self.task)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.seed(seed)
        # resetは obs, info を返す
        observation, info = self._env.reset()
        # infoに is_success を追加 (初期値はFalse)
        info["is_success"] = False
        return observation, info

    def step(self, action):
        # stepは obs, reward, terminated, truncated, info を返す
        observation, reward, terminated, truncated, info = self._env.step(action)
        is_success = (reward == 1.0) # 報酬が1.0なら成功
        info["is_success"] = is_success
        return observation, reward, terminated, truncated, info

    def save_video(self, file_name: str = "save", fps=30):
        self._env.save_videos(file_name=file_name, fps=fps)

    def close(self):
        # 必要であればシーンのクリーンアップ処理を追加
        pass

    def get_obs(self):
        return self._env.get_obs()

    def get_robot(self):
        #TODO: (jadechovhari) add assertion that a robot exist
        return self._env.franka

    def render(self):
        if "front" in self.observation_space.spaces:
            obs = self.get_obs()
            return obs["front"]
        else:
            warnings.warn("front observation is not enabled, cannot render.")
            return None

    def _make_env_task(self, task_name):
        if task_name == "sound":
            # SoundTaskの初期化から不要な引数を削除
            task = SoundTask(observation_height=self.observation_height,
                             observation_width=self.observation_width,
                             )
        else:
            raise NotImplementedError(task_name)
        return task