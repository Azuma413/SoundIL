import gymnasium as gym
import warnings
from env.tasks.normal import NormalTask
from env.tasks.sound import SoundTask
from env.tasks.sound_camera import SoundConfig

class GenesisEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    def __init__(
            self,
            task,
            observation_height = 480,
            observation_width = 640,
            show_viewer=False,
            render_mode=None,
            reset_freq=10,
            sound_config=None,
    ):
        super().__init__()
        self.task = task
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.show_viewer = show_viewer
        self.render_mode = render_mode
        self.sound_config = sound_config  # sound_configを保存
        self._env = self._make_env_task(sound_config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self._max_episode_steps = 700
        self.step_count = 0
        self.reset_freq = reset_freq
        self.episode_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # エピソード回数をインクリメント
        self.episode_count += 1
        # reset_freqの倍数回に達したらメモリ開放とリセット
        if self.episode_count % self.reset_freq == 0:
            # 現在の環境をクローズ
            self.close()
            # 新しい環境を作成（sound_configを渡す）
            self._env = self._make_env_task(self.sound_config)
            self.observation_space = self._env.observation_space
            self.action_space = self._env.action_space
        if seed is not None:
            self._env.seed(seed)
        # resetは obs, info を返す
        self.step_count = 0
        observation, info = self._env.reset()
        # infoに is_success を追加 (初期値はFalse)
        info["is_success"] = False
        return observation, info

    def step(self, action):
        # stepは obs, reward, terminated, truncated, info を返す
        observation, reward, terminated, truncated, info = self._env.step(action)
        is_success = (reward == 1.0)
        info["is_success"] = is_success
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            terminated = True
            truncated = True
        return observation, reward, terminated, truncated, info

    def save_video(self, file_name: str = "save", fps=30):
        self._env.save_videos(file_name=file_name, fps=fps)

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def get_obs(self):
        return self._env.get_obs()

    def render(self):
        if "observation.images.front" in self.observation_space.spaces:
            obs = self.get_obs()
            return obs["observation.images.front"]
        else:
            warnings.warn("front observation is not enabled, cannot render.")
            return None

    def get_task_description(self):
        return self._env.get_task_description()

    def _make_env_task(self, sound_config=None):
        if self.task == "normal":
            env = NormalTask(
                observation_height=self.observation_height,
                observation_width=self.observation_width,
                show_viewer=self.show_viewer,
            )
        elif "sound" in self.task:
            use_feat = self.task.split("-")[2] == "fo"
            use_spec = self.task.split("-")[3] == "so"
            mic_num = 3 if use_feat else int(self.task.split("-")[1][1])
            if sound_config is None:
                sound_config = SoundConfig(
                    mic_array_num=mic_num,
                    use_spectrogram=use_spec,
                    use_feature=use_feat,
                    audio_file_path="sounds/1.wav", # None
                )
            env = SoundTask(
                observation_height=self.observation_height,
                observation_width=self.observation_width,
                show_viewer=self.show_viewer,
                sound_config=sound_config,
            )
        else:
            raise NotImplementedError(f"Task {self.task} is not implemented.")
        return env
