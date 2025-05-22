from lerobot.common.envs import EnvConfig
from lerobot.common.constants import ACTION, OBS_ENV, OBS_IMAGE, OBS_IMAGES, OBS_ROBOT
from lerobot.configs.types import FeatureType, PolicyFeature
from dataclasses import dataclass, field

# 以下のファイルを参照
# lerobot/lerobot/common/envs/configs.py

@EnvConfig.register_subclass("sound")
@dataclass
class SoundEnv(EnvConfig):
    task: str = "sound"
    fps: int = 30
    episode_length: int = 500
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }

@EnvConfig.register_subclass("test")
@dataclass
class TestEnv(EnvConfig):
    task: str = "test"
    fps: int = 30
    episode_length: int = 500
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    
    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }

@EnvConfig.register_subclass("marker_sound")
@dataclass
class MarkerSoundEnv(EnvConfig):
    task: str = "marker_sound"
    fps: int = 30
    episode_length: int = 500
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }
@EnvConfig.register_subclass("weighted_sound")
@dataclass
class WeightedSoundEnv(EnvConfig):
    task: str = "weighted_sound"
    fps: int = 30
    episode_length: int = 500
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }
@EnvConfig.register_subclass("2sound")
@dataclass
class TwoSoundEnv(EnvConfig):
    task: str = "2sound"
    fps: int = 30
    episode_length: int = 1000
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }
@EnvConfig.register_subclass("marker_2sound")
@dataclass
class MarkerTwoSoundEnv(EnvConfig):
    task: str = "marker_2sound"
    fps: int = 30
    episode_length: int = 1000
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }
@EnvConfig.register_subclass("weighted_2sound")
@dataclass
class WeightedTwoSoundEnv(EnvConfig):
    task: str = "weighted_2sound"
    fps: int = 30
    episode_length: int = 1000
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }