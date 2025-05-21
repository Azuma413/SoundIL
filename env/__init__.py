from lerobot.common.envs import EnvConfig
from lerobot.common.constants import ACTION, OBS_ENV, OBS_IMAGE, OBS_IMAGES, OBS_ROBOT
from lerobot.configs.types import FeatureType, PolicyFeature
from dataclasses import dataclass, field

# 以下のファイルを参照
# lerobot/lerobot/common/envs/configs.py

@EnvConfig.register_subclass("genesis")
@dataclass
class SoundEnv(EnvConfig):
    task: str = "sound"
    fps: int = 30
    episode_length: int = 200
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    @property
    def gym_kwargs(self) -> dict:
        return {
            "task": self.task,
            "observation_height": 480,
            "observation_width": 640,
            "show_viewer": False,
        }

@EnvConfig.register_subclass("genesis")
@dataclass
class TestEnv(EnvConfig):
    task: str = "test"
    fps: int = 30
    episode_length: int = 200
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    @property
    def gym_kwargs(self) -> dict:
        return {
            "task": self.task,
            "observation_height": 480,
            "observation_width": 640,
            "show_viewer": False,
        }