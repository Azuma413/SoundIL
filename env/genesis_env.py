import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
class GenesisEnv(gym.Env):
    """

    """

    metadata = {"render_modes": "rgb_array", "render_fps": 30}

    def __init__(
        self,
        obs_type="state",
        render_mode="rgb_array",
        block_cog=None,
        damping=None,
        observation_width=96,
        observation_height=96,
        visualization_width=680,
        visualization_height=680,
    ):
        super().__init__()
        # Observations
        self.obs_type = obs_type

        # Rendering
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        # Initialize spaces
        self._initialize_observation_space()
        self.action_space = spaces.Box(low=0, high=512, shape=(2,), dtype=np.float32)

        # Physics
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["render_fps"]
        self.dt = 0.01
        self.block_cog = block_cog
        self.damping = damping

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None

        self.teleop = None
        self._last_action = None

        self.success_threshold = 0.95  # 95% coverage

    def step(self, action):
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        self._last_action = action
        for _ in range(n_steps):
            # Step PD control
            # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
            acceleration = self.k_p * (action - self.agent.position) + self.k_v * (
                Vec2d(0, 0) - self.agent.velocity
            )
            self.agent.velocity += acceleration * self.dt

            # Step physics
            self.space.step(self.dt)

        # Compute reward
        coverage = self._get_coverage()
        reward = np.clip(coverage / self.success_threshold, 0.0, 1.0)
        terminated = is_success = coverage > self.success_threshold

        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = is_success
        info["coverage"] = coverage

        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._setup()

        if options is not None and options.get("reset_to_state") is not None:
            state = np.array(options.get("reset_to_state"))
        else:
            # state = self.np_random.uniform(low=[50, 50, 100, 100, -np.pi], high=[450, 450, 400, 400, np.pi])
            rs = np.random.RandomState(seed=seed)
            state = np.array(
                [
                    rs.randint(50, 450),
                    rs.randint(50, 450),
                    rs.randint(100, 400),
                    rs.randint(100, 400),
                    rs.randn() * 2 * np.pi - np.pi,
                ],
                # dtype=np.float64
            )
        self._set_state(state)

        observation = self.get_obs()
        info = self._get_info()
        info["is_success"] = False

        if self.render_mode == "human":
            self.render()

        return observation, info

    def render(self):
        return self._render(visualize=True)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def teleop_agent(self):
        teleop_agent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return teleop_agent(act)

    def get_obs(self):
        if self.obs_type == "state":
            agent_position = np.array(self.agent.position)
            block_position = np.array(self.block.position)
            block_angle = self.block.angle % (2 * np.pi)
            return np.concatenate([agent_position, block_position, [block_angle]], dtype=np.float64)

        if self.obs_type == "environment_state_agent_pos":
            return {
                "environment_state": self.get_keypoints(self._block_shapes).flatten(),
                "agent_pos": np.array(self.agent.position),
            }

        pixels = self._render()
        if self.obs_type == "pixels":
            return pixels
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": pixels,
                "agent_pos": np.array(self.agent.position),
            }