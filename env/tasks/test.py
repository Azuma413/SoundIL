import genesis as gs
import numpy as np
from gymnasium import spaces
import random
import torch

joints_name = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
AGENT_DIM = len(joints_name)

class TestTask:
    def __init__(self, observation_height, observation_width, show_viewer=False, dummy=False, device="cuda"):
        self.device = device
        self.show_viewer = show_viewer
        self.observation_height = observation_height
        self.observation_width = observation_width
        self._random = np.random.RandomState()
        self.box_scale = 0.75
        self._build_scene(show_viewer, dummy)
        self.observation_space = self._make_obs_space()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

    def _build_scene(self, show_viewer, dummy):
        if not gs._initialized:
            print("Genesis is not initialized, initializing now...")
            if self.device == "cuda":
                gs.init(backend=gs.gpu, precision="32", debug=False, logging_level="WARNING")
            elif self.device == "cpu":
                gs.init(backend=gs.cpu, precision="32", debug=False, logging_level="WARNING")
            else:
                raise ValueError(f"Unsupported device: {self.device}. Use 'cuda' or 'cpu'.")
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(self.observation_width, self.observation_height),
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(
                box_box_detection=True,
                noslip_iterations=5,
                constraint_timeconst=0.001,
                # integrator=gs.integrator.implicitfast,
            ),
            show_viewer=show_viewer,
        )
        self.plane = self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Plastic(diffuse_texture=gs.textures.ImageTexture(image_path="images/wood.jpg"))
        )
        self.so_arm = self.scene.add_entity(gs.morphs.MJCF(file="URDF/so101_new_calib.xml"))
        self.cubeA = self.scene.add_entity(
            gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.45, 0.0, 0.02)),
            material=gs.materials.Rigid(rho=50, friction=1.5, coup_friction=1.0, coup_softness=0.001),
            surface=gs.surfaces.Plastic(color=(0.3, 0.7, 0.3)) if dummy else gs.surfaces.Plastic(color=(0.7, 0.3, 0.3))
        )
        self.cubeB = self.scene.add_entity(
            gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.15, 0.0, 0.02)),
            material=gs.materials.Rigid(rho=50, friction=1.5, coup_friction=1.0, coup_softness=0.001),
            surface=gs.surfaces.Plastic(color=(0.3, 0.7, 0.3)) if dummy else gs.surfaces.Plastic(color=(0.3, 0.3, 0.7))
        )
        self.box = self.scene.add_entity(
            gs.morphs.URDF(file="URDF/box/box.urdf", pos=(0.3, 0.0, 0.0), scale=self.box_scale),
            surface=gs.surfaces.Plastic(color=(0.8, 0.8, 0.8))
        )
        self.front_cam = self.scene.add_camera(
            res=(self.observation_width, self.observation_height),
            pos=(1.5, 0.0, 0.4),
            lookat=(0.3, 0.0, 0.1),
            fov=20,
            GUI=False
        )
        self.side_cam = self.scene.add_camera(
            res=(self.observation_width, self.observation_height),
            pos=(0.3, 1.0, 1.0),
            lookat=(0.3, 0.0, 0.05),
            fov=20,
            GUI=False
        )
        self.sound_cam = DummyCamera(
            self.cubeA,
            observation_height=self.observation_height,
            observation_width=self.observation_width
        )
        self.scene.build()
        self.motors_dof = np.arange(5)
        self.fingers_dof = np.arange(5, 6)
        self.eef = self.so_arm.get_link("gripper")

    def _make_obs_space(self):
        return spaces.Dict({
            "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(AGENT_DIM,), dtype=np.float32),
            "observation.images.front": spaces.Box(low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8),
            "observation.images.side": spaces.Box(low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8),
            "observation.images.sound": spaces.Box(low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8),
        })

    def set_random_state(self, target, x_range, y_range, z):
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        z = z
        pos_tensor = torch.tensor([x, y, z], dtype=torch.float32, device=gs.device)
        quat_tensor = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=gs.device)
        target.set_pos(pos_tensor)
        target.set_quat(quat_tensor)

    def reset(self):
        pos_tensor = torch.tensor([0.3, 0.0, 0.0], dtype=torch.float32, device=gs.device)
        quat_tensor = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=gs.device)
        self.box.set_pos(pos_tensor)
        self.box.set_quat(quat_tensor)
        self.set_random_state(self.cubeA, (0.15, 0.3), (-0.2, 0.2), 0.02) # 1回は必ず呼び出す
        while self.compute_reward() == 1.0:
            print("CubeA is in the box, resetting position...")
            self.set_random_state(self.cubeA, (0.15, 0.3), (-0.2, 0.2), 0.02)
        self.set_random_state(self.cubeB, (0.15, 0.3), (-0.2, 0.2), 0.02)
        while self.compute_reward(target="cubeB") == 1.0:
            print("CubeB is in the box, resetting position...")
            self.set_random_state(self.cubeB, (0.15, 0.3), (-0.2, 0.2), 0.02)
        qpos = np.array([0.0, -np.pi/2, np.pi/2, 1.0, 0.0, 0.04])
        qpos_tensor = torch.tensor(qpos, dtype=torch.float32, device=gs.device)
        self.so_arm.set_dofs_kp(
            np.array([500, 500, 500, 500, 500, 100]),
        )
        self.so_arm.set_dofs_kv(
            np.array([100, 100, 100, 100, 100, 100]),
        )
        self.so_arm.set_dofs_force_range(
            np.array([-20, -20, -20, -10, -10, -5]),
            np.array([ 20,  20,  20,  10,  10,  5]),
        )
        self.so_arm.set_qpos(qpos_tensor, zero_velocity=True)
        self.so_arm.control_dofs_position(qpos_tensor[:5], self.motors_dof)
        self.so_arm.control_dofs_position(qpos_tensor[5:], self.fingers_dof)
        self.scene.step()
        self.front_cam.start_recording()
        self.side_cam.start_recording()
        self.sound_cam.start_recording()
        return self.get_obs(), {}

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        self._random = np.random.RandomState(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.action_space.seed(seed)

    def step(self, action):
        action_tensor = torch.tensor(action, dtype=torch.float32, device=gs.device)
        self.so_arm.control_dofs_position(action_tensor[:5], self.motors_dof)
        self.so_arm.control_dofs_position(action_tensor[5:], self.fingers_dof)
        self.scene.step()
        reward = self.compute_reward()
        obs = self.get_obs()
        terminated = True if reward == 1.0 else False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def compute_reward(self, target="cubeA"):
        # CubeAがboxの中にあるかどうかをチェック
        if target == "cubeA":
            pos = self.cubeA.get_pos().cpu().numpy()
        elif target == "cubeB":
            pos = self.cubeB.get_pos().cpu().numpy()
        box_pos = self.box.get_pos().cpu().numpy()
        box_size = np.array([0.1, 0.1, 0.05])*self.box_scale  # Boxのサイズを取得
        cube_in_box = (
            (box_pos[0] - box_size[0] / 2 <= pos[0] <= box_pos[0] + box_size[0] / 2) and
            (box_pos[1] - box_size[1] / 2 <= pos[1] <= box_pos[1] + box_size[1] / 2) and
            (box_pos[2] <= pos[2] <= box_pos[2] + box_size[2])
        )
        reward = 1.0 if cube_in_box else 0.0
        return reward

    def get_obs(self):
        eef_pos = self.eef.get_pos().cpu().numpy() # 3次元
        eef_rot = self.eef.get_quat().cpu().numpy() # 4次元
        gripper = self.so_arm.get_dofs_position()[5:].cpu().numpy() # 1次元
        agent_pos = np.concatenate([eef_pos, eef_rot, gripper]) # 8次元
        front_pixels = self.front_cam.render()[0]
        assert front_pixels.ndim == 3, f"front_pixels shape {front_pixels.shape} is not 3D (H, W, 3)"
        side_pixels = self.side_cam.render()[0]
        assert side_pixels.ndim == 3, f"side_pixels shape {side_pixels.shape} is not 3D (H, W, 3)"
        sound_pixels = self.sound_cam.render()[0]
        assert sound_pixels.ndim == 3, f"sound_pixels shape {sound_pixels.shape} is not 3D (H, W, 3)"
        obs = {
            "agent_pos": agent_pos,
            "observation.images.front": front_pixels,
            "observation.images.side": side_pixels,
            "observation.images.sound": sound_pixels,
        }
        return obs

    def save_videos(self, file_name, fps=30):
        self.front_cam.stop_recording(save_to_filename=f"{file_name}_front.mp4", fps=fps)
        self.side_cam.stop_recording(save_to_filename=f"{file_name}_side.mp4", fps=fps)
        self.sound_cam.stop_recording(save_to_filename=f"{file_name}_sound.mp4", fps=fps)

    def close(self):
        gs.destroy()
class DummyCamera:
    def __init__(self, target, observation_height, observation_width):
        self.observation_height = observation_height
        self.observation_width = observation_width
    def start_recording(self):
        pass
    def stop_recording(self, save_to_filename, fps):
        pass
    def render(self):
        dummy_image = np.zeros((self.observation_height, self.observation_width, 3), dtype=np.uint8)
        return dummy_image, None

if __name__ == "__main__":
    import cv2
    gs.init(backend=gs.gpu, precision="32")
    task = TestTask(observation_height=480, observation_width=640, show_viewer=False)
    task.reset()
    # for _ in range(100):
    #     action = np.random.uniform(-1.0, 1.0, size=(AGENT_DIM,))
    #     task.step(action)
    # 最後の画像を保存
    obs = task.get_obs()
    for key, value in obs.items():
        if key == "agent_pos" or key == "sound":
            continue
        # rgbの入れ替え
        if value.shape[2] == 3:
            value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
        print(f"{key}: {value.shape}")
        cv2.imwrite(f"images/{key}.png", value)
