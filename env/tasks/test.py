import genesis as gs
import numpy as np
from gymnasium import spaces
import random
import torch

joints_name = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "finger_joint1",
    "finger_joint2",
)
AGENT_DIM = len(joints_name)

class TestTask:
    def __init__(self, observation_height, observation_width, show_viewer=False):
        self.observation_height = observation_height
        self.observation_width = observation_width
        self._random = np.random.RandomState()
        self._build_scene(show_viewer)
        self.observation_space = self._make_obs_space()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(AGENT_DIM,), dtype=np.float32)

    def _build_scene(self, show_viewer):
        if not gs._initialized:
          gs.init(backend=gs.gpu, precision="32")
        # シーンを初期化
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -1, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(self.observation_width, self.observation_height),
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=show_viewer,
        )
        # 平面を追加
        self.plane = self.scene.add_entity(morph=gs.morphs.Plane())
        # フランカロボットを追加
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        # キューブAを追加
        self.cubeA = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.65, 0.0, 0.025)),
            surface=gs.surfaces.Aluminium(color=(0.7, 0.3, 0.3))
        )
        # キューブBを追加
        self.cubeB = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.35, 0.0, 0.025)),
            surface=gs.surfaces.Aluminium(color=(0.3, 0.3, 0.7))
        )
        # 箱を追加
        self.box = self.scene.add_entity(gs.morphs.URDF(file="URDF/box/box.urdf", pos=(0.5, 0.0, 0.0)))
        # フロントカメラを追加
        self.front_cam = self.scene.add_camera(
            res=(self.observation_width, self.observation_height),
            pos=(2.5, 0.0, 1.5),
            lookat=(0.5, 0.0, 0.1),
            fov=18,
            GUI=False
        )
        # サイドカメラを追加
        self.side_cam = self.scene.add_camera(
            res=(self.observation_width, self.observation_height),
            pos=(0.5, 1.5, 1.5),
            lookat=(0.5, 0.0, 0.0),
            fov=20,
            GUI=False
        )
        
        # サウンドカメラを追加
        self.sound_cam = DummyCamera(
            self.cubeA,
            observation_height=self.observation_height,
            observation_width=self.observation_width
        )

        self.scene.build()
        self.motors_dof = np.arange(7)
        self.fingers_dof = np.arange(7, 9)
        self.eef = self.franka.get_link("hand")

    def _make_obs_space(self):
        return spaces.Dict({
            "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(AGENT_DIM,), dtype=np.float32),
            "front": spaces.Box(low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8),
            "side": spaces.Box(low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8),
            "sound": spaces.Box(low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8),
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
        # 箱を初期位置に設定
        pos_tensor = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32, device=gs.device)
        quat_tensor = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=gs.device)
        self.box.set_pos(pos_tensor)
        self.box.set_quat(quat_tensor)
        # CubeAの位置をランダムに設定
        self.set_random_state(self.cubeA, (0.3, 0.7), (-0.3, 0.3), 0.04) # 1回は必ず呼び出す
        while self.compute_reward() == 1.0:
            print("CubeA is in the box, resetting position...")
            self.set_random_state(self.cubeA, (0.3, 0.7), (-0.3, 0.3), 0.04)
        # CubeBの位置をランダムに設定
        self.set_random_state(self.cubeB, (0.3, 0.7), (-0.3, 0.3), 0.04)
        # フランカロボットを初期位置にリセット
        qpos = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8, 0.04, 0.04])
        qpos_tensor = torch.tensor(qpos, dtype=torch.float32, device=gs.device)
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
        )
        self.franka.set_qpos(qpos_tensor, zero_velocity=True)
        self.franka.control_dofs_position(qpos_tensor[:7], self.motors_dof)
        self.franka.control_dofs_position(qpos_tensor[7:], self.fingers_dof)

        # ステップ実行
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
        self.franka.control_dofs_position(action_tensor[:7], self.motors_dof)
        self.franka.control_dofs_position(action_tensor[7:], self.fingers_dof)
        self.scene.step()
        reward = self.compute_reward()
        obs = self.get_obs()
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
    
    def compute_reward(self):
        # CubeAがboxの中にあるかどうかをチェック
        cubeA_pos = self.cubeA.get_pos().cpu().numpy()
        box_pos = self.box.get_pos().cpu().numpy()
        box_size = np.array([0.1, 0.1, 0.05])
        cubeA_in_box = (
            (box_pos[0] - box_size[0] / 2 <= cubeA_pos[0] <= box_pos[0] + box_size[0] / 2) and
            (box_pos[1] - box_size[1] / 2 <= cubeA_pos[1] <= box_pos[1] + box_size[1] / 2) and
            (box_pos[2] <= cubeA_pos[2] <= box_pos[2] + box_size[2])
        )
        reward = 1.0 if cubeA_in_box else 0.0
        return reward

    def get_obs(self):
        # ロボットの状態を取得
        eef_pos = self.eef.get_pos().cpu().numpy()
        eef_rot = self.eef.get_quat().cpu().numpy()
        gripper = self.franka.get_dofs_position()[7:9].cpu().numpy()
        agent_pos = np.concatenate([eef_pos, eef_rot, gripper])
        # frontカメラの画像を取得
        front_pixels = self.front_cam.render()[0]
        assert front_pixels.ndim == 3, f"front_pixels shape {front_pixels.shape} is not 3D (H, W, 3)"
        # sideカメラの画像を取得
        side_pixels = self.side_cam.render()[0]
        assert side_pixels.ndim == 3, f"side_pixels shape {side_pixels.shape} is not 3D (H, W, 3)"
        # soundカメラの画像を取得
        sound_pixels = self.sound_cam.render()[0]
        assert sound_pixels.ndim == 3, f"sound_pixels shape {sound_pixels.shape} is not 3D (H, W, 3)"
        obs = {
            "agent_pos": agent_pos,
            "front": front_pixels,
            "side": side_pixels,
            "sound": sound_pixels,
        }
        return obs

    def save_videos(self, file_name, fps=30):
        self.front_cam.stop_recording(save_to_filename=f"{file_name}_front.mp4", fps=fps)
        self.side_cam.stop_recording(save_to_filename=f"{file_name}_side.mp4", fps=fps)
        self.sound_cam.stop_recording(save_to_filename=f"{file_name}_sound.mp4", fps=fps)

class DummyCamera:
    def __init__(self, target, observation_height, observation_width):
        self.observation_height = observation_height
        self.observation_width = observation_width
    def start_recording(self):
        pass
    def stop_recording(self, save_to_filename, fps):
        pass
    def render(self):
        # ダミーの画像を生成
        dummy_image = np.zeros((self.observation_height, self.observation_width, 3), dtype=np.uint8)
        return dummy_image, None

if __name__ == "__main__":
    import cv2
    gs.init(backend=gs.gpu, precision="32")
    task = TestTask(observation_height=480, observation_width=640, show_viewer=False)
    task.reset()
    for _ in range(100):
        action = np.random.uniform(-1.0, 1.0, size=(AGENT_DIM,))
        task.step(action)
    # 最後の画像を保存
    # obs = task.get_obs()
    # for key, value in obs.items():
    #     if key == "agent_pos" or key == "sound":
    #         continue
    #     # rgbの入れ替え
    #     if value.shape[2] == 3:
    #         value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
    #     print(f"{key}: {value.shape}")
    #     cv2.imwrite(f"{key}.png", value)