import genesis as gs
import numpy as np
from gymnasium import spaces
import random
import torch
import pyroomacoustics as pra
import cv2

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

class SoundTask:
    def __init__(self, observation_height, observation_width):
        self.observation_height = observation_height
        self.observation_width = observation_width
        self._random = np.random.RandomState()
        self._build_scene()
        self.observation_space = self._make_obs_space()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(AGENT_DIM,), dtype=np.float32)

    def _build_scene(self):
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
            show_viewer=True,
        )
        # 平面を追加
        self.plane = self.scene.add_entity(morph=gs.morphs.Plane())
        # フランカロボットを追加
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        # キューブAを追加
        self.cubeA = self.scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)),
            surface=gs.surfaces.Aluminium(color=(0.3, 0.7, 0.3))
        )
        # キューブBを追加
        self.cubeB = self.scene.add_entity(
            gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.35, 0.0, 0.02)),
            surface=gs.surfaces.Aluminium(color=(0.3, 0.7, 0.3))
        )
        # 箱を追加
        self.box = self.scene.add_entity(gs.morphs.URDF(file="URDF/box/box.urdf", pos=(0.5, 0.0, 0.0)))
        # フロントカメラを追加
        self.front_cam = self.scene.add_camera(
            res=(self.observation_width, self.observation_height),
            pos=(2.5, 0.0, 1.5),
            lookat=(0.5, 0.0, 0.4),
            fov=30,
            GUI=False
        )
        # サイドカメラを追加
        self.side_cam = self.scene.add_camera(
            res=(self.observation_width, self.observation_height),
            pos=(0.5, 1.5, 1.5),
            lookat=(0.5, 0.0, 0.2),
            fov=30,
            GUI=False
        )
        # サウンドカメラを追加
        self.sound_cam = SoundCamera(
            self.scene,
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
    
    def reset(self):
        # CubeAの位置をランダムに設定
        x = self._random.uniform(0.3, 0.7)
        y = self._random.uniform(-0.3, 0.3)
        z = 0.02
        pos_tensor = torch.tensor([x, y, z], dtype=torch.float32, device=gs.device)
        quat_tensor = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=gs.device)
        self.cubeA.set_pos(pos_tensor)
        self.cubeA.set_quat(quat_tensor)
        # CubeBの位置をランダムに設定
        x = self._random.uniform(0.3, 0.7)
        y = self._random.uniform(-0.3, 0.3)
        z = 0.02
        pos_tensor = torch.tensor([x, y, z], dtype=torch.float32, device=gs.device)
        quat_tensor = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=gs.device)
        self.cubeB.set_pos(pos_tensor)
        self.cubeB.set_quat(quat_tensor)

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
        print(f"CubeA position: {cubeA_pos}, Box position: {box_pos}, Reward: {reward}")
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

class SoundCamera:
    def __init__(self, scene, observation_height, observation_width):
        self.scene = scene
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.frames = []
        # DOAパラメータ
        self.fs = 16000
        self.nfft = 256
        self.freq_range = [300, 3500]
        # シミュレーション設定
        self.mic_pos = [
            [0.8, 0.0, 0.1],
            [0.2, -0.3, 0.1],
            [0.2, 0.3, 0.1],
        ]
        corners = np.array([
            [-0.5, 1.0, 3.0],
            [1.5, 1.0, 3.0],
            [1.5, -1.0, 3.0],
            [-0.5, -1.0, 3.0],
        ])
        self.aroom = pra.Room.from_corners(
            corners,
            fs=self.fs,
            materials=None,
            max_order=3,
            sigma2_awgn=10**(1/2) / (4 * np.pi * 2)**2,
            air_absorption=True,
        )
        for mic in self.mic_pos:
            self.aroom.add_microphone_array(
                np.concatenate(
                    pra.circular_2D_array(center=[mic[0], mic[1]], M=8, phi0=0, radius=0.035)
                ),
                np.ones((1, 8)) * mic[2],
                axis=0,
            )

    def start_recording(self):
        sound_pos = self.scene.get_entity("cubeA").get_pos() if self.scene is not None else torch.tensor([0.5, 0.0, 0.1])
        # 既に音源がある場合は削除
        self.aroom.sources = []
        # 新しい音源を追加
        self.aroom.add_source(
            sound_pos.cpu().numpy(),
            signal=np.random.randn(self.fs * 2),
            delay=0,
        )

    def stop_recording(self, save_to_filename, fps):
        sound_image = np.array(self.frames)
        # cv2で動画を保存
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(save_to_filename, fourcc, fps, (self.observation_width, self.observation_height))
        for i in range(sound_image.shape[0]):
            out.write(sound_image[i])
        out.release()
        self.frames = []
        # 音源を削除
        self.aroom.sources = []

    def render(self):
        # CubeAが音を発していると仮定して、画像を生成
        sound_pos = self.scene.get_entity("cubeA").get_pos() if self.scene is not None else torch.tensor([0.5, 0.0, 0.1])
        # 音源を移動
        self.aroom.sources[0].position = sound_pos.cpu().numpy()
        self.aroom.simulate()
        sound_image = []
        for i, mic_array in enumerate(self.aroom.microphone_array):
            X = pra.transform.stft.analysis(mic_array.signals.T, self.nfft, self.nfft // 2)
            X = X.transpose([2, 1, 0])
            # DOAの計算
            doa = pra.doa.algorithms['MUSIC'](mic_array.R, self.fs, self.nfft, c=343., num_src=1, max_four=4)
            doa.locate_sources(X, freq_range=self.freq_range)
            spatial_resp = doa.grid.values
            # 画像上の各pixelのマイクロフォンアレイからの角度を計算してanglesに格納
            mic_coord = [int((0.8 - self.mic_pos[i][0])*480/0.6), int((0.4 + self.mic_pos[i][1])*640/0.8)]
            points = np.array(np.meshgrid(
                np.arrange(self.observation_height),
                np.arrange(self.observation_width),
            )).T.reshape(-1, 2)
            angles = (np.arctan2(points[:, 0] - mic_coord[0], points[:, 1] - mic_coord[1]) * 180 / np.pi + 180) % 360
            sound_map = np.zeros((self.observation_height, self.observation_width))
            for j in range(len(angles)):
                angle = angles[j]
                if angle < 0 or angle >= 360:
                    continue
                sound_map[points[j, 0], points[j, 1]] = spatial_resp[int(angle)]
            sound_image.append(sound_map)
        sound_image = np.array(sound_image)
        sound_image = np.clip(sound_image, 0, 255).astype(np.uint8)
        sound_image = np.transpose(sound_image, (1, 2, 0))
        self.frames.append(sound_image)
        return sound_image, None


if __name__ == "__main__":
    # SondCameraのテスト
    sound_camera = SoundCamera(None, 480, 640)
    sound_camera.start_recording()
    for i in range(10):
        sound_pixels = sound_camera.render()[0]
    sound_camera.stop_recording(save_to_filename="test_sound.mp4", fps=30)
    