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
            surface=gs.surfaces.Aluminium(color=(0.3, 0.7, 0.3))
        )
        # キューブBを追加
        self.cubeB = self.scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.35, 0.0, 0.025)),
            surface=gs.surfaces.Aluminium(color=(0.3, 0.7, 0.3))
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
        self.sound_cam = SoundCamera(
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
        x = self._random.uniform(x_range[0], x_range[1])
        y = self._random.uniform(y_range[0], y_range[1])
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

class SoundCamera:
    def __init__(self, target, observation_height, observation_width):
        self.target = target
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
        # cornersを2次元に変更し、転置する
        self.corners = np.array([
            [-0.5, 1.0],
            [1.5, 1.0],
            [1.5, -1.0],
            [-0.5, -1.0],
        ]).T

    def start_recording(self):
        pass

    def stop_recording(self, save_to_filename, fps):
        sound_image = np.array(self.frames)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(save_to_filename, fourcc, fps, (self.observation_width, self.observation_height))
        for i in range(sound_image.shape[0]):
            frame_to_write = sound_image[i]
            if frame_to_write.dtype != np.uint8:
                 frame_to_write = frame_to_write.astype(np.uint8)
            out.write(frame_to_write)
        out.release()
        self.frames = []

    def render(self):
        # CubeAが音を発していると仮定して、画像を生成
        sound_pos = self.target.get_pos() if self.target is not None else torch.tensor([0.5, 0.3, 0.1])
        sound_image = []
        for i in range(3):
            try:
                aroom = pra.Room.from_corners(
                    self.corners,
                    fs=self.fs,
                    materials=None,
                    max_order=3,
                    sigma2_awgn=10**(1/2) / (4 * np.pi * 2)**2,
                    air_absorption=True,
                )
                aroom.extrude(3.0) # 部屋の高さを3mに設定
                # マイクロフォンアレイを追加
                aroom.add_microphone_array(
                    np.concatenate(
                        ( # Add parentheses here
                            pra.circular_2D_array(center=[self.mic_pos[i][0], self.mic_pos[i][1]], M=8, phi0=0, radius=0.035),
                            np.ones((1, 8)) * self.mic_pos[i][2]
                        ), # Add parentheses here
                        axis=0,
                    ),
                )
                aroom.add_source(
                    sound_pos.cpu().numpy(),
                    signal=np.random.randn(self.fs), # 1秒間のホワイトノイズ
                    delay=0,
                )
                aroom.simulate()
                X = pra.transform.stft.analysis(aroom.mic_array.signals.T, self.nfft, self.nfft // 2)
                X = X.transpose([2, 1, 0])
                # DOAの計算
                doa = pra.doa.algorithms['MUSIC'](aroom.mic_array.R, self.fs, self.nfft, c=343., num_src=1, max_four=4)
                doa.locate_sources(X, freq_range=self.freq_range)
                spatial_resp = doa.grid.values * 25
                # 画像上の各pixelのマイクロフォンアレイからの角度を計算してanglesに格納
                mic_coord = [int((0.8 - self.mic_pos[i][0])*self.observation_height/0.6), int((0.4 + self.mic_pos[i][1])*self.observation_width/0.8)]
                points = np.array(np.meshgrid(
                    np.arange(self.observation_height),
                    np.arange(self.observation_width),
                )).T.reshape(-1, 2)
                angles = (np.arctan2(points[:, 0] - mic_coord[0], points[:, 1] - mic_coord[1]) * 180 / np.pi + 90) % 360
                sound_map = np.zeros((self.observation_height, self.observation_width))
                for j, angle in enumerate(angles): # Changed loop variable from i to j to avoid conflict
                    sound_map[points[j, 0], points[j, 1]] = spatial_resp[int(angle)]
                sound_image.append(sound_map)
            except ValueError as e:
                if "The source must be added inside the room." in str(e):
                    print(f"Warning: Sound source is outside the room. Skipping sound simulation for mic {i}. Error: {e}")
                    sound_image.append(np.zeros((self.observation_height, self.observation_width)))
                else:
                    raise e # Re-raise other ValueErrors

        if not sound_image: # Handle case where all simulations failed
            print("Warning: All sound simulations failed. Returning zero array.")
            sound_image_array = np.zeros((self.observation_height, self.observation_width, 3), dtype=np.uint8)
            self.frames.append(sound_image_array)
            return sound_image_array, None

        sound_image_array = np.array(sound_image)
        # y軸を反転
        sound_image_array = np.flip(sound_image_array, axis=2)
        sound_image_array = np.clip(sound_image_array, 0, 255).astype(np.uint8)
        sound_image_array = np.transpose(sound_image_array, (1, 2, 0))
        self.frames.append(sound_image_array)
        return sound_image_array, None

if __name__ == "__main__":
    # SondCameraのテスト
    sound_camera = SoundCamera(None, 480, 640)
    sound_camera.start_recording()
    for i in range(10):
        sound_pixels = sound_camera.render()[0]
        # 画像を保存
        cv2.imwrite(f"test_sound_{i}.png", sound_pixels)
    # 出力ファイル名が .mp4 であることを確認
    output_filename = "test_sound.mp4"
    sound_camera.stop_recording(save_to_filename=output_filename, fps=30)
    print(f"Test video saved to {output_filename}")
