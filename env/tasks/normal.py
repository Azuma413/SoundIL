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

class NormalTask:
    def __init__(self, observation_height, observation_width, show_viewer=False, device="cuda", same_color=False, fix_color=False, num_cubes=3, use_two_boxes=False):
        self.device = device
        self.same_color = same_color
        self.fix_color = fix_color
        self.num_cubes = num_cubes
        self.use_two_boxes = use_two_boxes
        self.show_viewer = show_viewer
        self.observation_height = observation_height
        self.observation_width = observation_width
        self._random = np.random.RandomState()
        self.box_scale = 0.75
        self._build_scene(show_viewer)
        self.observation_space = self._make_obs_space()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(AGENT_DIM,), dtype=np.float32)

    def _build_scene(self, show_viewer):
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
        self.so_arm = self.scene.add_entity(gs.morphs.MJCF(file="URDF/so101/so101_new_calib.xml"))
        if self.num_cubes >= 1:
            self.cubeR = self.scene.add_entity(
                gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.45, 0.0, 0.02)),
                material=gs.materials.Rigid(rho=50, friction=1.5, coup_friction=1.0, coup_softness=0.001),
                surface=gs.surfaces.Plastic(color=(0.7, 0.3, 0.3)) if not self.same_color else gs.surfaces.Plastic(color=(0.3, 0.7, 0.3)),
            )
        if self.num_cubes >= 2:
            self.cubeG = self.scene.add_entity(
                gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.30, 0.0, 0.02)),
                material=gs.materials.Rigid(rho=50, friction=1.5, coup_friction=1.0, coup_softness=0.001),
                surface=gs.surfaces.Plastic(color=(0.3, 0.7, 0.3))
            )
        if self.num_cubes >= 3:
            self.cubeB = self.scene.add_entity(
                gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0.15, 0.0, 0.02)),
                material=gs.materials.Rigid(rho=50, friction=1.5, coup_friction=1.0, coup_softness=0.001),
                surface=gs.surfaces.Plastic(color=(0.3, 0.3, 0.7)) if not self.same_color else gs.surfaces.Plastic(color=(0.3, 0.7, 0.3)),
            )
        
        if self.use_two_boxes:
            self.box_left = self.scene.add_entity(
                gs.morphs.URDF(file="URDF/box/box.urdf", pos=(0.3, 0.15, 0.0), scale=self.box_scale),
                surface=gs.surfaces.Plastic(color=(0.8, 0.8, 0.8))
            )
            self.box_right = self.scene.add_entity(
                gs.morphs.URDF(file="URDF/box/box.urdf", pos=(0.3, -0.15, 0.0), scale=self.box_scale),
                surface=gs.surfaces.Plastic(color=(0.8, 0.8, 0.8))
            )
            # 互換性のためにself.boxも定義しておく（デフォルトは左にしておくが、タスクによって使い分ける）
            self.box = self.box_left 
        else:
            self.box = self.scene.add_entity(
                gs.morphs.URDF(file="URDF/box/box.urdf", pos=(0.3, 0.0, 0.0), scale=self.box_scale),
                surface=gs.surfaces.Plastic(color=(0.8, 0.8, 0.8))
            )
        self.front_cam = self.scene.add_camera(
            res=(self.observation_width, self.observation_height),
            pos=(1.2, 0.0, 0.4),
            lookat=(0.3, 0.0, 0.15),
            fov=25,
            GUI=False
        )
        self.side_cam = self.scene.add_camera(
            res=(self.observation_width, self.observation_height),
            pos=(0.15, 0.8, 1.0),
            lookat=(0.15, 0.0, 0.05),
            fov=20,
            GUI=False
        )
        self.scene.build()
        self.motors_dof = np.arange(5)
        self.fingers_dof = np.arange(5, 6)
        self.eef = self.so_arm.get_link("gripper")

    def _make_obs_space(self):
        return spaces.Dict({
            "observation.state": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
            "observation.images.front": spaces.Box(low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8),
            "observation.images.side": spaces.Box(low=0, high=255, shape=(self.observation_height, self.observation_width, 3), dtype=np.uint8),
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
        if self.fix_color:
            self.color = "red"
        else:
            self.color = random.choice(["red", "green", "blue"])
        if self.use_two_boxes:
            pos_tensor_l = torch.tensor([0.3, 0.15, 0.0], dtype=torch.float32, device=gs.device)
            pos_tensor_r = torch.tensor([0.3, -0.15, 0.0], dtype=torch.float32, device=gs.device)
            quat_tensor = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=gs.device)
            self.box_left.set_pos(pos_tensor_l)
            self.box_left.set_quat(quat_tensor)
            self.box_right.set_pos(pos_tensor_r)
            self.box_right.set_quat(quat_tensor)
        else:
            pos_tensor = torch.tensor([0.3, 0.0, 0.0], dtype=torch.float32, device=gs.device)
            quat_tensor = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=gs.device)
            self.box.set_pos(pos_tensor)
            self.box.set_quat(quat_tensor)
        
        # CubeR
        if self.num_cubes >= 1:
            self.set_random_state(self.cubeR, (0.15, 0.3), (-0.2, 0.2), 0.02)
            while self.compute_reward(target="cubeR") == 1.0:
                print("CubeR is in the box, resetting position...")
                self.set_random_state(self.cubeR, (0.15, 0.3), (-0.2, 0.2), 0.02)
        
        # CubeG
        if self.num_cubes >= 2:
            self.set_random_state(self.cubeG, (0.15, 0.3), (-0.2, 0.2), 0.02)
            while self.compute_reward(target="cubeG") == 1.0:
                print("CubeG is in the box, resetting position...")
                self.set_random_state(self.cubeG, (0.15, 0.3), (-0.2, 0.2), 0.02)
        
        # CubeB
        if self.num_cubes >= 3:
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

    def compute_reward(self, target=None, target_box=None):
        # CubeがBoxの中にあるかどうかを判定
        if target is not None:
            if target == "cubeR":
                pos = self.cubeR.get_pos().cpu().numpy()
            elif target == "cubeG":
                pos = self.cubeG.get_pos().cpu().numpy()
            elif target == "cubeB":
                pos = self.cubeB.get_pos().cpu().numpy()
            else:
                raise ValueError(f"Invalid target: {target}. Choose from 'cubeR', 'cubeG', or 'cubeB'.")
        else:
            if self.color == "red":
                pos = self.cubeR.get_pos().cpu().numpy()
            elif self.color == "blue":
                pos = self.cubeB.get_pos().cpu().numpy()
            elif self.color == "green":
                pos = self.cubeG.get_pos().cpu().numpy()
            else:
                raise ValueError(f"Invalid color: {self.color}. Choose from 'red', 'blue', or 'green'.")
        
        if target_box is not None:
            box_pos = target_box.get_pos().cpu().numpy()
        elif self.use_two_boxes:
            # デフォルトではどちらかの箱に入っていればOKとする（soundDiffでオーバーライド可能）
            box_pos_l = self.box_left.get_pos().cpu().numpy()
            box_pos_r = self.box_right.get_pos().cpu().numpy()
            box_size = np.array([0.1, 0.1, 0.05])*self.box_scale
            in_left = (
                (box_pos_l[0] - box_size[0] / 2 <= pos[0] <= box_pos_l[0] + box_size[0] / 2) and
                (box_pos_l[1] - box_size[1] / 2 <= pos[1] <= box_pos_l[1] + box_size[1] / 2) and
                (box_pos_l[2] <= pos[2] <= box_pos_l[2] + box_size[2])
            )
            in_right = (
                (box_pos_r[0] - box_size[0] / 2 <= pos[0] <= box_pos_r[0] + box_size[0] / 2) and
                (box_pos_r[1] - box_size[1] / 2 <= pos[1] <= box_pos_r[1] + box_size[1] / 2) and
                (box_pos_r[2] <= pos[2] <= box_pos_r[2] + box_size[2])
            )
            return 1.0 if (in_left or in_right) else 0.0
        else:
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
        obs = {
            "observation.state": agent_pos,
            "observation.images.front": front_pixels,
            "observation.images.side": side_pixels,
        }
        return obs

    def save_videos(self, file_name, fps=30):
        self.front_cam.stop_recording(save_to_filename=f"{file_name}_front.mp4", fps=fps)
        self.side_cam.stop_recording(save_to_filename=f"{file_name}_side.mp4", fps=fps)

    def close(self):
        gs.destroy()
    
    def get_task_description(self):
        return f"Pick up a {self.color} cube and place it in a box."

if __name__ == "__main__":
    import cv2
    gs.init(backend=gs.gpu, precision="32")
    task = NormalTask(observation_height=224, observation_width=224, show_viewer=False)
    task.reset()
    print("box pos: ", task.box.get_pos().cpu().numpy())
    for _ in range(10):
        action = np.random.uniform(-1.0, 1.0, size=(AGENT_DIM,))
        task.step(action)
    # 最後の画像を保存
    obs = task.get_obs()
    for key, value in obs.items():
        if key == "observation.state":
            continue
        # rgbの入れ替え
        if value.shape[2] == 3:
            value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
        print(f"{key}: {value.shape}")
        cv2.imwrite(f"images/{key}.png", value)
