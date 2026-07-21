import genesis as gs
import numpy as np
from gymnasium import spaces
from s2a2.env.tasks.normal import NormalTask, AGENT_DIM
from s2a2.env.tasks.sound_camera import SoundCamera, SoundConfig

SOUND_A_PATH = "sounds/0.wav"
SOUND_B_PATH = "sounds/1.wav"
SOUND_C_PATH = "sounds/2.wav"
SOUND_DIFF_B_PATH = "sounds/3.wav"


class SoundTask(NormalTask):
    """
    Task environment that wraps NormalTask and adds acoustic observations via SoundCamera.
    """
    
    def __init__(
        self,
        observation_height,
        observation_width,
        show_viewer=False,
        device="cuda",
        sound_config: SoundConfig = None,
        task_type="sound" # "sound", "soundDiff", "soundShake", "soundAll", "soundSim"
    ):
        self.task_type = task_type
        # Use default values if sound_config is None
        if sound_config is None:
            sound_config = SoundConfig()
        
        sound_config.observation_height = observation_height
        sound_config.observation_width = observation_width
        self.sound_config = sound_config
        
        # Settings per task type
        num_cubes = 2 # default for sound and soundShake
        use_two_boxes = False
        same_color = True # default for sound and soundShake
        
        if task_type == "sound":
            num_cubes = 2
            same_color = True
        elif task_type == "soundDiff":
            num_cubes = 1
            use_two_boxes = True
            same_color = False # color is irrelevant; any value works since there is only one cube
        elif task_type == "soundShake":
            num_cubes = 2
            same_color = True
            self.sound_config.shake_mode = True
        elif task_type == "soundAll":
            num_cubes = 2
            use_two_boxes = True
            same_color = True
            self.sound_config.sound_all_mode = True
            self.sound_config.audio_file_path = None # when stationary: sound A
        elif task_type == "soundSim":
            num_cubes = 2
            use_two_boxes = True
            same_color = True
            self.sound_config.sound_all_mode = False

        self.target_cube_name = None
        self.current_sound_type = "Unknown"
        self.target_box = None

        # Initialize the parent class (_build_scene is called)
        super().__init__(
            observation_height=observation_height,
            observation_width=observation_width,
            show_viewer=show_viewer,
            device=device,
            same_color=same_color,
            num_cubes=num_cubes,
            use_two_boxes=use_two_boxes
        )

    def _reset_soundshake_cube_layout(self):
        # In soundShake, fix left/right positions so that the first pick from the left
        # is correct/incorrect 50/50 based only on target_cube_name
        self.set_random_state(self.cubeR, (0.3, 0.7), (0.05, 0.3), 0.04)
        self.set_random_state(self.cubeG, (0.3, 0.7), (-0.3, -0.05), 0.04)
    
    def _build_scene(self, show_viewer):
        """
        Override NormalTask._build_scene to add SoundCamera.
        """
        super()._build_scene(show_viewer)
        # Initial target is cubeR (changed on reset)
        # For soundDiff, only cubeR exists
        self.sound_cam = SoundCamera(
            target=self.cubeR,
            config=self.sound_config
        )
    
    def _make_obs_space(self):
        """
        Extend the observation space to add sound0, sound1, and spec (when use_spectrogram is True).
        """
        obs_space_dict = {
            "observation.state": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
            "observation.images.front": spaces.Box(
                low=0, high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8
            ),
            "observation.images.side": spaces.Box(
                low=0, high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8
            ),
        }
        
        if self.sound_config.use_soundmap:
            obs_space_dict["observation.images.sound0"] = spaces.Box(
                low=0, high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8
            )
            obs_space_dict["observation.images.sound1"] = spaces.Box(
                low=0, high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8
            )
        
        # When use_spectrogram is True, add spec (3 channels)
        if self.sound_config.use_spectrogram:
            obs_space_dict["observation.images.spec"] = spaces.Box(
                low=0, high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8
            )
        
        return spaces.Dict(obs_space_dict)
    
    def reset(self, options=None):
        """
        Reset the environment and update the sound source target.
        """
        options = options or {}
        # Call the parent class reset (self.color is determined)
        obs, info = super().reset(options=options)

        if self.task_type == "sound":
            # One of the two cubes becomes the sound source
            # same_color=True, so they look identical
            # color is randomly chosen from red, green, blue, but
            # num_cubes=2, so cubeR and cubeG exist
            # The logic to decide the target from self.color depends on the color randomly chosen in the parent reset
            # However, in NormalTask.reset the color is random, but with same_color=True the appearance is the same
            # Here, choose the target randomly
            target_name = options.get("target_cube_name")
            if target_name is None:
                target_name = np.random.choice(["cubeR", "cubeG"])
            elif target_name not in ["cubeR", "cubeG"]:
                raise ValueError(f"Invalid target_cube_name: {target_name}")
            if target_name == "cubeR":
                self.sound_cam.target = self.cubeR
                self.target_cube_name = "cubeR"
            else:
                self.sound_cam.target = self.cubeG
                self.target_cube_name = "cubeG"
                
        elif self.task_type == "soundDiff":
            # The sound source is cubeR (there is only one)
            self.sound_cam.target = self.cubeR
            self.target_cube_name = "cubeR"

            # Randomly choose the sound type (A or B)
            sound_type = options.get("sound_type")
            if sound_type is None:
                sound_type = np.random.choice(["A", "B"])
            elif sound_type not in ["A", "B"]:
                raise ValueError(f"Invalid sound_type: {sound_type}")
            if sound_type == "A":
                # Sound A settings
                self.sound_cam._load_audio_file(SOUND_A_PATH)
                self.target_box = self.box_right # sound A -> right box
            else:
                # Sound B settings
                self.sound_cam._load_audio_file(SOUND_DIFF_B_PATH)
                self.target_box = self.box_left # sound B -> left box

            # For updating the task description
            self.current_sound_type = sound_type

        elif self.task_type == "soundShake":
            self._reset_soundshake_cube_layout()

            # One of the two cubes is the target
            target_name = options.get("target_cube_name")
            if target_name is None:
                target_name = np.random.choice(["cubeR", "cubeG"])
            elif target_name not in ["cubeR", "cubeG"]:
                raise ValueError(f"Invalid target_cube_name: {target_name}")
            if target_name == "cubeR":
                self.sound_cam.target = self.cubeR
                self.target_cube_name = "cubeR"
            else:
                self.sound_cam.target = self.cubeG
                self.target_cube_name = "cubeG"

        elif self.task_type == "soundAll":
            # One of the two cubes is the sound source (sound A: 0.wav)
            target_name = options.get("target_cube_name")
            if target_name is None:
                target_name = np.random.choice(["cubeR", "cubeG"])
            elif target_name not in ["cubeR", "cubeG"]:
                raise ValueError(f"Invalid target_cube_name: {target_name}")
            if target_name == "cubeR":
                self.sound_cam.target = self.cubeR
                self.target_cube_name = "cubeR"
            else:
                self.sound_cam.target = self.cubeG
                self.target_cube_name = "cubeG"
            
            # Randomly choose the sound played while moving (sound B or sound C)
            sound_type = options.get("sound_type")
            if sound_type is None:
                sound_type = np.random.choice(["B", "C"])
            elif sound_type not in ["B", "C"]:
                raise ValueError(f"Invalid sound_type: {sound_type}")
            if sound_type == "B":
                # sound B (1.wav) -> right box
                self.sound_cam.set_moving_audio(SOUND_B_PATH)
                self.target_box = self.box_right
            else:
                # sound C (2.wav) -> left box
                self.sound_cam.set_moving_audio(SOUND_C_PATH)
                self.target_box = self.box_left
            
            self.current_sound_type = sound_type

        elif self.task_type == "soundSim":
            target_name = options.get("target_cube_name")
            if target_name is None:
                target_name = np.random.choice(["cubeR", "cubeG"])
            elif target_name not in ["cubeR", "cubeG"]:
                raise ValueError(f"Invalid target_cube_name: {target_name}")
            if target_name == "cubeR":
                self.sound_cam.target = self.cubeR
                self.target_cube_name = "cubeR"
            else:
                self.sound_cam.target = self.cubeG
                self.target_cube_name = "cubeG"

            sound_type = options.get("sound_type")
            if sound_type is None:
                sound_type = np.random.choice(["A", "B"])
            elif sound_type not in ["A", "B"]:
                raise ValueError(f"Invalid sound_type: {sound_type}")
            if sound_type == "A":
                self.sound_cam._load_audio_file(SOUND_A_PATH)
                self.target_box = self.box_right
            else:
                self.sound_cam._load_audio_file(SOUND_B_PATH)
                self.target_box = self.box_left

            self.current_sound_type = sound_type
        
        return self.get_obs(), info
    
    def compute_reward(self, target=None, target_box=None, custom_pos=None):
        if self.task_type == "soundDiff":
            # Check whether it is in the specified box
            return super().compute_reward(target="cubeR", target_box=self.target_box, custom_pos=custom_pos)
        elif self.task_type in ["soundAll", "soundSim"]:
            # Check whether the target cube is in the specified box
            return super().compute_reward(target=self.target_cube_name, target_box=self.target_box, custom_pos=custom_pos)
        else:
            # For sound and soundShake, check whether the target cube is in (any) box
            # NormalTask.compute_reward checks whether the given cube is in a box
            # target_box=None means either box is OK (per the NormalTask fix)
            actual_target = target if target is not None else self.target_cube_name
            return super().compute_reward(target=actual_target, target_box=target_box, custom_pos=custom_pos)
    
    def get_obs(self):
        """
        Get the observation (including sound0, sound1, spec).
        """
        obs = super().get_obs()
        # Get the sound map and spectrogram from SoundCamera
        sound_map0, sound_map1, spectrogram = self.sound_cam.render()

        # Store sound0 and sound1 (both 3 channels)
        if self.sound_config.use_soundmap:
            assert sound_map0.ndim == 3 and sound_map0.shape[2] == 3, \
                f"sound_map0 shape {sound_map0.shape} is not (H, W, 3)"
            assert sound_map1.ndim == 3 and sound_map1.shape[2] == 3, \
                f"sound_map1 shape {sound_map1.shape} is not (H, W, 3)"
            
            obs["observation.images.sound0"] = sound_map0
            obs["observation.images.sound1"] = sound_map1
        
        # When use_spectrogram is True, add spec (3 channels)
        if self.sound_config.use_spectrogram:
            assert spectrogram is not None, "Spectrogram is None despite use_spectrogram=True"
            assert spectrogram.ndim == 3 and spectrogram.shape[2] == 3, \
                f"spectrogram shape {spectrogram.shape} is not (H, W, 3)"
            obs["observation.images.spec"] = spectrogram
        
        return obs
    
    def get_task_description(self):
        """
        Return the task description.
        """
        if self.task_type == "sound":
            return "Listen to the sound and pick up the cube making sound, then place it in the box."
        elif self.task_type == "soundDiff":
            return f"Listen to the sound. If sound A, place in right box. If sound B, place in left box."
        elif self.task_type == "soundShake":
            return "Shake the cubes. Pick up the one that makes sound and place it in the box."
        elif self.task_type == "soundAll":
            return f"Listen to find the cube making sound A, pick it up. When moved, if sound B plays, place in right box. If sound C, place in left box. "
        elif self.task_type == "soundSim":
            return f"Listen to find the speaker playing sound, pick it up, and place it in the correct box. Sound A goes to the right box, and sound B goes to the left box."
        return "Sound Task"
    
    def save_videos(self, file_name, fps=30):
        """
        Save videos (including sound and spec).
        """
        self.front_cam.stop_recording(save_to_filename=f"{file_name}_front.mp4", fps=fps)
        self.side_cam.stop_recording(save_to_filename=f"{file_name}_side.mp4", fps=fps)        

if __name__ == "__main__":
    import cv2
    import re
    task = "soundDiff-m3-f30-s2-p0"
    
    parts = task.split("-")
    task_type = parts[0]
    pattern = r"-m(\d+)-f(\d+)-s(\d+)-p(\d+)"
    match = re.search(pattern, task)
    if match:
        m, f, s, p = map(int, match.groups())
        # m: mic_array_num
        mic_array_num = m
        # f: update_freq (Hz) -> SoundConfig.update_freq (Interval frames)
        # Base FPS = 30
        update_freq = max(1, int(30 / f))
        # s: sound info
        use_spectrogram = False
        use_soundmap = True
        if s == 0:
            mic_array_num = 0 # no sound info
            use_soundmap = False
        elif s == 1:
            use_spectrogram = False
        elif s == 2:
            use_spectrogram = True
        elif s == 3:
            use_spectrogram = True
            use_soundmap = False
        # p: processing
        use_gaussian_filter = False
        use_temporal_smoothing = False
        use_feature = False
        if p == 1:
            use_gaussian_filter = True
        elif p == 2:
            use_temporal_smoothing = True
        elif p == 3:
            use_gaussian_filter = True
            use_temporal_smoothing = True
        elif p == 4:
            use_feature = True
        sound_config = SoundConfig(
            mic_array_num=mic_array_num,
            update_freq=update_freq,
            use_spectrogram=use_spectrogram,
            use_soundmap=use_soundmap,
            use_gaussian_filter=use_gaussian_filter,
            use_temporal_smoothing=use_temporal_smoothing,
            use_feature=use_feature,
            audio_file_path="sounds/1.wav" # default
        )
    else:
        raise ValueError(f"Invalid task format: {self.task}")

    env = SoundTask(
        observation_height=224,
        observation_width=224,
        show_viewer=False,
        sound_config=sound_config,
        task_type=task_type
    )
    # Reset and step
    obs, info = env.reset()
    for _ in range(10):
        action = np.random.uniform(-1.0, 1.0, size=(AGENT_DIM,))
        obs, reward, terminated, truncated, info = env.step(action)
    # Save observations
    for key, value in obs.items():
        if key == "observation.state":
            continue
        # RGB->BGR conversion (for OpenCV)
        if value.ndim == 3 and value.shape[2] == 3:
            value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
        print(f"{key}: {value.shape}")
        # Format the file name
        filename = key.replace("observation.images.", "")
        cv2.imwrite(f"images/test_{filename}.png", value)
    # Save videos
    env.save_videos("test_sound_task", fps=30)
    # Cleanup
    env.close()
