import genesis as gs
import numpy as np
from gymnasium import spaces
from env.tasks.normal import NormalTask, AGENT_DIM
from env.tasks.sound_camera import SoundCamera, SoundConfig

class SoundTask(NormalTask):
    """
    NormalTaskをラップして、SoundCameraによる音響観測を追加したタスク環境
    """
    
    def __init__(
        self,
        observation_height,
        observation_width,
        show_viewer=False,
        device="cuda",
        sound_config: SoundConfig = None,
        task_type="sound" # "sound", "soundDiff", "soundShake", "soundAll"
    ):
        self.task_type = task_type
        # sound_configがNoneの場合はデフォルト値を使用
        if sound_config is None:
            sound_config = SoundConfig()
        
        sound_config.observation_height = observation_height
        sound_config.observation_width = observation_width
        self.sound_config = sound_config
        
        # タスクタイプに応じた設定
        num_cubes = 2 # default for sound and soundShake
        use_two_boxes = False
        same_color = True # default for sound and soundShake
        
        if task_type == "sound":
            num_cubes = 2
            same_color = True
        elif task_type == "soundDiff":
            num_cubes = 1
            use_two_boxes = True
            same_color = False # 色は関係ないが、1つなので何でも良い
        elif task_type == "soundShake":
            num_cubes = 2
            same_color = True
            self.sound_config.shake_mode = True
        elif task_type == "soundAll":
            num_cubes = 2
            use_two_boxes = True
            same_color = True
            self.sound_config.sound_all_mode = True
            self.sound_config.audio_file_path = None # 静止時: 音A
            
        self.target_cube_name = None
        self.current_sound_type = "Unknown"
        self.target_box = None

        # 親クラスの初期化（_build_sceneが呼ばれる）
        super().__init__(
            observation_height=observation_height,
            observation_width=observation_width,
            show_viewer=show_viewer,
            device=device,
            same_color=same_color,
            num_cubes=num_cubes,
            use_two_boxes=use_two_boxes
        )
    
    def _build_scene(self, show_viewer):
        """
        NormalTaskの_build_sceneをオーバーライドして、
        SoundCameraを追加する
        """
        super()._build_scene(show_viewer)
        # 初期ターゲットはcubeR（reset時に変更される）
        # soundDiffの場合はcubeRのみ
        self.sound_cam = SoundCamera(
            target=self.cubeR,
            config=self.sound_config
        )
    
    def _make_obs_space(self):
        """
        観測空間を拡張して、sound0、sound1、spec（use_spectrogramがTrueの場合）を追加
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
        
        # use_spectrogramがTrueの場合、specを追加（3チャンネル）
        if self.sound_config.use_spectrogram:
            obs_space_dict["observation.images.spec"] = spaces.Box(
                low=0, high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8
            )
        
        return spaces.Dict(obs_space_dict)
    
    def reset(self):
        """
        環境をリセットし、音源ターゲットを更新
        """
        # 親クラスのresetを呼び出し（self.colorが決定される）
        obs, info = super().reset()
        
        if self.task_type == "sound":
            # 2つのCubeのうち、どちらかが音源になる
            # same_color=Trueなので、見た目は同じ
            # colorはred, green, blueのいずれかがランダムに選ばれているが、
            # num_cubes=2なので、cubeRとcubeGが存在する
            # self.colorを使ってターゲットを決めるロジックは親クラスのresetでランダムに選ばれたcolorに依存する
            # しかし、NormalTaskのresetではcolorはランダムに選ばれるが、same_color=Trueの場合、見た目は同じ
            # ここでは、ランダムにターゲットを選ぶ
            target_name = np.random.choice(["cubeR", "cubeG"])
            if target_name == "cubeR":
                self.sound_cam.target = self.cubeR
                self.target_cube_name = "cubeR"
            else:
                self.sound_cam.target = self.cubeG
                self.target_cube_name = "cubeG"
                
        elif self.task_type == "soundDiff":
            # 音源はcubeR（1つしかない）
            self.sound_cam.target = self.cubeR
            self.target_cube_name = "cubeR"
            
            # 音の種類をランダムに選択（A or B）
            sound_type = np.random.choice(["A", "B"])
            if sound_type == "A":
                # 音Aの設定
                self.sound_cam._load_audio_file("sounds/1.wav")
                self.target_box = self.box_right # 音Aなら右の箱
            else:
                # 音Bの設定
                self.sound_cam._load_audio_file("sounds/3.wav")
                self.target_box = self.box_left # 音Bなら左の箱
            
            # タスク記述更新用
            self.current_sound_type = sound_type

        elif self.task_type == "soundShake":
            # 2つのCubeのうち、どちらかがターゲット
            target_name = np.random.choice(["cubeR", "cubeG"])
            if target_name == "cubeR":
                self.sound_cam.target = self.cubeR
                self.target_cube_name = "cubeR"
            else:
                self.sound_cam.target = self.cubeG
                self.target_cube_name = "cubeG"

        elif self.task_type == "soundAll":
            # 2つのCubeのうち、どちらかが音源（音A: 0.wav）
            target_name = np.random.choice(["cubeR", "cubeG"])
            if target_name == "cubeR":
                self.sound_cam.target = self.cubeR
                self.target_cube_name = "cubeR"
            else:
                self.sound_cam.target = self.cubeG
                self.target_cube_name = "cubeG"
            
            # 移動時の音をランダムに選択（音B or 音C）
            sound_type = np.random.choice(["B", "C"])
            if sound_type == "B":
                # 音B (1.wav) → 右の箱
                self.sound_cam.set_moving_audio("sounds/1.wav")
                self.target_box = self.box_right
            else:
                # 音C (2.wav) → 左の箱
                self.sound_cam.set_moving_audio("sounds/2.wav")
                self.target_box = self.box_left
            
            self.current_sound_type = sound_type
        
        return obs, info
    
    def compute_reward(self, target=None):
        if self.task_type == "soundDiff":
            # 指定された箱に入っているかチェック
            return super().compute_reward(target="cubeR", target_box=self.target_box)
        elif self.task_type == "soundAll":
            # ターゲットキューブが指定された箱に入っているかチェック
            return super().compute_reward(target=self.target_cube_name, target_box=self.target_box)
        else:
            # sound, soundShakeの場合は、ターゲットのCubeが（任意の）箱に入っているかチェック
            # NormalTask.compute_rewardはtargetを指定するとそのCubeが箱に入っているかチェックする
            # target_box=Noneならどちらの箱でもOK（NormalTaskの修正による）
            actual_target = target if target is not None else self.target_cube_name
            return super().compute_reward(target=actual_target)
    
    def get_obs(self):
        """
        観測を取得（sound0、sound1、specを含む）
        """
        obs = super().get_obs()
        # SoundCameraからsound mapとスペクトログラムを取得
        sound_map0, sound_map1, spectrogram = self.sound_cam.render()
        
        # sound0とsound1を格納（両方とも3チャンネル）
        if self.sound_config.use_soundmap:
            assert sound_map0.ndim == 3 and sound_map0.shape[2] == 3, \
                f"sound_map0 shape {sound_map0.shape} is not (H, W, 3)"
            assert sound_map1.ndim == 3 and sound_map1.shape[2] == 3, \
                f"sound_map1 shape {sound_map1.shape} is not (H, W, 3)"
            
            obs["observation.images.sound0"] = sound_map0
            obs["observation.images.sound1"] = sound_map1
        
        # use_spectrogramがTrueの場合、specを追加（3チャンネル）
        if self.sound_config.use_spectrogram:
            assert spectrogram is not None, "Spectrogram is None despite use_spectrogram=True"
            assert spectrogram.ndim == 3 and spectrogram.shape[2] == 3, \
                f"spectrogram shape {spectrogram.shape} is not (H, W, 3)"
            obs["observation.images.spec"] = spectrogram
        
        return obs
    
    def get_task_description(self):
        """
        タスクの説明を返す
        """
        if self.task_type == "sound":
            return "Listen to the sound and pick up the cube making sound, then place it in the box."
        elif self.task_type == "soundDiff":
            return f"Listen to the sound ({self.current_sound_type}). If sound A, place in right box. If sound B, place in left box."
        elif self.task_type == "soundShake":
            return "Shake the cubes. Pick up the one that makes sound and place it in the box."
        elif self.task_type == "soundAll":
            return f"Listen to find the cube making sound A, pick it up. When moved, if sound B plays, place in right box. If sound C, place in left box. (Current: {self.current_sound_type})"
        return "Sound Task"
    
    def save_videos(self, file_name, fps=30):
        """
        動画を保存（soundとspecを含む）
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
            mic_array_num = 0 # 音情報なし
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
            audio_file_path="sounds/1.wav" # デフォルト
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
    # リセットとステップ
    obs, info = env.reset()
    for _ in range(10):
        action = np.random.uniform(-1.0, 1.0, size=(AGENT_DIM,))
        obs, reward, terminated, truncated, info = env.step(action)
    # 観測の保存
    for key, value in obs.items():
        if key == "observation.state":
            continue
        # RGB→BGR変換（OpenCV用）
        if value.ndim == 3 and value.shape[2] == 3:
            value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
        print(f"{key}: {value.shape}")
        # ファイル名を整形
        filename = key.replace("observation.images.", "")
        cv2.imwrite(f"images/test_{filename}.png", value)
    # 動画の保存
    env.save_videos("test_sound_task", fps=30)
    # クリーンアップ
    env.close()
