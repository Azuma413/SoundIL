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
        sound_config: SoundConfig = None
    ):
        # sound_configがNoneの場合はデフォルト値を使用
        if sound_config is None:
            sound_config = SoundConfig()
        
        sound_config.observation_height = observation_height
        sound_config.observation_width = observation_width
        self.sound_config = sound_config
        # 親クラスの初期化（_build_sceneが呼ばれる）
        super().__init__(
            observation_height=observation_height,
            observation_width=observation_width,
            show_viewer=show_viewer,
            device=device,
            same_color=sound_config.same_color,
        )
    
    def _build_scene(self, show_viewer):
        """
        NormalTaskの_build_sceneをオーバーライドして、
        SoundCameraを追加する
        """
        super()._build_scene(show_viewer)
        # 初期ターゲットはcubeR（reset時に変更される）
        self.sound_cam = SoundCamera(
            target=self.cubeR,
            config=self.sound_config
        )
    
    def _make_obs_space(self):
        """
        観測空間を拡張して、sound0、sound1、spec（use_spectrogramがTrueの場合）を追加
        """
        obs_space_dict = {
            "observation.state": spaces.Box(low=-np.inf, high=np.inf, shape=(AGENT_DIM,), dtype=np.float32),
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
            "observation.images.sound0": spaces.Box(
                low=0, high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8
            ),
            "observation.images.sound1": spaces.Box(
                low=0, high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8
            ),
        }
        
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
        # self.colorに応じてSoundCameraのターゲットを更新
        if self.color == "red":
            self.sound_cam.target = self.cubeR
        elif self.color == "blue":
            self.sound_cam.target = self.cubeB
        elif self.color == "green":
            self.sound_cam.target = self.cubeG
        return obs, info
    
    def get_obs(self):
        """
        観測を取得（sound0、sound1、specを含む）
        """
        obs = super().get_obs()
        # SoundCameraからsound mapとスペクトログラムを取得
        sound_map0, sound_map1, spectrogram = self.sound_cam.render()
        
        # sound0とsound1を格納（両方とも3チャンネル）
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
        return f"Listen to the sound and pick up the {self.color} cube making sound, then place it in the box."
    
    def save_videos(self, file_name, fps=30):
        """
        動画を保存（soundとspecを含む）
        """
        self.front_cam.stop_recording(save_to_filename=f"{file_name}_front.mp4", fps=fps)
        self.side_cam.stop_recording(save_to_filename=f"{file_name}_side.mp4", fps=fps)        

if __name__ == "__main__":
    import cv2
    
    # SoundConfigの設定
    sound_config = SoundConfig(
        observation_height=480,
        observation_width=640,
        mic_array_num=3,
        mics_per_array=8,
        use_spectrogram=True,
        audio_file_path="sounds/1.wav"
    )
    
    # タスクの初期化
    gs.init(backend=gs.gpu, precision="32")
    task = SoundTask(
        observation_height=480,
        observation_width=640,
        show_viewer=False,
        sound_config=sound_config
    )
    
    # リセットとステップ
    obs, info = task.reset()
    
    for _ in range(10):
        action = np.random.uniform(-1.0, 1.0, size=(AGENT_DIM,))
        obs, reward, terminated, truncated, info = task.step(action)
    
    # 観測の保存
    for key, value in obs.items():
        if key == "agent_pos":
            continue
        
        # RGB→BGR変換（OpenCV用）
        if value.ndim == 3 and value.shape[2] == 3:
            value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
        
        print(f"{key}: {value.shape}")
        
        # ファイル名を整形
        filename = key.replace("observation.images.", "")
        cv2.imwrite(f"images/test_{filename}.png", value)
    
    # 動画の保存
    task.save_videos("test_sound_task", fps=30)
    
    # クリーンアップ
    task.close()
