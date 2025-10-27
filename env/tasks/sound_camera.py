import numpy as np
import torch
import pyroomacoustics as pra
import cv2
from dataclasses import dataclass

@dataclass
class CameraConfig:
    observation_height: int = 128
    observation_width: int = 128
    mic_array_num: int = 3
    use_spectrogram: bool = False
    use_gaussian_filter: bool = False
    use_temporal_smoothing: bool = False
    use_feature: bool = False

class SoundCamera:
    def __init__(self, target, config: CameraConfig):
        self.target = target
        self.config = config
        self.frames = []
        self.fs = 16000
        self.nfft = 256
        self.freq_range = [300, 3500]
        self.mic_pos = [
            [0.8, 0.0, 0.1],
            [0.2, -0.3, 0.1],
            [0.2, 0.3, 0.1],
        ]
        self.corners = np.array([
            [-0.5, 1.0],
            [1.5, 1.0],
            [1.5, -1.0],
            [-0.5, -1.0],
        ]).T

    def start_recording(self):
        self.frames = []

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
                doa = pra.doa.algorithms['MUSIC'](aroom.mic_array.R, self.fs, self.nfft, c=343., num_src=1, max_four=4)
                doa.locate_sources(X, freq_range=self.freq_range)
                spatial_resp = doa.grid.values * 25
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
        sound_image_array = np.flip(sound_image_array, axis=2)
        sound_image_array = np.clip(sound_image_array, 0, 255).astype(np.uint8)
        sound_image_array = np.transpose(sound_image_array, (1, 2, 0))
        self.frames.append(sound_image_array)
        return sound_image_array, None

class MarkerSoundCamera(SoundCamera):
    def render(self):
        sound_image_array, _ = super().render()
        marked_image = np.zeros_like(sound_image_array)
        marked_image[:, :, 0] = np.mean(sound_image_array, axis=2)
        max_coords = np.unravel_index(np.argmax(marked_image[:, :, 0]), marked_image.shape[:2])
        row = int(np.mean(max_coords[0]))
        col = int(np.mean(max_coords[1]))
        size = 5
        for r in range(max(0, row - size), min(marked_image.shape[0], row + size + 1)):
            for c in range(max(0, col - size), min(marked_image.shape[1], col + size + 1)):
                marked_image[r, c, 1] = 255
        threshold = 100
        marked_image[:, :, 2] = np.where(marked_image[:, :, 0] > threshold, 255, 0)
        self.frames[-1] = marked_image  # 最後のフレームを更新
        return marked_image, None

class WeightedSoundCamera(SoundCamera):
    def __init__(self, target, observation_height, observation_width, weight=0.5):
        super().__init__(target, observation_height, observation_width)
        self.weight = weight  # 重みの初期化
        self.past_frame = np.ones((self.observation_height, self.observation_width, 3), dtype=np.float32)*255/2

    def render(self):
        sound_image_array, _ = super().render()
        if self.past_frame is None:
            return sound_image_array, None
        normalized_array = np.zeros_like(sound_image_array, dtype=np.float32)
        for i in range(3):  # RGB各チャンネルに対して処理
            channel = sound_image_array[:, :, i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val > min_val:
                normalized_array[:, :, i] = (channel - min_val) / (max_val - min_val)
            else:
                normalized_array[:, :, i] = 0.5  # または任意のデフォルト値
        self.past_frame *= normalized_array + self.weight
        self.past_frame = np.clip(self.past_frame, 30, 255)
        weighted_frame = self.past_frame.astype(np.uint8)
        self.frames[-1] = weighted_frame
        return weighted_frame, None