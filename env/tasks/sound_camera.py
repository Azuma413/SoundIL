import numpy as np
import torch
import pyroomacoustics as pra
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from scipy.signal import stft
from numpy.fft import rfft, irfft, rfftfreq
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import NMF
from pydub import AudioSegment


@dataclass
class SoundConfig:
    """音響シミュレーションとSoundMap生成の設定"""
    # 画像サイズ
    observation_height: int = 128
    observation_width: int = 128
    
    # マイクロフォンアレイ関連
    mic_array_num: int = 3  # マイクロフォンアレイの数
    mics_per_array: int = 8  # 各アレイのマイク数
    mic_radius: float = 0.035  # アレイの半径（メートル）
    
    # 音響シミュレーション関連
    fs: int = 16000  # サンプリング周波数
    nfft: int = 512  # FFT長
    freq_range: List[int] = field(default_factory=lambda: [300, 3500])  # 周波数範囲
    room_max_order: int = 3  # 反射の最大次数
    sound_speed: float = 343.0  # 音速（m/s）
    
    # MUSIC法関連
    music_num_src: int = 3  # 推定する音源数
    spatial_resp_scale: float = 25.0  # 空間応答のスケーリング係数
    
    # ピーク検出関連（ビームフォーミング用）
    num_peaks: int = 2  # 検出するピーク数
    gaussian_sigma: float = 1.0  # ガウシアンフィルタの強度
    peak_distance: int = 10  # ピーク間の最小距離
    
    # ビームフォーミング関連
    beamform_normalize: bool = True  # ビームフォーミング出力の正規化
    
    # スペクトログラム関連
    use_spectrogram: bool = False  # スペクトログラムを返すか
    nmf_components: int = 16  # NMFの成分数
    nmf_threshold: float = 0.01  # NMFマスクの閾値
    
    # 画像処理オプション
    use_gaussian_filter: bool = False  # SoundMapにガウシアンフィルタ
    use_temporal_smoothing: bool = False  # 時間的平滑化
    temporal_smoothing_weight: float = 0.5  # 時間的平滑化の重み
    use_feature: bool = False  # 特徴画像を生成
    
    # 閾値
    feature_threshold: float = 0.9  # 2値化の閾値
    marker_size: int = 5  # マーカーのサイズ
    
    # 部屋の設定
    room_corners: np.ndarray = field(default_factory=lambda: np.array([
        [-0.5, 1.0],
        [1.5, 1.0],
        [1.5, -1.0],
        [-0.5, -1.0],
    ]).T)
    room_height: float = 3.0  # 部屋の高さ
    
    # マイク位置（自動生成されない場合に使用）
    mic_positions: Optional[List[List[float]]] = None
    
    # 音源ファイル関連
    audio_file_path: Optional[str] = None  # 音源ファイルのパス（Noneの場合はホワイトノイズ）
    noise_intensity: float = 0.0  # ノイズ強度（マイク信号に加算するノイズの強度）


class SoundCamera:
    """音響シミュレーションとSoundMap生成を行うカメラクラス"""
    
    def __init__(self, target, config: SoundConfig):
        self.target = target
        self.config = config
        self.frames = []
        
        # マイクロフォン位置の初期化
        if config.mic_positions is not None and len(config.mic_positions) == config.mic_array_num:
            self.mic_positions = config.mic_positions
        else:
            self.mic_positions = self._generate_default_mic_positions()
        
        # 音声ファイルの読み込み
        self.audio_signal = None
        if config.audio_file_path is not None:
            self._load_audio_file(config.audio_file_path)
        
        # 時間的平滑化用の過去フレーム
        self.past_frame = None
        if config.use_temporal_smoothing:
            num_channels = 3 if config.use_feature else config.mic_array_num
            self.past_frame = np.ones(
                (config.observation_height, config.observation_width, num_channels),
                dtype=np.float32
            ) * 127.5
    
    def _load_audio_file(self, audio_file_path: str):
        """音声ファイルを読み込む（WAV、MP3など）"""
        format = audio_file_path.split(".")[-1]
        sound = AudioSegment.from_file(audio_file_path, format=format)
        signal = np.array(sound.get_array_of_samples())
        self.audio_signal = signal
    
    def _generate_default_mic_positions(self) -> List[List[float]]:
        """デフォルトのマイクロフォン位置を生成（10x10x3の部屋用）"""
        # test.pyと同じように円形配置
        positions = []
        for i in range(self.config.mic_array_num):
            theta = np.pi * (4*i - self.config.mic_array_num + 2) / (2 * self.config.mic_array_num)
            x = 5 + 2 * np.cos(theta)
            y = 5 + 2 * np.sin(theta)
            positions.append([x, y, 0.3])
        
        return positions
    
    def start_recording(self):
        """録画を開始"""
        self.frames = []
    
    def stop_recording(self, save_to_filename: str, fps: int = 30):
        """録画を停止し、ビデオファイルに保存"""
        if not self.frames:
            print("Warning: No frames to save.")
            return
        
        sound_image = np.array(self.frames)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            save_to_filename,
            fourcc,
            fps,
            (self.config.observation_width, self.config.observation_height)
        )
        
        for i in range(sound_image.shape[0]):
            frame_to_write = sound_image[i]
            if frame_to_write.dtype != np.uint8:
                frame_to_write = frame_to_write.astype(np.uint8)
            out.write(frame_to_write)
        
        out.release()
        self.frames = []
    
    def render(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        SoundMapとスペクトログラムを生成
        
        Returns:
            sound_map: (height, width, channels) の画像 (uint8)
            spectrogram: use_spectrogram=Trueの場合のスペクトログラム、それ以外はNone
        """
        # 音源位置の取得
        sound_pos = self.target.get_pos() if self.target is not None else torch.tensor([0.5, 0.3, 0.1])
        if isinstance(sound_pos, torch.Tensor):
            sound_pos = sound_pos.cpu().numpy()
        
        mic_signals_list, music_results = self._simulate_all_arrays(sound_pos)
        
        # 各マイクアレイのMUSIC結果からSoundMapを生成
        sound_maps = []
        for i in range(self.config.mic_array_num):
            sound_map = self._generate_soundmap_from_doa(
                music_results[i],
                self.mic_positions[i]
            )
            sound_maps.append(sound_map)
        
        # SoundMapの結合と処理
        if not sound_maps:
            print("Warning: All sound simulations failed. Returning zero array.")
            num_channels = 3 if self.config.use_feature else self.config.mic_array_num
            sound_map_image = np.zeros(
                (self.config.observation_height, self.config.observation_width, num_channels),
                dtype=np.uint8
            )
            self.frames.append(sound_map_image)
            return sound_map_image, None
        
        # shape: (mic_array_num, height, width) -> (height, width, mic_array_num)
        sound_map_array = np.array(sound_maps).transpose(1, 2, 0)
        
        # ガウシアンフィルタの適用
        if self.config.use_gaussian_filter:
            for i in range(sound_map_array.shape[2]):
                sound_map_array[:, :, i] = gaussian_filter(
                    sound_map_array[:, :, i],
                    sigma=self.config.gaussian_sigma
                )
        
        # 特徴画像への変換
        if self.config.use_feature:
            sound_map_array = self._convert_to_feature_image(sound_map_array)
        
        # 0-255にスケーリング
        sound_map_array = self._normalize_to_uint8(sound_map_array)
        
        # 時間的平滑化
        if self.config.use_temporal_smoothing:
            sound_map_array = self._apply_temporal_smoothing(sound_map_array)
        
        # フレームに追加
        self.frames.append(sound_map_array)
        
        # スペクトログラム生成
        spectrogram = None
        if self.config.use_spectrogram:
            spectrogram = self._generate_spectrogram(
                sound_maps,
                mic_signals_list,
                music_results
            )
        
        return sound_map_array, spectrogram
    
    def _simulate_all_arrays(
        self,
        sound_pos: np.ndarray
    ) -> Tuple[List[np.ndarray], List[pra.doa.MUSIC]]:
        """
        test.pyと同じ方法で、すべてのマイクアレイを1つの部屋でシミュレーション
        
        Returns:
            mic_signals_list: 各マイクアレイの信号リスト
            music_results: 各マイクアレイのMUSIC結果リスト
        """
        # 部屋の作成（test.pyと同じShoeBox、問題4の修正：config.room_heightを使用）
        room_dim = [10, 10, self.config.room_height]
        room = pra.ShoeBox(room_dim, fs=self.config.fs, max_order=self.config.room_max_order)
        
        # 音源の追加（問題2の修正：test.pyと同じ方法）
        room.add_source(sound_pos)
        
        # すべてのマイクアレイを追加（問題1の修正）
        for i in range(self.config.mic_array_num):
            mic_array_positions = self._generate_circular_array(
                self.mic_positions[i],
                self.config.mics_per_array,
                self.config.mic_radius
            )
            room.add_microphone_array(mic_array_positions)
        
        # RIRの計算（問題2の修正：test.pyと同じ方法）
        room.compute_rir()
        
        # 音源信号の生成と設定（問題2の修正：test.pyと同じ方法）
        if self.audio_signal is not None:
            signal = self.audio_signal.copy()
            if len(signal) < self.config.fs:
                num_repeats = int(np.ceil(self.config.fs / len(signal)))
                signal = np.tile(signal, num_repeats)[:self.config.fs]
        else:
            signal = np.random.randn(self.config.fs)
        
        room.sources[0].signal = signal
        
        # シミュレーション実行
        room.simulate()
        
        # 各マイクアレイの信号を分離
        mic_signals_list = []
        for i in range(self.config.mic_array_num):
            start_idx = i * self.config.mics_per_array
            end_idx = (i + 1) * self.config.mics_per_array
            signals = room.mic_array.signals[start_idx:end_idx]
            
            # ノイズの加算
            if self.config.noise_intensity > 0:
                noise = np.random.randn(*signals.shape) * np.mean(signals) * self.config.noise_intensity
                signals = signals*(1.0 - self.config.noise_intensity) + noise
            
            mic_signals_list.append(signals)
        
        # 各マイクアレイでMUSIC法を実行
        music_results = []
        for i in range(self.config.mic_array_num):
            # STFT
            stft_results = []
            for j in range(mic_signals_list[i].shape[0]):
                f, t, Zxx = stft(
                    mic_signals_list[i][j, :],
                    fs=self.config.fs,
                    nperseg=self.config.nfft,
                    noverlap=self.config.nfft // 2
                )
                stft_results.append(Zxx)
            Z = np.array(stft_results)
            
            # マイクアレイ位置の生成
            mic_array_positions = self._generate_circular_array(
                self.mic_positions[i],
                self.config.mics_per_array,
                self.config.mic_radius
            )
            
            # MUSIC法
            doa = pra.doa.MUSIC(
                mic_array_positions,
                fs=self.config.fs,
                nfft=self.config.nfft,
                c=self.config.sound_speed,
                num_src=self.config.music_num_src
            )
            doa.locate_sources(Z)
            
            music_results.append(doa)
        
        return mic_signals_list, music_results
    
    def _generate_circular_array(
        self,
        center: List[float],
        num_mics: int,
        radius: float
    ) -> np.ndarray:
        """円形マイクアレイの位置を生成"""
        angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        z = np.ones(num_mics) * center[2]
        return np.array([x, y, z])
    
    def _generate_soundmap_from_doa(
        self,
        doa: pra.doa.MUSIC,
        mic_center: List[float]
    ) -> np.ndarray:
        """DOA推定結果からSoundMapを生成（test.pyのmake_2dmap方式）"""
        # MUSIC法の空間スペクトル（平均）- test.pyと同じくPsslを使用
        spec = np.log10(np.mean(doa.Pssl, axis=1))
        spec /= np.sum(spec)
        
        # 部屋のサイズとマップの解像度
        room_size = 10.0  # 部屋のサイズ（10x10）
        map_scale = self.config.observation_height / room_size
        
        # マイク中心位置
        cx = mic_center[0]
        cy = mic_center[1]
        
        # 2Dマップを生成
        sound_map = np.zeros((self.config.observation_height, self.config.observation_width))
        
        for i in range(self.config.observation_height):
            for j in range(self.config.observation_width):
                # ピクセル座標を実空間座標に変換
                x = i / map_scale
                y = j / map_scale
                
                # マイク中心からの角度と距離
                theta = np.arctan2((y - cy), (x - cx))
                d = np.sqrt((y - cy)**2 + (x - cx)**2)
                
                # 角度インデックスを計算
                angle_idx = int(theta / (2 * np.pi / len(spec))) % len(spec)
                
                # 距離による減衰を考慮した値を設定
                sound_map[i, j] = spec[angle_idx] / (d + 10)
        
        return sound_map
    
    def _convert_to_feature_image(self, sound_map: np.ndarray) -> np.ndarray:
        """
        SoundMapを特徴画像に変換
        
        Args:
            sound_map: (height, width, mic_array_num)
        
        Returns:
            feature_image: (height, width, 3)
        """
        # チャンネル0: 平均
        mean_map = np.mean(sound_map, axis=2)
        
        # チャンネル1: 最大値位置のマーカー
        marker_map = np.zeros_like(mean_map)
        max_coords = np.unravel_index(np.argmax(mean_map), mean_map.shape)
        row, col = int(max_coords[0]), int(max_coords[1])
        size = self.config.marker_size
        
        row_min = max(0, row - size)
        row_max = min(mean_map.shape[0], row + size + 1)
        col_min = max(0, col - size)
        col_max = min(mean_map.shape[1], col + size + 1)
        marker_map[row_min:row_max, col_min:col_max] = 255
        
        # チャンネル2: 2値画像
        binary_map = np.where(mean_map > np.max(mean_map) * self.config.feature_threshold, 255, 0)
        
        # 3チャンネルに結合
        feature_image = np.stack([mean_map, marker_map, binary_map], axis=2)
        
        return feature_image
    
    def _normalize_to_uint8(self, array: np.ndarray) -> np.ndarray:
        """配列を0-255のuint8にスケーリング"""
        array_normalized = np.zeros_like(array, dtype=np.float32)
        
        for i in range(array.shape[2]):
            channel = array[:, :, i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            
            if max_val > min_val:
                array_normalized[:, :, i] = (channel - min_val) / (max_val - min_val) * 255
            else:
                array_normalized[:, :, i] = 127.5
        
        return np.clip(array_normalized, 0, 255).astype(np.uint8)
    
    def _apply_temporal_smoothing(self, current_frame: np.ndarray) -> np.ndarray:
        """時間的平滑化を適用"""
        if self.past_frame is None:
            self.past_frame = current_frame.astype(np.float32)
            return current_frame
        
        # 現在のフレームを正規化
        current_normalized = current_frame.astype(np.float32) / 255.0
        
        # 重み付き平均
        weight = self.config.temporal_smoothing_weight
        self.past_frame = self.past_frame * (1 - weight) + current_normalized * weight * 255
        self.past_frame = np.clip(self.past_frame, 0, 255)
        
        return self.past_frame.astype(np.uint8)
    
    def _generate_spectrogram(
        self,
        sound_maps: List[np.ndarray],
        mic_signals_list: List[Optional[np.ndarray]],
        music_results: List[Optional[pra.doa.MUSIC]]
    ) -> Optional[np.ndarray]:
        """ビームフォーミングによるスペクトログラムを生成し、画像化して返す"""
        # SoundMapからピーク検出
        combined_map = np.mean(sound_maps, axis=0)
        
        # ガウシアンフィルタでノイズ除去
        smoothed_map = gaussian_filter(combined_map, self.config.gaussian_sigma)
        
        # ピーク検出
        top_peaks = self._find_top_k_peaks(smoothed_map, self.config.num_peaks)
        
        if not top_peaks:
            return None
        
        # 各マイクアレイでビームフォーミング
        beamform_specs = []
        
        for i, (mic_signals, mic_pos) in enumerate(zip(mic_signals_list, self.mic_positions)):
            if mic_signals is None:
                continue
            
            # マイク位置の生成
            mic_array_pos = self._generate_circular_array(
                mic_pos,
                self.config.mics_per_array,
                self.config.mic_radius
            ).T
            
            # 中心からの相対位置
            center = np.mean(mic_array_pos, axis=0)
            mic_array_pos_rel = mic_array_pos - center
            
            # 各ピークに対してビームフォーミング
            for peak_x, peak_y in top_peaks:
                # ピーク位置（実空間座標）からマイクへの方位角を計算
                theta_deg = self._pixel_to_azimuth(peak_x, peak_y, mic_pos)
                
                # DSビームフォーミング
                beamformed_signal = self._ds_beamform(
                    mic_signals,
                    mic_array_pos_rel,
                    theta_deg
                )
                
                # スペクトログラム計算
                f, t, Zxx = stft(beamformed_signal, fs=self.config.fs)
                power_spec = np.abs(Zxx) ** 2
                beamform_specs.append(power_spec)
        
        if not beamform_specs:
            return None
        
        # 平均スペクトログラム
        mean_spec = np.mean(beamform_specs, axis=0)
        
        # test.pyと同様の処理：対数スケールに変換
        spec_db = 10 * np.log10(mean_spec + 1e-10)
        
        # 画像化：observation_height x observation_widthにリサイズ
        spectrogram_image = self._convert_spectrogram_to_image(spec_db)
        
        return spectrogram_image
    
    def _convert_spectrogram_to_image(self, spec_db: np.ndarray) -> np.ndarray:
        """
        スペクトログラムを画像化（observation_height x observation_widthにリサイズ）
        
        Args:
            spec_db: 対数スケールのスペクトログラム (周波数ビン数, 時間フレーム数)
        
        Returns:
            spectrogram_image: (observation_height, observation_width) の画像 (uint8)
        """
        # 正規化（0-255の範囲に）
        min_val = np.min(spec_db)
        max_val = np.max(spec_db)
        
        if max_val > min_val:
            normalized = (spec_db - min_val) / (max_val - min_val) * 255
        else:
            normalized = np.ones_like(spec_db) * 127.5
        
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # observation_height x observation_widthにリサイズ
        resized = cv2.resize(
            normalized,
            (self.config.observation_width, self.config.observation_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        return resized
    
    def _find_top_k_peaks(
        self,
        data: np.ndarray,
        k: int
    ) -> List[Tuple[float, float]]:
        """2Dデータから上位k個のピークを検出（実空間座標で返す）"""
        # ローカル最大値を検出
        is_peak = np.ones_like(data, dtype=bool)
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                shifted_data = np.roll(data, shift=(dr, dc), axis=(0, 1))
                comparison = data >= shifted_data
                
                # 境界処理
                if dr != 0:
                    if dr == 1:
                        comparison[:dr, :] = True
                    else:
                        comparison[dr:, :] = True
                if dc != 0:
                    if dc == 1:
                        comparison[:, :dc] = True
                    else:
                        comparison[:, dc:] = True
                
                is_peak &= comparison
        
        # ピーク点の抽出
        peak_rows, peak_cols = np.where(is_peak)
        peak_values = data[peak_rows, peak_cols]
        
        # 上位k個を選択
        sorted_indices = np.argsort(peak_values)[::-1]
        top_k_indices = sorted_indices[:min(k, len(sorted_indices))]
        
        # ピクセル座標を実空間座標に変換（test.pyと同じ）
        room_size = 10.0
        map_scale = self.config.observation_height / room_size
        
        return [(peak_rows[i] / map_scale, peak_cols[i] / map_scale) for i in top_k_indices]
    
    def _pixel_to_azimuth(
        self,
        x: float,
        y: float,
        mic_center: List[float]
    ) -> float:
        """実空間座標から方位角（度）を計算（test.pyに合わせた方式）"""
        # マイク位置からの相対座標
        dx = x - mic_center[0]
        dy = y - mic_center[1]
        
        # 角度計算（ラジアンから度に変換）
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.rad2deg(angle_rad)
        
        return angle_deg
    
    def _ds_beamform(
        self,
        mic_signals: np.ndarray,
        mic_positions: np.ndarray,
        azimuth_deg: float,
        elevation_deg: float = 0.0
    ) -> np.ndarray:
        """DSビームフォーミング"""
        # 転置して (N, M) の形式に
        if mic_signals.shape[0] < mic_signals.shape[1]:
            y_multi = mic_signals.T
        else:
            y_multi = mic_signals
        
        N, M = y_multi.shape
        
        # FFT
        Y = rfft(y_multi, axis=0)
        freqs = rfftfreq(N, d=1.0/self.config.fs)
        
        # ステアリングベクトル
        u = self._unit_vec_from_angles(azimuth_deg, elevation_deg)
        A = self._steering_vector(freqs, mic_positions, u)
        
        # ビームフォーミング
        Y_beam = np.sum(A * Y, axis=1)
        
        if self.config.beamform_normalize:
            Y_beam /= M
        
        # 逆FFT
        y_beam = irfft(Y_beam, n=N)
        
        return y_beam
    
    def _unit_vec_from_angles(
        self,
        azimuth_deg: float,
        elevation_deg: float
    ) -> np.ndarray:
        """方位角と仰角から単位ベクトルを生成"""
        az = np.deg2rad(azimuth_deg)
        el = np.deg2rad(elevation_deg)
        
        u = np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el)
        ], dtype=float)
        
        return u / np.linalg.norm(u)
    
    def _steering_vector(
        self,
        freqs: np.ndarray,
        mic_positions: np.ndarray,
        doa_unitvec: np.ndarray
    ) -> np.ndarray:
        """ステアリングベクトルを計算"""
        # 各マイクの幾何遅延
        taus = -(mic_positions @ doa_unitvec) / self.config.sound_speed
        
        # 位相項
        phase = -2j * np.pi * freqs[:, None] * taus[None, :]
        
        return np.exp(phase)
