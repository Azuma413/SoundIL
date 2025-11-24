import numpy as np
import torch
import pyroomacoustics as pra
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from scipy.signal import stft, istft 
from numpy.fft import rfft, irfft, rfftfreq
from scipy.ndimage import gaussian_filter
from pydub import AudioSegment
from sklearn.decomposition import NMF

@dataclass
class SoundConfig:
    """音響シミュレーションとSoundMap生成の設定"""
    # 画像サイズ
    observation_height: int = 128
    observation_width: int = 128
    # マイクロフォンアレイ関連
    mic_array_num: int = 6  # マイクロフォンアレイの数
    mic_array_radius: float = 0.25 # 円形に配置する場合のアレイの半径（メートル）
    mics_per_array: int = 8  # 各アレイのマイク数
    mic_radius: float = 0.035  # マイクロフォンアレイにおけるマイクの配列の半径（メートル）
    # 音響シミュレーション関連
    fs: int = 16000  # サンプリング周波数
    nfft: int = 512  # FFT長
    room_max_order: int = 3  # 反射の最大次数
    sound_speed: float = 343.0  # 音速（m/s）
    room_height: float = 3.0  # 部屋の高さ
    # MUSIC法関連
    music_num_src: int = 3  # 推定する音源数
    # ピーク検出関連（ビームフォーミング用）
    num_peaks: int = 1  # 検出するピーク数
    gaussian_sigma: float = 1.0  # ガウシアンフィルタの強度
    # ビームフォーミング関連
    beamform_normalize: bool = True  # ビームフォーミング出力の正規化
    # スペクトログラム関連
    use_spectrogram: bool = False  # スペクトログラムを返すか
    nmf_components: int = 50  # NMFの成分数
    nmf_threshold: float = 1.6e-3  # NMFマスクの閾値
    # 画像処理オプション
    use_gaussian_filter: bool = False  # SoundMapにガウシアンフィルタ
    use_temporal_smoothing: bool = False  # 時間的平滑化
    temporal_smoothing_weight: float = 0.5  # 時間的平滑化の重み
    use_feature: bool = False  # 特徴画像を生成
    # 閾値
    feature_threshold: float = 0.9  # 2値化の閾値
    marker_size: int = 5  # マーカーのサイズ
    # 音源ファイル関連
    audio_file_path: Optional[str] = None  # 音源ファイルのパス（Noneの場合はホワイトノイズ）
    processing_time: float = 1.0  # シミュレーションで使用する音源の長さ（秒）
    noise_intensity: float = 0.0  # ノイズ強度（マイク信号に加算するノイズの強度）
    # Cubeの色
    same_color: bool = True
    update_freq: int = 1 # 5 # update_freq回呼び出されるごとに情報を更新

class SoundCamera:
    """音響シミュレーションとSoundMap生成を行うカメラクラス"""
    
    def __init__(self, target, config: SoundConfig):
        self.target = target
        self.config = config
        self.frames = []
        # マイクロフォン位置の初期化
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
        # 更新頻度管理用
        self.call_count = 0
        self.cached_sound_map0 = None
        self.cached_sound_map1 = None
        self.cached_spectrogram = None
    
    def _load_audio_file(self, audio_file_path: str):
        """音声ファイルを読み込む（WAV、MP3など）"""
        format = audio_file_path.split(".")[-1]
        sound = AudioSegment.from_file(audio_file_path, format=format)
        sound = sound.set_frame_rate(self.config.fs).set_channels(1)
        signal = np.array(sound.get_array_of_samples())
        self.audio_signal = signal
    
    def _generate_default_mic_positions(self) -> List[List[float]]:
        """デフォルトのマイクロフォン位置を生成（10x10x3の部屋用）"""
        positions = []
        for i in range(self.config.mic_array_num):
            theta = np.pi * (4*i - self.config.mic_array_num + 2) / (2 * self.config.mic_array_num)
            x = 5.0 + self.config.mic_array_radius * np.cos(theta)
            y = 5.0 + self.config.mic_array_radius * np.sin(theta)
            positions.append([x, y, 0.1])
        return positions
    
    def render(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        SoundMapとスペクトログラムを生成
        Returns:
            sound_map0: (height, width, 3) の画像 (uint8) - チャンネル0-2
            sound_map1: (height, width, 3) の画像 (uint8) - チャンネル3-5
            spectrogram: (height, width, 3) の画像 (uint8) or None
        """
        # 更新頻度のチェック
        should_update = (self.call_count % self.config.update_freq == 0)
        self.call_count += 1
        
        # キャッシュが存在し、更新不要な場合は古い画像を返す
        if not should_update and self.cached_sound_map0 is not None:
            return self.cached_sound_map0, self.cached_sound_map1, self.cached_spectrogram
        
        # 音源位置の取得
        sound_pos = self.target.get_pos()
        if isinstance(sound_pos, torch.Tensor):
            sound_pos = sound_pos.cpu().numpy()
        # genesisの座標系からpyroomacousticsの座標系へ変換
        sound_pos += np.array([4.85, 5.0, 0.0])
        mic_signals_list, music_results = self._simulate_all_arrays(sound_pos)
        sound_maps = []
        for i in range(self.config.mic_array_num):
            sound_map = self._generate_soundmap_from_doa(
                music_results[i],
                self.mic_positions[i]
            )
            sound_maps.append(sound_map)
        if not sound_maps:
            print("Warning: All sound simulations failed. Returning zero array.")
            num_channels = 3 if self.config.use_feature else self.config.mic_array_num
            sound_map_image = np.zeros(
                (self.config.observation_height, self.config.observation_width, num_channels),
                dtype=np.uint8
            )
            self.frames.append(sound_map_image)
            return sound_map_image, None
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
        sound_map_array = self._normalize_to_uint8(sound_map_array)
        # 時間的平滑化
        if self.config.use_temporal_smoothing:
            sound_map_array = self._apply_temporal_smoothing(sound_map_array)
        # チャンネルを2つの3チャンネル画像に分割
        sound_map0 = self._split_channels(sound_map_array, 0, 3)
        sound_map1 = self._split_channels(sound_map_array, 3, 6)
        self.frames.append(sound_map_array)
        # スペクトログラム生成
        spectrogram = None
        if self.config.use_spectrogram:
            spectrogram = self._generate_spectrogram(
                sound_maps, # ここは個別のマップのリスト (M, H, W)
                mic_signals_list,
            )
            if spectrogram is not None:
                spectrogram = self._pad_to_3ch(spectrogram)
        
        # 生成した画像をキャッシュに保存
        self.cached_sound_map0 = sound_map0
        self.cached_sound_map1 = sound_map1
        self.cached_spectrogram = spectrogram
        
        return sound_map0, sound_map1, spectrogram
    
    def _simulate_all_arrays(
        self,
        sound_pos: np.ndarray
    ) -> Tuple[List[np.ndarray], List[pra.doa.MUSIC]]:
        """
        すべてのマイクアレイを1つの部屋でシミュレーション
        Returns:
            mic_signals_list: 各マイクアレイの信号リスト
            music_results: 各マイクアレイのMUSIC結果リスト
        """
        room_dim = [10, 10, self.config.room_height]
        room = pra.ShoeBox(room_dim, fs=self.config.fs, max_order=self.config.room_max_order)
        room.add_source(sound_pos)
        for i in range(self.config.mic_array_num):
            mic_array_positions = self._generate_circular_array(
                self.mic_positions[i],
                self.config.mics_per_array,
                self.config.mic_radius
            )
            room.add_microphone_array(mic_array_positions)
        room.compute_rir()
        required_length = int(self.config.fs * self.config.processing_time)
        if self.audio_signal is not None:
            signal = self.audio_signal.copy()
            if len(signal) < required_length:
                num_repeats = int(np.ceil(required_length / len(signal)))
                signal = np.tile(signal, num_repeats)
            signal = signal[:required_length]
        else:
            signal = np.random.randn(required_length)
        room.sources[0].signal = signal
        room.simulate()
        mic_signals_list = []
        for i in range(self.config.mic_array_num):
            start_idx = i * self.config.mics_per_array
            end_idx = (i + 1) * self.config.mics_per_array
            signals = room.mic_array.signals[start_idx:end_idx]
            if self.config.noise_intensity > 0:
                noise = np.random.randn(*signals.shape) * np.mean(signals) * self.config.noise_intensity
                signals = signals*(1.0 - self.config.noise_intensity) + noise
            mic_signals_list.append(signals)
        music_results = []
        for i in range(self.config.mic_array_num):
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
            mic_array_positions = self._generate_circular_array(
                self.mic_positions[i],
                self.config.mics_per_array,
                self.config.mic_radius
            )
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
        spec = np.log10(np.mean(doa.Pssl, axis=1))
        spec /= np.sum(spec) # リファレンスコードのmake_2dmap内の正規化
        # マップの解像度
        map_scale = self.config.observation_height / (2*self.config.mic_array_radius)
        cx = mic_center[0] - 5.0 + self.config.mic_array_radius
        cy = mic_center[1] - 5.0 + self.config.mic_array_radius
        # 2Dマップを生成
        sound_map = np.zeros((self.config.observation_height, self.config.observation_width))
        for i in range(self.config.observation_height):
            for j in range(self.config.observation_width):
                # ピクセル座標を実空間座標に変換 (i, j -> x, y)
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
        mean_map = np.mean(sound_map, axis=2)
        marker_map = np.zeros_like(mean_map)
        max_coords = np.unravel_index(np.argmax(mean_map), mean_map.shape)
        row, col = int(max_coords[0]), int(max_coords[1])
        size = self.config.marker_size
        row_min = max(0, row - size)
        row_max = min(mean_map.shape[0], row + size + 1)
        col_min = max(0, col - size)
        col_max = min(mean_map.shape[1], col + size + 1)
        marker_map[row_min:row_max, col_min:col_max] = 255
        binary_map = np.where(mean_map > np.max(mean_map) * self.config.feature_threshold, 255, 0)
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
        current_normalized = current_frame.astype(np.float32) / 255.0
        # 重み付き平均
        weight = self.config.temporal_smoothing_weight
        self.past_frame = self.past_frame * (1 - weight) + current_normalized * weight * 255
        self.past_frame = np.clip(self.past_frame, 0, 255)
        return self.past_frame.astype(np.uint8)

    def _perform_spotforming_nmf(
        self,
        beamformed_signals, # ある1つの音源に対する全アレイのビームフォーミング結果 (M, N_samples)
        fs,
        n_components,
        threshold=0.01,
        nfft=512,
        noverlap=256
    ):
        """
        論文のセクションIII-B に記載のNMFベースのスポットフォーミングを実行する
        """
        M = len(beamformed_signals)
        all_amp_specs = []
        all_phases = []
        for wav in beamformed_signals:
            f_stft, t_stft, Zxx = stft(wav, fs=fs, nperseg=nfft, noverlap=noverlap)
            amp_spec = np.abs(Zxx) 
            phase = np.angle(Zxx)
            all_amp_specs.append(amp_spec)
            all_phases.append(phase)
        concatenated_spec = np.concatenate(all_amp_specs, axis=1)
        # F, T_total = concatenated_spec.shape
        # T_frame = all_amp_specs[0].shape[1]
        model = NMF(n_components=n_components, init='nndsvd', random_state=0, max_iter=2000, tol=1e-3)
        W = model.fit_transform(concatenated_spec)
        H = model.components_
        split_H = np.array_split(H, M, axis=1)
        stacked_H = np.stack(split_H, axis=0)
        min_activations = np.min(stacked_H, axis=0)
        binary_mask = (min_activations > threshold).astype(float) # これが B_i
        reconstructed_wavs = []
        for m in range(M):
            H_m = split_H[m]      # H_i^(m)
            phase_m = all_phases[m] # 位相
            reconstructed_amp_spec = W @ (H_m * binary_mask)
            reconstructed_complex_spec = reconstructed_amp_spec * np.exp(1j * phase_m)
            _, wav_m = istft(reconstructed_complex_spec, fs=fs, nperseg=nfft, noverlap=noverlap)
            reconstructed_wavs.append(wav_m)
        max_len = max(len(w) for w in reconstructed_wavs)
        padded_wavs = []
        for w in reconstructed_wavs:
            if len(w) < max_len:
                padded_wavs.append(np.pad(w, (0, max_len - len(w))))
            else:
                padded_wavs.append(w)
        final_wav = np.mean(np.stack(padded_wavs, axis=0), axis=0)
        return final_wav, f_stft, t_stft # f_stft, t_stft はNMFの結果のものを返す（が、ここでは使わない）

    def _generate_spectrogram(
        self,
        sound_maps: List[np.ndarray], # (M, H, W) のリスト
        mic_signals_list: List[Optional[np.ndarray]],
    ) -> Optional[np.ndarray]:
        """
        NMFベースのスポットフォーミングを実行し、
        分離された音源のスペクトログラムを画像化して返す
        """
        combined_map = np.sum(sound_maps, axis=0)
        smoothed_map = gaussian_filter(combined_map, self.config.gaussian_sigma)
        top_peaks = self._find_top_k_peaks(smoothed_map, self.config.num_peaks)
        if not top_peaks:
            print("Spotforming: No peaks found.")
            return None
        beamform_wav_per_peak = []
        for (peak_x, peak_y) in top_peaks:
            signals_for_this_peak = []
            for i in range(self.config.mic_array_num):
                mic_signals = mic_signals_list[i]
                mic_pos_center = self.mic_positions[i] # アレイ中心 [x, y, z]
                if mic_signals is None:
                    print(f"Warning: Missing signals for mic array {i}. Skipping peak.")
                    signals_for_this_peak = [] # このピークは無効
                    break
                mic_array_pos_abs = self._generate_circular_array(
                    mic_pos_center,
                    self.config.mics_per_array,
                    self.config.mic_radius
                ).T # (M_mics, 3)
                center = np.mean(mic_array_pos_abs, axis=0)
                mic_array_pos_rel = mic_array_pos_abs - center
                theta_deg = self._pixel_to_azimuth(peak_x, peak_y, mic_pos_center)
                beamformed_signal = self._ds_beamform(
                    mic_signals,
                    mic_array_pos_rel,
                    theta_deg
                )
                signals_for_this_peak.append(beamformed_signal)
            if signals_for_this_peak: # すべてのアレイで成功した場合
                beamform_wav_per_peak.append(signals_for_this_peak)

        if not beamform_wav_per_peak:
            print("Spotforming: No valid peaks found after beamforming.")
            return None
        signals_for_first_peak = beamform_wav_per_peak[0] # (M_arrays, N_samples)
        final_wav, _, _ = self._perform_spotforming_nmf(
            beamformed_signals=signals_for_first_peak,
            fs=self.config.fs,
            n_components=self.config.nmf_components,
            threshold=self.config.nmf_threshold,
            nfft=self.config.nfft,
            noverlap=self.config.nfft // 2
        )
        final_wav = final_wav[:int(self.config.fs * self.config.processing_time)]
        f, t, Zxx = stft(
            final_wav, 
            fs=self.config.fs, 
            nperseg=self.config.nfft, 
            noverlap=self.config.nfft // 2
        )
        power_spec = np.abs(Zxx)**2
        spec_db = 10 * np.log10(power_spec + 1e-10)
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
        min_val = np.min(spec_db)
        max_val = np.max(spec_db)
        if max_val > min_val:
            normalized = (spec_db - min_val) / (max_val - min_val) * 255
        else:
            normalized = np.ones_like(spec_db) * 127.5
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
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
        is_peak = np.ones_like(data, dtype=bool)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                shifted_data = np.roll(data, shift=(dr, dc), axis=(0, 1))
                comparison = data >= shifted_data
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
        peak_rows, peak_cols = np.where(is_peak)
        peak_values = data[peak_rows, peak_cols]
        sorted_indices = np.argsort(peak_values)[::-1]
        top_k_indices = sorted_indices[:min(k, len(sorted_indices))]
        room_size = 10.0
        map_scale = self.config.observation_height / room_size
        return [(peak_rows[i] / map_scale, peak_cols[i] / map_scale) for i in top_k_indices]
    
    def _pixel_to_azimuth(
        self,
        x: float, # 実空間座標 x
        y: float, # 実空間座標 y
        mic_center: List[float]
    ) -> float:
        """実空間座標から方位角（度）を計算（test.pyに合わせた方式）"""
        dx = x - mic_center[0]
        dy = y - mic_center[1]
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
        if mic_signals.shape[0] < mic_signals.shape[1]:
            y_multi = mic_signals.T
        else:
            y_multi = mic_signals
        N, M = y_multi.shape
        Y = rfft(y_multi, axis=0)
        freqs = rfftfreq(N, d=1.0/self.config.fs)
        u = self._unit_vec_from_angles(azimuth_deg, elevation_deg)
        A = self._steering_vector(freqs, mic_positions, u)
        Y_beam = np.sum(A.conj() * Y, axis=1)
        if self.config.beamform_normalize:
            Y_beam /= M
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
    
    def _split_channels(
        self,
        array: np.ndarray,
        start_ch: int,
        end_ch: int
    ) -> np.ndarray:
        """
        配列から指定範囲のチャンネルを抽出し、3チャンネルにゼロ埋め
        Args:
            array: (height, width, channels) の画像
            start_ch: 開始チャンネル
            end_ch: 終了チャンネル（含まない）
        Returns:
            result: (height, width, 3) の画像（不足分はゼロ埋め）
        """
        height, width, total_channels = array.shape
        result = np.zeros((height, width, 3), dtype=array.dtype)
        available_channels = min(end_ch, total_channels) - start_ch
        available_channels = max(0, available_channels)  # 負の値を防ぐ
        for i in range(min(available_channels, 3)):
            if start_ch + i < total_channels:
                result[:, :, i] = array[:, :, start_ch + i]
        return result
    
    def _pad_to_3ch(self, array: np.ndarray) -> np.ndarray:
        """
        2D配列または1/2チャンネル配列を3チャンネルにゼロ埋め
        Args:
            array: (height, width) または (height, width, 1) または (height, width, 2)
        Returns:
            result: (height, width, 3)
        """
        if array.ndim == 2:
            result = np.zeros((array.shape[0], array.shape[1], 3), dtype=array.dtype)
            result[:, :, 0] = array
        elif array.ndim == 3:
            if array.shape[2] == 3:
                return array
            else:
                result = np.zeros((array.shape[0], array.shape[1], 3), dtype=array.dtype)
                for i in range(min(array.shape[2], 3)):
                    result[:, :, i] = array[:, :, i]
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")
        return result
