import numpy as np
import torch
import pyroomacoustics as pra
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal
from scipy.signal import stft, istft
from numpy.fft import rfft, irfft, rfftfreq
from scipy.ndimage import gaussian_filter
from pydub import AudioSegment
from torchnmf.nmf import NMF as TorchNMF
import time


REALTIME_NMF_INIT_MAX_ITER = 40
REALTIME_NMF_WARM_MAX_ITER = 15
REALTIME_NMF_TOL = 2e-2
REALTIME_NMF_EPS = 1e-12
REALTIME_NMF_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REALTIME_NMF_BETA = 2
REALTIME_NMF_MAX_MASK_RATIO = 0.2

@dataclass
class SoundConfig:
    """Configuration for acoustic simulation and SoundMap generation"""
    # Image size
    observation_height: int = 128
    observation_width: int = 128
    # Microphone array settings
    mic_array_num: int = 6 # Number of microphone arrays
    mic_array_radius: float = 0.3 # Radius of the array circle when placed in a circle (meters)
    mics_per_array: int = 8 # Number of mics per array
    mic_radius: float = 0.035 # Radius of the mic layout within a microphone array (meters)
    # Acoustic simulation settings
    fs: int = 16000 # Sampling frequency
    nfft: int = 512 # FFT length
    room_max_order: int = 3 # Maximum reflection order
    sound_speed: float = 343.0 # Speed of sound (m/s)
    room_height: float = 3.0 # Room height
    # MUSIC method settings
    music_num_src: int = 3 # Number of sources to estimate
    # Peak detection settings (for beamforming)
    num_peaks: int = 1 # Number of peaks to detect
    gaussian_sigma: float = 1.0 # Gaussian filter strength
    # Beamforming settings
    beamform_normalize: bool = True # Normalize beamforming output
    # Spectrogram settings
    use_spectrogram: bool = False # Whether to return the spectrogram
    use_soundmap: bool = True # Whether to return the sound environment map (sound0, sound1)
    nmf_components: int = 50 # Number of NMF components
    nmf_threshold: float = 1.6e-3 # NMF mask threshold
    spectrogram_display_min_hz: float = 0.0
    spectrogram_display_max_hz: Optional[float] = 4000.0
    spectrogram_normalization: Literal["minmax", "percentile"] = "percentile"
    # Image processing options
    use_gaussian_filter: bool = False # Apply Gaussian filter to SoundMap
    use_temporal_smoothing: bool = False # Temporal smoothing
    temporal_smoothing_weight: float = 0.5 # Temporal smoothing weight
    use_feature: bool = False # Generate feature image
    # Threshold
    feature_threshold: float = 0.9 # Binarization threshold
    marker_size: int = 5 # Marker size
    # Source file settings
    audio_file_path: Optional[str] = None # Path to source file (white noise if None)
    processing_time: float = 1.0 # Length of source used in the simulation (seconds)
    noise_intensity: float = 0.0 # Noise intensity (strength of noise added to mic signals)
    # Cube color
    same_color: bool = True
    update_freq: int = 5 # Update info every update_freq calls
    # Shake mode
    shake_mode: bool = False
    velocity_threshold: float = 0.0005
    # Sound All mode
    sound_all_mode: bool = False
    sound_all_moving_audio_path: Optional[str] = None  # Path to the source file switched to when moving
    # Common sound observation settings
    azimuth_offset_deg: float = 0.0
    doa_distance_floor_m: float = 10.0
    doa_distance_decay_exponent: float = 1.0
    combined_map_power: float = 1.0
    combined_map_gaussian_sigma: Optional[float] = None
    peak_selection_mode: Literal["local_max", "argmax"] = "local_max"
    doa_azimuth_grid_size: Optional[int] = None
    doa_use_grid_azimuth: bool = False
    doa_spectrum_mode: Literal["legacy", "relative_db"] = "legacy"
    doa_spectrum_noise_percentile: float = 50.0
    doa_spectrum_dynamic_range_db: float = 18.0
    doa_frequency_normalization: bool = False

class SoundCamera:
    """Camera class that performs acoustic simulation and SoundMap generation"""

    def __init__(self, target, config: SoundConfig):
        self._target = None
        self.config = config
        self.elapsed_time = 0.0
        self.frames = []
        # Initialize microphone positions
        self.mic_positions = self._generate_default_mic_positions()
        # Load audio file
        self.audio_signal = None
        self.moving_audio_signal = None
        self._nmf_state = None
        self._local_soundmap_grid = None
        if config.audio_file_path is not None:
            self._load_audio_file(config.audio_file_path)
        # Past frame for temporal smoothing
        self.past_frame = None
        if config.use_temporal_smoothing:
            num_channels = 3 if config.use_feature else config.mic_array_num
            self.past_frame = np.ones(
                (config.observation_height, config.observation_width, num_channels),
                dtype=np.float32
            ) * 127.5
        # Update frequency management
        self.call_count = 0
        self.cached_sound_map0 = None
        self.cached_sound_map1 = None
        self.cached_spectrogram = None

        # For velocity computation
        self.prev_pos = None
        self.prev_time = None

        # For signal buffering
        self.required_length = int(config.fs * config.processing_time)
        self.signal_buffer = np.zeros(self.required_length, dtype=np.float32)
        self.audio_cursor = 0

        # Velocity history (for shake_mode/sound_all_mode)
        # List of (velocity, num_samples)
        self.velocity_history = []

        # For sound_all_mode: source signal used when moving
        self.moving_audio_cursor = 0
        if config.sound_all_mode and config.sound_all_moving_audio_path is not None:
            self._load_moving_audio_file(config.sound_all_moving_audio_path)
        self.target = target
    
    @property
    def target(self):
        return self._target
    
    @target.setter
    def target(self, val):
        self._target = val
        self.reset_state()
    
    def reset_state(self):
        self.prev_pos = None
        self.prev_time = None
        self.velocity_history = []
        self.elapsed_time = 0.0
        self.signal_buffer = np.zeros(self.required_length, dtype=np.float32)
        self.audio_cursor = 0
        self.moving_audio_cursor = 0
        self.call_count = 0
        self.cached_sound_map0 = None
        self.cached_sound_map1 = None
        self.cached_spectrogram = None
        self.reset_nmf_state()

    def reset_nmf_state(self):
        self._nmf_state = None

    def get_target_velocity(self) -> float:
        """Compute the target's velocity"""
        current_pos = self.target.get_pos()
        if isinstance(current_pos, torch.Tensor):
            current_pos = current_pos.cpu().numpy()
        
        current_time = time.time()
        
        if self.prev_pos is None:
            self.prev_pos = current_pos
            self.prev_time = current_time
            return 0.0
        
        dt = current_time - self.prev_time
        if dt == 0:
            return 0.0
            
        dist = np.linalg.norm(current_pos - self.prev_pos)
        velocity = dist / dt
        
        self.prev_pos = current_pos
        self.prev_time = current_time
        
        return velocity
    
    def _load_audio_file(self, audio_file_path: str):
        """Load an audio file (WAV, MP3, etc.)"""
        format = audio_file_path.split(".")[-1]
        sound = AudioSegment.from_file(audio_file_path, format=format)
        sound = sound.set_frame_rate(self.config.fs).set_channels(1)
        signal = np.array(sound.get_array_of_samples())
        # Normalize (-1.0 ~ 1.0)
        if signal.dtype != np.float32:
            signal = signal.astype(np.float32)
        if np.abs(signal).max() > 0:
            signal = signal / np.abs(signal).max()
        self.audio_signal = signal
        self.audio_cursor = 0

    def _load_moving_audio_file(self, audio_file_path: str):
        """Load the audio file used when moving (for sound_all_mode)"""
        format = audio_file_path.split(".")[-1]
        sound = AudioSegment.from_file(audio_file_path, format=format)
        sound = sound.set_frame_rate(self.config.fs).set_channels(1)
        signal = np.array(sound.get_array_of_samples())
        if signal.dtype != np.float32:
            signal = signal.astype(np.float32)
        if np.abs(signal).max() > 0:
            signal = signal / np.abs(signal).max()
        self.moving_audio_signal = signal

    def set_moving_audio(self, audio_file_path: str):
        """Dynamically change the source used when moving (for sound_all_mode)"""
        self._load_moving_audio_file(audio_file_path)
        self.moving_audio_cursor = 0

    def _get_audio_chunk(self, n_samples: int) -> np.ndarray:
        """Get the specified number of samples from the audio file (looped playback)"""
        if self.audio_signal is None:
            return np.random.randn(n_samples).astype(np.float32)
        
        chunk = np.zeros(n_samples, dtype=np.float32)
        total_len = len(self.audio_signal)
        current_idx = 0
        
        while current_idx < n_samples:
            remaining = n_samples - current_idx
            available = total_len - self.audio_cursor
            
            take = min(remaining, available)
            chunk[current_idx:current_idx+take] = self.audio_signal[self.audio_cursor:self.audio_cursor+take]
            
            self.audio_cursor = (self.audio_cursor + take) % total_len
            current_idx += take
            
        return chunk

    def _update_state(self, dt: float, velocity: float):
        """Update state (signal buffer and velocity history)"""
        self.elapsed_time += dt
        # Number of samples for the elapsed time
        n_samples = int(dt * self.config.fs)
        if n_samples == 0:
            return

        # 1. Get a new source chunk
        new_chunk = self._get_audio_chunk(n_samples)
        # Create a new buffer by concatenating new_chunk and the previous buffer (minus last n_samples)
        if n_samples >= self.required_length:
            self.signal_buffer = new_chunk[:self.required_length][::-1] # Just take new chunk if it's too long? No, logic says prepend.
            pass
        
        # Efficient rolling
        self.signal_buffer = np.roll(self.signal_buffer, n_samples)
        self.signal_buffer[:n_samples] = new_chunk[::-1] # Store new chunk in reverse so index 0 is newest sample?

        self.signal_buffer[n_samples:] = self.signal_buffer[:-n_samples]
        self.signal_buffer[:n_samples] = new_chunk[::-1] # Store time-reversed new chunk?

        # 3. Update velocity history
        self.velocity_history.insert(0, (velocity, n_samples))
        total_samples = 0
        valid_history = []
        for v, s in self.velocity_history:
            if total_samples >= self.required_length:
                break
            valid_history.append((v, s))
            total_samples += s
        self.velocity_history = valid_history
    
    def _generate_default_mic_positions(self) -> List[List[float]]:
        """Generate default microphone positions (for a 10x10x3 room)"""
        positions = []
        for i in range(self.config.mic_array_num):
            theta = np.pi * (4*i - self.config.mic_array_num - 3) / (2 * self.config.mic_array_num)
            x = 5.0 + self.config.mic_array_radius * np.cos(theta)
            y = 5.0 + self.config.mic_array_radius * np.sin(theta)
            positions.append([x, y, 0.1])
        return positions
    
    def render(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Generate SoundMap and spectrogram
        Returns:
            sound_map0: (height, width, 3) image (uint8) - channels 0-2
            sound_map1: (height, width, 3) image (uint8) - channels 3-5
            spectrogram: (height, width, 3) image (uint8) or None
        """
        # Check update frequency
        should_update = (self.call_count % self.config.update_freq == 0)
        self.call_count += 1

        # Update state (always run)
        last_time = self.prev_time
        velocity = self.get_target_velocity()
        
        if last_time is None:
            dt = 0.033 # Assume 30FPS for first frame
        else:
            dt = self.prev_time - last_time
            if dt > 0.1: dt = 0.1 # Cap at 100ms to avoid huge jumps
        
        self._update_state(dt, velocity)

        signal_to_process = self.signal_buffer.copy()
        zero_flag = False
        if self.config.shake_mode:
            # Create mask
            mask = np.zeros_like(signal_to_process)
            silent_time = 2.0
            if self.elapsed_time >= silent_time:
                current_idx = 0
                for v, s in self.velocity_history:
                    if current_idx >= len(mask):
                        break
                    end_idx = min(current_idx + s, len(mask))
                    if v >= self.config.velocity_threshold:
                        mask[current_idx:end_idx] = 1.0
                    else:
                        mask[current_idx:end_idx] = 0.0
                    current_idx = end_idx
            
            signal_to_process *= mask
            
            if np.all(signal_to_process == 0):
                signal_to_process += np.random.randn(*signal_to_process.shape) * 1e-5
                zero_flag = True
        elif self.config.sound_all_mode and self.moving_audio_signal is not None:
            # sound_all_mode: switch between stationary (sound A) and moving (sound B/C) based on velocity
            # Generate the source chunk for moving
            moving_chunk = np.zeros_like(signal_to_process)
            n_total = len(moving_chunk)
            cursor = self.moving_audio_cursor
            total_len = len(self.moving_audio_signal)
            idx = 0
            while idx < n_total:
                remaining = n_total - idx
                available = total_len - cursor
                take = min(remaining, available)
                moving_chunk[idx:idx+take] = self.moving_audio_signal[cursor:cursor+take]
                cursor = (cursor + take) % total_len
                idx += take
            
            # Velocity-based blend: moving source while moving, original source while stationary
            # Do not switch sound during silent_time right after start
            blend_mask = np.zeros(n_total, dtype=np.float32)
            silent_time = 2.0
            if self.elapsed_time >= silent_time:
                current_idx = 0
                for v, s in self.velocity_history:
                    if current_idx >= n_total:
                        break
                    end_idx = min(current_idx + s, n_total)
                    if v >= self.config.velocity_threshold:
                        blend_mask[current_idx:end_idx] = 1.0
                    current_idx = end_idx
            
            # Moving portions use the moving source, stationary portions use the original source
            signal_to_process = signal_to_process * (1.0 - blend_mask) + moving_chunk * blend_mask
        signal_for_simulation = signal_to_process[::-1]
        if not should_update and self.cached_sound_map0 is not None:
            return self.cached_sound_map0, self.cached_sound_map1, self.cached_spectrogram
        # Get source position
        sound_pos = self.target.get_pos()
        if isinstance(sound_pos, torch.Tensor):
            sound_pos = sound_pos.cpu().numpy()
        sound_pos += np.array([4.5, 5.0, 0.0])
        sound_maps = []
        mic_signals_list = []
        music_results = []
        
        if self.config.mic_array_num > 0:
            # Run simulation (pass signal)
            mic_signals_list, music_results = self._simulate_all_arrays(sound_pos, signal_for_simulation, zero_flag)
            
            for i in range(self.config.mic_array_num):
                sound_map = self._generate_soundmap_from_doa(
                    music_results[i],
                    self.mic_positions[i]
                )
                sound_maps.append(sound_map)
        if not sound_maps:
            print("Warning: All sound simulations failed or mic_array_num=0. Returning zero array.")
            # Always return two 3-channel zero images
            sound_map_image = np.zeros(
                (self.config.observation_height, self.config.observation_width, 3),
                dtype=np.uint8
            )
            self.frames.append(sound_map_image)
            # sound_map0, sound_map1, spectrogram
            return sound_map_image, sound_map_image, None 
        sound_map_array = np.array(sound_maps).transpose(1, 2, 0)
        # Apply Gaussian filter
        if self.config.use_gaussian_filter:
            for i in range(sound_map_array.shape[2]):
                sound_map_array[:, :, i] = gaussian_filter(
                    sound_map_array[:, :, i],
                    sigma=self.config.gaussian_sigma
                )
        # Convert to feature image
        if self.config.use_feature:
            sound_map_array = self._convert_to_feature_image(sound_map_array)
        sound_map_array = self._normalize_to_uint8(sound_map_array)
        # Temporal smoothing
        if self.config.use_temporal_smoothing:
            sound_map_array = self._apply_temporal_smoothing(sound_map_array)
        # Split channels into two 3-channel images
        sound_map0 = self._split_channels(sound_map_array, 0, 3)
        sound_map1 = self._split_channels(sound_map_array, 3, 6)
        self.frames.append(sound_map_array)
        # Generate spectrogram
        spectrogram = None
        if self.config.use_spectrogram:
            spectrogram = self._generate_spectrogram(
                sound_maps, # This is a list of individual maps (M, H, W)
                mic_signals_list,
            )
            if spectrogram is not None:
                spectrogram = self._pad_to_3ch(spectrogram)

        # Cache the generated images
        self.cached_sound_map0 = sound_map0
        self.cached_sound_map1 = sound_map1
        self.cached_spectrogram = spectrogram
        
        return sound_map0, sound_map1, spectrogram
    
    def _simulate_all_arrays(
        self,
        sound_pos: np.ndarray,
        signal: np.ndarray,
        zero_flag: bool = False
    ) -> Tuple[List[np.ndarray], List[pra.doa.MUSIC]]:
        """
        Simulate all microphone arrays in a single room
        Returns:
            mic_signals_list: list of signals for each microphone array
            music_results: list of MUSIC results for each microphone array
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
        
        # Set the signal
        room.sources[0].signal = signal
        room.simulate()
        mic_signals_list = []
        for i in range(self.config.mic_array_num):
            start_idx = i * self.config.mics_per_array
            end_idx = (i + 1) * self.config.mics_per_array
            signals = room.mic_array.signals[start_idx:end_idx]
            mean_signal = np.mean(signals)
            if self.config.noise_intensity > 0:
                noise = np.random.randn(*signals.shape) * mean_signal * self.config.noise_intensity
                signals = signals*(1.0 - self.config.noise_intensity) + noise
            mic_signals_list.append(signals)
        if zero_flag:
            doa_mic_signals_list = [np.zeros_like(signals) for signals in mic_signals_list]
        else:
            doa_mic_signals_list = mic_signals_list
        music_results = []
        for i in range(self.config.mic_array_num):
            music_results.append(self.estimate_music(doa_mic_signals_list[i], i))
        return mic_signals_list, music_results
    
    def _generate_circular_array(
        self,
        center: List[float],
        num_mics: int,
        radius: float
    ) -> np.ndarray:
        """Generate positions for a circular microphone array"""
        angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        z = np.ones(num_mics) * center[2]
        return np.array([x, y, z])
    
    def _generate_soundmap_from_doa(
        self,
        doa: pra.doa.MUSIC,
        mic_center: List[float],
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
        azimuth_offset_deg: Optional[float] = None,
        distance_floor_m: Optional[float] = None,
        distance_decay_exponent: Optional[float] = None,
    ) -> np.ndarray:
        """Generate a SoundMap from DOA estimation results"""
        spec = self._get_doa_spectrum(doa)
        azimuth_offset_deg = self.config.azimuth_offset_deg if azimuth_offset_deg is None else azimuth_offset_deg
        distance_floor_m = self.config.doa_distance_floor_m if distance_floor_m is None else distance_floor_m
        distance_decay_exponent = (
            self.config.doa_distance_decay_exponent
            if distance_decay_exponent is None
            else distance_decay_exponent
        )

        if x_coords is None or y_coords is None:
            x_grid, y_grid = self._get_local_soundmap_grid()
            cx = mic_center[0] - 5.0 + self.config.mic_array_radius
            cy = mic_center[1] - 5.0 + self.config.mic_array_radius
            theta = np.arctan2(y_grid - cy, x_grid - cx)
            distance = np.hypot(y_grid - cy, x_grid - cx)
        else:
            x_grid, y_grid = np.meshgrid(
                np.asarray(x_coords, dtype=np.float32),
                np.asarray(y_coords, dtype=np.float32),
                indexing="xy",
            )
            theta = np.arctan2(y_grid - mic_center[1], x_grid - mic_center[0]) + np.deg2rad(azimuth_offset_deg)
            distance = np.hypot(y_grid - mic_center[1], x_grid - mic_center[0])

        angle_idx = self._theta_to_doa_angle_indices(theta, doa, len(spec))
        if distance_decay_exponent == 0.0:
            distance_weight = np.ones_like(distance, dtype=np.float32)
        else:
            distance_weight = 1.0 / np.power(distance + distance_floor_m, distance_decay_exponent)
        return (spec[angle_idx] * distance_weight).astype(np.float32)

    def _get_doa_spectrum(self, doa: pra.doa.MUSIC) -> np.ndarray:
        if self.config.doa_spectrum_mode == "legacy":
            spec = np.log10(np.mean(doa.Pssl, axis=1))
            spec_sum = np.sum(spec)
            if spec_sum != 0:
                spec = spec / spec_sum
            return spec

        power = np.asarray(np.mean(doa.Pssl, axis=1), dtype=np.float32)
        power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
        power = np.maximum(power, 1e-12)
        spectrum_db = 10.0 * np.log10(power)
        baseline = np.percentile(
            spectrum_db,
            np.clip(float(self.config.doa_spectrum_noise_percentile), 0.0, 100.0),
        )
        dynamic_range = max(float(self.config.doa_spectrum_dynamic_range_db), 1e-6)
        spectrum = np.clip(spectrum_db - baseline, 0.0, dynamic_range) / dynamic_range
        peak = float(np.max(spectrum)) if spectrum.size else 0.0
        if peak > 0.0:
            spectrum = spectrum / peak
        return spectrum.astype(np.float32, copy=False)

    def _theta_to_doa_angle_indices(
        self,
        theta: np.ndarray,
        doa: pra.doa.MUSIC,
        spectrum_size: int,
    ) -> np.ndarray:
        if not self.config.doa_use_grid_azimuth:
            return (theta / (2 * np.pi / spectrum_size)).astype(np.int32) % spectrum_size

        azimuth = getattr(getattr(doa, "grid", None), "azimuth", None)
        if azimuth is None:
            return (theta / (2 * np.pi / spectrum_size)).astype(np.int32) % spectrum_size

        azimuth = np.asarray(azimuth, dtype=np.float32).reshape(-1)
        if azimuth.size != spectrum_size:
            return (theta / (2 * np.pi / spectrum_size)).astype(np.int32) % spectrum_size

        wrapped_grid = (azimuth + 2.0 * np.pi) % (2.0 * np.pi)
        order = np.argsort(wrapped_grid)
        sorted_grid = wrapped_grid[order]
        wrapped_theta = (theta + 2.0 * np.pi) % (2.0 * np.pi)
        right = np.searchsorted(sorted_grid, wrapped_theta, side="left") % spectrum_size
        left = (right - 1) % spectrum_size
        right_dist = np.abs(np.angle(np.exp(1j * (wrapped_theta - sorted_grid[right]))))
        left_dist = np.abs(np.angle(np.exp(1j * (wrapped_theta - sorted_grid[left]))))
        nearest_sorted = np.where(right_dist < left_dist, right, left)
        return order[nearest_sorted]

    def _get_local_soundmap_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._local_soundmap_grid is None:
            map_scale = self.config.observation_height / (2 * self.config.mic_array_radius)
            x_coords = np.arange(self.config.observation_height, dtype=np.float32) / map_scale
            y_coords = np.arange(self.config.observation_width, dtype=np.float32) / map_scale
            self._local_soundmap_grid = np.meshgrid(x_coords, y_coords, indexing="ij")
        return self._local_soundmap_grid
    
    def _convert_to_feature_image(self, sound_map: np.ndarray) -> np.ndarray:
        """
        Convert a SoundMap to a feature image
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
        """Scale the array to uint8 in the range 0-255"""
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
        """Apply temporal smoothing"""
        if self.past_frame is None:
            self.past_frame = current_frame.astype(np.float32)
            return current_frame
        current_normalized = current_frame.astype(np.float32) / 255.0
        # Weighted average
        weight = self.config.temporal_smoothing_weight
        self.past_frame = self.past_frame * (1 - weight) + current_normalized * weight * 255
        self.past_frame = np.clip(self.past_frame, 0, 255)
        return self.past_frame.astype(np.uint8)

    def _perform_spotforming_nmf(
        self,
        beamformed_signals, # Beamforming results from all arrays for one source (M, N_samples)
        fs,
        n_components,
        threshold=0.01,
        nfft=512,
        noverlap=256
    ):
        """
        Perform the NMF-based spotforming described in Section III-B of the paper
        """
        M = len(beamformed_signals)
        all_amp_specs = []
        all_phases = []
        f_stft = None
        t_stft = None
        for wav in beamformed_signals:
            f_stft, t_stft, Zxx = stft(wav, fs=fs, nperseg=nfft, noverlap=noverlap)
            amp_spec = np.abs(Zxx).astype(np.float32, copy=False)
            phase = np.angle(Zxx)
            all_amp_specs.append(amp_spec)
            all_phases.append(phase)
        concatenated_spec = np.concatenate(all_amp_specs, axis=1).astype(np.float32, copy=False)
        W, H = self._fit_realtime_nmf(concatenated_spec, n_components)
        split_H = np.array_split(H, M, axis=1)
        stacked_H = np.stack(split_H, axis=0)
        min_activations = np.min(stacked_H, axis=0)
        max_activation = float(np.max(min_activations))
        if max_activation > 0.0:
            normalized_activations = min_activations / max_activation
        else:
            normalized_activations = min_activations
        binary_mask = (normalized_activations > threshold).astype(np.float64)
        mask_ratio = float(binary_mask.mean()) if binary_mask.size else 0.0
        if mask_ratio > REALTIME_NMF_MAX_MASK_RATIO:
            adaptive_threshold = float(
                np.quantile(normalized_activations, 1.0 - REALTIME_NMF_MAX_MASK_RATIO)
            )
            binary_mask = (normalized_activations >= max(threshold, adaptive_threshold)).astype(np.float64)
        if not np.any(binary_mask) and max_activation > 0.0:
            binary_mask = (normalized_activations >= np.max(normalized_activations)).astype(np.float64)
        reconstructed_wavs = []
        for m in range(M):
            H_m = split_H[m]      # H_i^(m)
            phase_m = all_phases[m] # Phase
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
        return final_wav, f_stft, t_stft # Return f_stft, t_stft from the NMF result (unused here)

    def _fit_realtime_nmf(
        self,
        concatenated_spec: np.ndarray,
        n_components: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        concatenated_spec = np.asarray(concatenated_spec, dtype=np.float32)
        n_components = min(
            int(n_components),
            int(concatenated_spec.shape[0]),
            int(concatenated_spec.shape[1]),
        )
        if n_components <= 0:
            raise RuntimeError(f"Invalid NMF input shape: {concatenated_spec.shape}")

        device_name = REALTIME_NMF_DEVICE
        device = torch.device(device_name)
        state = self._nmf_state
        use_warm_start = (
            state is not None
            and state.get("shape") == concatenated_spec.shape
            and state.get("n_components") == n_components
            and state.get("device") == device_name
            and isinstance(state.get("model"), TorchNMF)
        )

        model = state["model"] if use_warm_start and state is not None else None
        if model is None:
            model = TorchNMF(concatenated_spec.shape, rank=n_components).to(device)

        target = torch.from_numpy(concatenated_spec).to(device)
        try:
            model.fit(
                target,
                beta=REALTIME_NMF_BETA,
                tol=REALTIME_NMF_TOL,
                max_iter=REALTIME_NMF_WARM_MAX_ITER if use_warm_start else REALTIME_NMF_INIT_MAX_ITER,
                verbose=False,
            )
        except Exception:
            self._nmf_state = None
            model = TorchNMF(concatenated_spec.shape, rank=n_components).to(device)
            model.fit(
                target,
                beta=REALTIME_NMF_BETA,
                tol=REALTIME_NMF_TOL,
                max_iter=REALTIME_NMF_INIT_MAX_ITER,
                verbose=False,
            )

        W = model.H.detach().cpu().numpy().astype(np.float64, copy=False)
        H = model.W.detach().cpu().numpy().T.astype(np.float64, copy=False)
        self._nmf_state = {
            "shape": concatenated_spec.shape,
            "n_components": n_components,
            "device": device_name,
            "model": model,
        }
        return np.maximum(W, REALTIME_NMF_EPS), np.maximum(H, REALTIME_NMF_EPS)

    def _generate_spectrogram(
        self,
        sound_maps: List[np.ndarray], # List of (M, H, W)
        mic_signals_list: List[Optional[np.ndarray]],
    ) -> Optional[np.ndarray]:
        """
        Perform NMF-based spotforming and return the separated source's
        spectrogram rendered as an image
        """
        combined_map = self.build_combined_sound_map(
            sound_maps,
            power=self.config.combined_map_power,
            gaussian_sigma=self.config.combined_map_gaussian_sigma,
        )
        top_peaks = self.compute_top_peaks(
            combined_map,
            topk=self.config.num_peaks,
            selection_mode=self.config.peak_selection_mode,
        )
        if not top_peaks:
            print("Spotforming: No peaks found.")
            return None
        reconstructed = self.reconstruct_primary_peak(
            mic_signals_list,
            top_peaks,
        )
        if reconstructed is None:
            print("Spotforming: No valid peaks found after beamforming.")
            return None
        return self.create_spectrogram_image_from_audio(reconstructed)

    def _convert_spectrogram_to_image(self, spec_db: np.ndarray) -> np.ndarray:
        """
        Render the spectrogram as an image (resized to observation_height x observation_width)
        Args:
            spec_db: log-scale spectrogram (num frequency bins, num time frames)
        Returns:
            spectrogram_image: (observation_height, observation_width) image (uint8)
        """
        if self.config.spectrogram_normalization == "minmax":
            min_val = np.min(spec_db)
            max_val = np.max(spec_db)
        else:
            finite_values = spec_db[np.isfinite(spec_db)]
            if finite_values.size == 0:
                spec_db = np.zeros_like(spec_db, dtype=np.float32)
                finite_values = spec_db.ravel()
            min_val = np.percentile(finite_values, 2.0)
            max_val = np.percentile(finite_values, 98.0)
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
        """Detect the top-k peaks in 2D data (returned in real-space coordinates)"""
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

    def estimate_music(
        self,
        mic_signals: np.ndarray,
        array_index: int,
    ) -> pra.doa.MUSIC:
        z = np.asarray(
            [
                stft(
                    mic_signals[ch],
                    fs=self.config.fs,
                    nperseg=self.config.nfft,
                    noverlap=self.config.nfft // 2,
                )[2]
                for ch in range(mic_signals.shape[0])
            ]
        )
        mic_positions = self._generate_circular_array(
            self.mic_positions[array_index],
            self.config.mics_per_array,
            self.config.mic_radius,
        )
        azimuth = None
        if self.config.doa_azimuth_grid_size is not None:
            azimuth = np.linspace(
                -np.pi,
                np.pi,
                int(self.config.doa_azimuth_grid_size),
                endpoint=False,
            )
        doa = pra.doa.MUSIC(
            mic_positions,
            fs=self.config.fs,
            nfft=self.config.nfft,
            c=self.config.sound_speed,
            num_src=self.config.music_num_src,
            azimuth=azimuth,
            frequency_normalization=self.config.doa_frequency_normalization,
        )
        doa.locate_sources(z)
        return doa

    def build_combined_sound_map(
        self,
        sound_maps: List[np.ndarray],
        power: Optional[float] = None,
        gaussian_sigma: Optional[float] = None,
    ) -> np.ndarray:
        power = self.config.combined_map_power if power is None else power
        gaussian_sigma = (
            self.config.gaussian_sigma if gaussian_sigma is None else gaussian_sigma
        )
        combined = np.sum(
            [np.power(sound_map, power, dtype=np.float32) for sound_map in sound_maps],
            axis=0,
        )
        if gaussian_sigma and gaussian_sigma > 0.0:
            combined = gaussian_filter(combined, sigma=gaussian_sigma)
        return combined

    def compute_top_peaks(
        self,
        combined_sound_map: np.ndarray,
        topk: Optional[int] = None,
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
        selection_mode: Optional[str] = None,
    ) -> List[Tuple[float, float]]:
        topk = self.config.num_peaks if topk is None else topk
        selection_mode = self.config.peak_selection_mode if selection_mode is None else selection_mode
        if selection_mode == "local_max":
            return self._find_top_k_peaks(combined_sound_map, topk)

        flat = combined_sound_map.ravel()
        if flat.size == 0 or topk <= 0:
            return []
        topk = min(topk, flat.size)
        indices = np.argpartition(flat, -topk)[-topk:]
        indices = indices[np.argsort(flat[indices])[::-1]]
        if x_coords is None or y_coords is None:
            row_coords = np.arange(combined_sound_map.shape[0], dtype=np.float32)
            col_coords = np.arange(combined_sound_map.shape[1], dtype=np.float32)
        else:
            row_coords = np.asarray(y_coords, dtype=np.float32)
            col_coords = np.asarray(x_coords, dtype=np.float32)
        return [
            (float(col_coords[col]), float(row_coords[row]))
            for row, col in (np.unravel_index(int(flat_index), combined_sound_map.shape) for flat_index in indices)
        ]

    def reconstruct_primary_peak(
        self,
        mic_signals_list: List[np.ndarray],
        top_peaks: List[Tuple[float, float]],
        array_relative_positions: Optional[List[np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        if not top_peaks:
            return None

        peak_x, peak_y = top_peaks[0]
        beamformed_signals = []
        for array_index, mic_signals in enumerate(mic_signals_list):
            theta_deg = self._pixel_to_azimuth(
                peak_x,
                peak_y,
                self.mic_positions[array_index],
            )
            if array_relative_positions is None:
                mic_array_abs = self._generate_circular_array(
                    self.mic_positions[array_index],
                    self.config.mics_per_array,
                    self.config.mic_radius,
                ).T
                relative_positions = mic_array_abs - np.mean(mic_array_abs, axis=0, keepdims=True)
            else:
                relative_positions = array_relative_positions[array_index]
            beamformed_signals.append(
                self._ds_beamform(
                    mic_signals,
                    relative_positions,
                    theta_deg,
                )
            )

        final_wav, _, _ = self._perform_spotforming_nmf(
            beamformed_signals=beamformed_signals,
            fs=self.config.fs,
            n_components=self.config.nmf_components,
            threshold=self.config.nmf_threshold,
            nfft=self.config.nfft,
            noverlap=self.config.nfft // 2,
        )
        return final_wav[: self.required_length]

    def create_spectrogram_image_from_audio(self, audio: np.ndarray) -> np.ndarray:
        freqs, _, zxx = stft(
            audio,
            fs=self.config.fs,
            nperseg=self.config.nfft,
            noverlap=self.config.nfft // 2,
        )
        min_hz = max(0.0, float(self.config.spectrogram_display_min_hz))
        max_hz = self.config.spectrogram_display_max_hz
        if max_hz is None:
            max_hz = float(freqs[-1]) if freqs.size else self.config.fs / 2.0
        max_hz = min(float(max_hz), float(freqs[-1]) if freqs.size else self.config.fs / 2.0)
        band_mask = (freqs >= min_hz) & (freqs <= max_hz)
        if np.any(band_mask):
            zxx = zxx[band_mask]
        image = self._convert_spectrogram_to_image(10 * np.log10(np.abs(zxx) ** 2 + 1e-10))
        return self._pad_to_3ch(image)

    def build_sound_observation_from_mic_signals(
        self,
        mic_signals_list: List[np.ndarray],
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
        azimuth_offset_deg: Optional[float] = None,
        distance_floor_m: Optional[float] = None,
        distance_decay_exponent: Optional[float] = None,
        combined_map_power: Optional[float] = None,
        combined_map_gaussian_sigma: Optional[float] = None,
        peak_selection_mode: Optional[str] = None,
        array_relative_positions: Optional[List[np.ndarray]] = None,
        create_spectrogram: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        music_results = [
            self.estimate_music(mic_signals, idx)
            for idx, mic_signals in enumerate(mic_signals_list)
        ]
        sound_maps = [
            self._generate_soundmap_from_doa(
                music_results[idx],
                self.mic_positions[idx],
                x_coords=x_coords,
                y_coords=y_coords,
                azimuth_offset_deg=azimuth_offset_deg,
                distance_floor_m=distance_floor_m,
                distance_decay_exponent=distance_decay_exponent,
            )
            for idx in range(len(music_results))
        ]

        if not sound_maps:
            zero_shape = (
                self.config.observation_height,
                self.config.observation_width,
                3,
            )
            zeros = np.zeros(zero_shape, dtype=np.uint8)
            return zeros, zeros, zeros

        stacked_maps = np.stack(sound_maps, axis=2)
        per_array_uint8 = self._normalize_to_uint8(stacked_maps)
        sound0 = self._split_channels(per_array_uint8, 0, 3)
        sound1 = self._split_channels(per_array_uint8, 3, 6)

        if not create_spectrogram:
            return sound0, sound1, np.zeros_like(sound0)

        combined_map = self.build_combined_sound_map(
            sound_maps,
            power=combined_map_power,
            gaussian_sigma=combined_map_gaussian_sigma,
        )
        top_peaks = self.compute_top_peaks(
            combined_map,
            x_coords=x_coords,
            y_coords=y_coords,
            selection_mode=peak_selection_mode,
        )
        reconstructed = self.reconstruct_primary_peak(
            mic_signals_list,
            top_peaks,
            array_relative_positions=array_relative_positions,
        )
        if reconstructed is None:
            spec = np.zeros_like(sound0)
        else:
            spec = self.create_spectrogram_image_from_audio(reconstructed)
        return sound0, sound1, spec
    
    def _pixel_to_azimuth(
        self,
        x: float, # real-space coordinate x
        y: float, # real-space coordinate y
        mic_center: List[float]
    ) -> float:
        """Compute the azimuth (degrees) from real-space coordinates (matching test.py's approach)"""
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
        """Delay-and-sum beamforming"""
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
        """Generate a unit vector from azimuth and elevation"""
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
        """Compute the steering vector"""
        # Geometric delay for each mic
        taus = -(mic_positions @ doa_unitvec) / self.config.sound_speed
        # Phase term
        phase = -2j * np.pi * freqs[:, None] * taus[None, :]
        return np.exp(phase)
    
    def _split_channels(
        self,
        array: np.ndarray,
        start_ch: int,
        end_ch: int
    ) -> np.ndarray:
        """
        Extract the specified channel range from the array and zero-pad to 3 channels
        Args:
            array: (height, width, channels) image
            start_ch: start channel
            end_ch: end channel (exclusive)
        Returns:
            result: (height, width, 3) image (missing channels zero-padded)
        """
        height, width, total_channels = array.shape
        result = np.zeros((height, width, 3), dtype=array.dtype)
        available_channels = min(end_ch, total_channels) - start_ch
        available_channels = max(0, available_channels)  # Prevent negative values
        for i in range(min(available_channels, 3)):
            if start_ch + i < total_channels:
                result[:, :, i] = array[:, :, start_ch + i]
        return result
    
    def _pad_to_3ch(self, array: np.ndarray) -> np.ndarray:
        """
        Zero-pad a 2D array or 1/2-channel array to 3 channels
        Args:
            array: (height, width) or (height, width, 1) or (height, width, 2)
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
