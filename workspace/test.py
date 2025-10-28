import pyroomacoustics as pra
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft, rfftfreq
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import NMF
from pydub import AudioSegment

num_microphones = 8 # 1つのマイクロフォンアレイに含まれるマイクの数
mic_num = 6 # マイクロフォンアレイの数
sound_path = "sounds/1.wav"
music_num_src = 3 # MUSIC法で仮定する音源数
k = 2 # ビームフォーミングで利用するピーク点の数
source_loc = [5.5, 5.5, 0.3]

def doa(mic_signals, mic_loc, fs, nfft=512, noverlap=512//2, c=343.0, num_src=3):
    stft_results = []
    for i in range(mic_signals.shape[0]):
        f, t, Zxx = stft(mic_signals[i,:], fs=fs, nperseg=nfft, noverlap=noverlap)
        stft_results.append(Zxx)
    Z=np.array(stft_results) # shape: (マイク数, 周波数ビン数, タイムフレーム数)
    music = pra.doa.MUSIC(mic_loc, fs=fs, nfft=nfft, c=c, num_src=num_src)
    music.locate_sources(Z)
    return music

def circular_array_positions(num_mics: int, radius_m: float) -> np.ndarray:
    angles = np.linspace(0, 2*np.pi, num_mics, endpoint=False)
    x = radius_m * np.cos(angles)
    y = radius_m * np.sin(angles)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)

def unit_vec_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    u = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el)
    ], dtype=float)
    return u / np.linalg.norm(u)

def steering_vector(freqs: np.ndarray,
                    mic_positions: np.ndarray,
                    doa_unitvec: np.ndarray,
                    c: float = 343.0) -> np.ndarray:
    taus = -(mic_positions @ doa_unitvec) / c  # shape: (M,)
    phase = -2j * np.pi * freqs[:, None] * taus[None, :]  # (K, M)
    return np.exp(phase)

def ds_beamform_time(
        y_multi: np.ndarray,
        fs: int,
        mic_positions: np.ndarray,
        azimuth_deg: float,
        elevation_deg: float = 0.0,
        c: float = 343.0,
        normalize: bool = True
    ) -> np.ndarray:
    assert y_multi.ndim == 2, "y_multi must be (N, M)"
    N, M = y_multi.shape
    Y = rfft(y_multi, axis=0)                # (K, M)
    freqs = rfftfreq(N, d=1.0/fs)            # (K,)
    u = unit_vec_from_angles(azimuth_deg, elevation_deg)
    A = steering_vector(freqs, mic_positions, u, c=c)     # (K, M)
    Y_beam = np.sum(A * Y, axis=1)  # (K,)
    if normalize:
        Y_beam /= M  # 振幅の正規化（任意）
    y_beam = irfft(Y_beam, n=N)
    return y_beam

def get_micarray_pos(x,y):
    radius = 0.03
    angles = np.linspace(0, 2 * np.pi, num_microphones, endpoint=False)
    center=[x,y,0.3]
    mic_loc = np.array([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
        [center[2]] * num_microphones
    ])
    return mic_loc

def make_2dmap(N, map_scale, music_list, mic_loc_list):
    X = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for music, loc in zip(music_list, mic_loc_list):
                spec = np.log10(np.mean(music.Pssl, axis=1))
                spec /= np.sum(spec)
                cx = np.mean(loc[0])
                cy = np.mean(loc[1])
                x = i/map_scale
                y = j/map_scale
                theta = np.atan2((y-cy), (x-cx))
                d = np.sqrt((y-cy)**2+(x-cx)**2)
                X[i,j] += spec[int(theta / (2*np.pi / len(spec)))] / (d + 10)
    return X

def find_top_k_peaks_numpy(data, k, map_scale, distance=10):
    is_peak = np.ones_like(data, dtype=bool)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            shifted_data = np.roll(data, shift=(dr, dc), axis=(0, 1))
            comparison = data >= shifted_data
            if dr != 0:
                if dr == 1:
                    comparison[:dr, :] = True # 上端から来た部分
                else:
                    comparison[dr:, :] = True # 下端から来た部分
            if dc != 0:
                if dc == 1:
                    comparison[:, :dc] = True # 左端から来た部分
                else:
                    comparison[:, dc:] = True # 右端から来た部分
            is_peak &= comparison
    peak_rows, peak_cols = np.where(is_peak)
    peak_values = data[peak_rows, peak_cols]
    sorted_indices = np.argsort(peak_values)[::-1]
    top_k_indices = sorted_indices[:k]
    top_k_peaks = [(peak_rows[i]/map_scale, peak_cols[i]/map_scale) for i in top_k_indices]
    return top_k_peaks

def get_peek_direction_from_mic(top_peaks, mic_loc):
    dx=np.array(top_peaks)-np.mean(mic_loc,axis=1)[:2]
    return np.atan2(dx[:,1],dx[:,0])

def select_binary_mask(beamform_wav, index_estimated_src, fs, threshold=0.01, n_components = 3):
    n_estimated_src=len(beamform_wav[0])
    all_mic_results=[]
    for wavs in beamform_wav:
        wav=wavs[index_estimated_src]
        f_stft, t_stft, Zxx = stft(wav, fs=fs)
        power_spectrogram = np.abs(Zxx) ** 2
        model = NMF(n_components=n_components, init='random', random_state=0)
        W = model.fit_transform(power_spectrogram) # 基底 (Basis)
        H = model.components_ # アクティベーション (Activation)
        all_mic_results.append((wav,power_spectrogram, W, H))
    _,_,_,H = all_mic_results[0]
    M=H
    for _,_,_,H in all_mic_results[1:]:
        M=np.minimum(M,H)
    Mask=M>threshold
    return f_stft, t_stft, Mask, all_mic_results

def recons_spec(beamform_wav, n_components, fs=None):
    n_estimated_src=len(beamform_wav[0])
    for i in range(n_estimated_src):
        f_stft, t_stft, Mask, results = select_binary_mask(beamform_wav,n_components=n_components,fs=fs, index_estimated_src=i, threshold=0.01)
        recons = [W @ (Mask * H) for wav, power_spectrogram, W, H in results]
    return recons, f_stft, t_stft

room_dim = [10, 10, 3]
room = pra.ShoeBox(room_dim, fs=16000, max_order=17)
room.add_source(source_loc)
mic_loc = []
for i in range(mic_num):
    theta = np.pi * 2 / mic_num * i
    mic_loc.append(get_micarray_pos(5 + 2*np.cos(theta), 5 + 2*np.sin(theta)))
for i in range(mic_num):
    room.add_microphone_array(mic_loc[i])
room.compute_rir()
if not sound_path:
    signal = np.random.randn(16000)
else:
    format = sound_path.split(".")[-1]
    sound = AudioSegment.from_file(sound_path, format=format)
    signal = np.array(sound.get_array_of_samples())
room.sources[0].signal = signal
room.simulate()
signal_list = []
for i in range(mic_num):
    start_idx = i * num_microphones
    end_idx = (i + 1) * num_microphones
    signals = room.mic_array.signals[start_idx:end_idx]
    signal_list.append(signals)
music_list = []
for i in range(mic_num):
    music = doa(signal_list[i], mic_loc[i], fs=room.fs, num_src=music_num_src)
    music_list.append(music)
mic_loc = np.array(mic_loc)
for i in range(mic_num):
    music = music_list[i]
    loc = mic_loc[i]
    theta = music.azimuth_recon
    x1 = np.mean(loc[0])
    y1 = np.mean(loc[1])
    x2_list = np.mean(loc[0]) + np.cos(theta)*3.0
    y2_list = np.mean(loc[1]) + np.sin(theta)*3.0
map_scale = 10
X = make_2dmap(N=100, map_scale=map_scale, music_list=music_list, mic_loc_list=mic_loc)
X2 = gaussian_filter(X, 1)
top_peaks = find_top_k_peaks_numpy(X2, k, map_scale=map_scale)
beamform_wav=[]
for mic_loc,signal in zip(mic_loc,signal_list):
    pos=mic_loc.T
    cm=np.mean(pos,axis=0)
    pos=pos-cm
    theta_list=get_peek_direction_from_mic(top_peaks, mic_loc)
    wav=[]
    for theta in theta_list:
        wav.append(ds_beamform_time(signal.T, fs=room.fs, mic_positions=pos,azimuth_deg=theta))
    beamform_wav.append(wav)

X_recons,f_stft, t_stft = recons_spec(beamform_wav, n_components=16, fs=room.fs)
X_recons = np.array(X_recons)
recons_spec = np.mean(X_recons, axis=0) # 129, 190
beamform_wav = np.array(beamform_wav)
beamform_spec=[]
for wavs in beamform_wav:
    for wav in wavs:
        f_stft, t_stft, Zxx = stft(wav, fs=room.fs)
        power_spectrogram = np.abs(Zxx) ** 2
        beamform_spec.append(power_spectrogram)
beamform_spec = np.array(beamform_spec)
mean_spec = np.mean(beamform_spec, axis=0) # 129, 190

plt.figure(figsize=(10, 4))
plt.pcolormesh(t_stft, f_stft, 10 * np.log10(mean_spec + 1e-10), shading='gouraud')
plt.title('Power Spectrogram of Beamformed Signal')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power [dB]')
plt.ylim([0, room.fs / 2])  # サンプリング周波数の半分までを表示
# 画像を保存
plt.savefig("images/beamformed_spectrogram.png")