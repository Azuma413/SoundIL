import pyroomacoustics as pra
import numpy as np
from scipy.signal import stft, istft
from scipy.io import wavfile
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft, rfftfreq
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import NMF
from pydub import AudioSegment
import os

num_microphones = 8 # 1つのマイクロフォンアレイに含まれるマイクの数
mic_num = 6 # マイクロフォンアレイの数
sound_path = "sounds/1.wav"
music_num_src = 3 # MUSIC法で仮定する音源数
k = 1 # ビームフォーミングで利用するピーク点の数
source_loc = [5.5, 5.5, 0.3]
processing_time = 1.0 # 秒

def doa(mic_signals, mic_loc, fs, nfft=512, noverlap=512//2, c=343.0, num_src=3):
    stft_results = []
    for i in range(mic_signals.shape[0]):
        f, t, Zxx = stft(mic_signals[i,:], fs=fs, nperseg=nfft, noverlap=noverlap)
        stft_results.append(Zxx)
    Z=np.array(stft_results) # shape: (マイク数, 周波数ビン数, タイムフレーム数)
    music = pra.doa.MUSIC(mic_loc, fs=fs, nfft=nfft, c=c, num_src=num_src)
    music.locate_sources(Z)
    return music

# def circular_array_positions(num_mics: int, radius_m: float) -> np.ndarray:
#     angles = np.linspace(0, 2*np.pi, num_mics, endpoint=False)
#     x = radius_m * np.cos(angles)
#     y = radius_m * np.sin(angles)
#     z = np.zeros_like(x)
#     return np.stack([x, y, z], axis=1)

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
    freqs = rfftfreq(N, d=1.0/fs)           # (K,)
    u = unit_vec_from_angles(azimuth_deg, elevation_deg)
    A = steering_vector(freqs, mic_positions, u, c=c)   # (K, M)
    Y_beam = np.sum(A.conj() * Y, axis=1)  # (K,)
    if normalize:
        Y_beam /= M
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
    dx=np.array(top_peaks)-np.mean(mic_loc,axis=0)[:2]
    return np.atan2(dx[:,1],dx[:,0])

def perform_spotforming_nmf(
    beamformed_signals, # ある1つの音源に対する全アレイのビームフォーミング結果 (M, N_samples)
    fs,
    n_components,
    threshold=0.01,
    nfft=512,
    noverlap=256
):
    """
    論文のセクションIII-B [cite: 121] に記載のNMFベースのスポットフォーミングを実行する
    """
    M = len(beamformed_signals)
    all_amp_specs = []
    all_phases = []
    # 1. 全アレイのSTFTと振幅スペクトログラムを取得
    for wav in beamformed_signals:
        f_stft, t_stft, Zxx = stft(wav, fs=fs, nperseg=nfft, noverlap=noverlap)
        # 論文に従い「振幅スペクトログラム」を使用 (パワースペクトログラム np.abs(Zxx)**2 でも可)
        amp_spec = np.abs(Zxx) 
        phase = np.angle(Zxx)
        all_amp_specs.append(amp_spec)
        all_phases.append(phase)
    # 2. 振幅スペクトログラムを時間方向に連結 
    # これが Y_i = [Y_i^(1), ..., Y_i^(M)] に相当
    concatenated_spec = np.concatenate(all_amp_specs, axis=1)
    F, T_total = concatenated_spec.shape
    T_frame = all_amp_specs[0].shape[1]
    # 3. 連結したスペクトログラムに対し、NMFを1回だけ実行 [cite: 124]
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=500, tol=1e-4)
    # W が共通基底 V_i (F, K) 
    W = model.fit_transform(concatenated_spec)
    # H が連結されたアクティベーション H_i (K, T_total) [cite: 124]
    H = model.components_
    # 4. 連結された H を、アレイ毎の H^(m) に分割 [cite: 129]
    # split_H は [H^(1), H^(2), ..., H^(M)] のリストになる
    split_H = np.array_split(H, M, axis=1)
    # 5. バイナリマスクを作成 [cite: 144]
    # (M, K, T_frame) の形状にスタックして、アレイ間で最小値を取る
    stacked_H = np.stack(split_H, axis=0)
    # min_activations の形状は (K, T_frame)
    min_activations = np.min(stacked_H, axis=0)
    # しきい値を超えたら 1, それ以外は 0
    binary_mask = (min_activations > threshold).astype(float) # これが B_i
    # 6. 各アレイのスペクトログラムを再構成 [cite: 153]
    reconstructed_wavs = []
    for m in range(M):
        H_m = split_H[m]      # H_i^(m)
        phase_m = all_phases[m] # 位相
        # S_hat^(m) = V_i * (H_i^(m) element-wise* B_i) [cite: 153]
        reconstructed_amp_spec = W @ (H_m * binary_mask)
        # 位相を戻して複素スペクトログラムを復元
        reconstructed_complex_spec = reconstructed_amp_spec * np.exp(1j * phase_m)
        # 逆STFTで音声波形に戻す
        _, wav_m = istft(reconstructed_complex_spec, fs=fs, nperseg=nfft, noverlap=noverlap)
        reconstructed_wavs.append(wav_m)
    # 7. 全アレイの再構成音を平均化 [cite: 155]
    # (論文[cite: 155]では相関関数で時間シフト補正推奨だが、ここでは簡略化のため単純平均)
    # 最大長に合わせるためゼロパディング
    max_len = max(len(w) for w in reconstructed_wavs)
    padded_wavs = []
    for w in reconstructed_wavs:
        if len(w) < max_len:
            padded_wavs.append(np.pad(w, (0, max_len - len(w))))
        else:
            padded_wavs.append(w)
    final_wav = np.mean(np.stack(padded_wavs, axis=0), axis=0)
    return final_wav, f_stft, t_stft

room_dim = [10, 10, 3]
room = pra.ShoeBox(room_dim, fs=16000, max_order=3)
room.add_source(source_loc)
mic_loc = []
for i in range(mic_num):
    theta = np.pi * 2 / mic_num * i
    mic_loc.append(get_micarray_pos(5 + 2*np.cos(theta), 5 + 2*np.sin(theta)))

for i in range(mic_num):
    room.add_microphone_array(mic_loc[i])

if not sound_path:
    signal = np.random.randn(room.fs)
else:
    format = sound_path.split(".")[-1]
    sound = AudioSegment.from_file(sound_path, format=format)
    sound = sound.set_frame_rate(room.fs).set_channels(1)
    signal = np.array(sound.get_array_of_samples())

signal = signal[:int(room.fs * processing_time)] # signalの最初のroom.fs * processing_time秒分だけを使う
room.sources[0].signal = signal
room.compute_rir()
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
for mic_array_loc, signal in zip(mic_loc, signal_list):
    pos = mic_array_loc.T # (3, N_mic) -> (N_mic, 3) が正しい
    cm = np.mean(pos, axis=0) # (3,)
    pos_centered = pos - cm # (N_mic, 3)
    theta_list = get_peek_direction_from_mic(top_peaks, mic_array_loc.T)
    wav=[]
    for theta in theta_list:
        wav.append(ds_beamform_time(signal.T, fs=room.fs, mic_positions=pos_centered, azimuth_deg=np.rad2deg(theta)))
    beamform_wav.append(wav)
beamform_wav_per_peak = np.array(beamform_wav).transpose((1, 0, 2))
actual_num_peaks = beamform_wav_per_peak.shape[0]
print(f"{actual_num_peaks}個のピーク（音源）を推定しました。")
final_separated_signals = []
for i in range(actual_num_peaks):
    print(f"ピーク {i+1}/{actual_num_peaks} のスポットフォーミングを実行中...")
    signals_for_peak_i = beamform_wav_per_peak[i] # (M_arrays, N_samples)
    final_wav, f_stft, t_stft = perform_spotforming_nmf(
        signals_for_peak_i,
        fs=room.fs,
        n_components=100,
        threshold=1.6e-3,
    )
    final_separated_signals.append(final_wav)
output_dir = "separated_sounds"
os.makedirs(output_dir, exist_ok=True)
print(f"分離した音声を '{output_dir}' フォルダに保存します。")
target_signal = final_separated_signals[0][:int(room.fs * processing_time)]
max_val = np.max(np.abs(target_signal))
if max_val > 0:
    normalized_signal = target_signal / max_val
else:
    normalized_signal = target_signal # 無音の場合はそのまま
signal_int16 = (normalized_signal * 32767).astype(np.int16)
filename = os.path.join(output_dir, f"separated_peak_0.wav")
wavfile.write(filename, room.fs, signal_int16)
print(f"  -> {filename} に保存しました。")
f_final, t_final, Zxx_final = stft(target_signal, fs=room.fs, nperseg=512, noverlap=256)
power_spec_final = np.abs(Zxx_final)**2
plt.figure(figsize=(10, 4))
plt.pcolormesh(t_final, f_final, 10 * np.log10(power_spec_final + 1e-10), shading='gouraud')
plt.title(f'Power Spectrogram of Final Separated Signal (Peak 0/{actual_num_peaks})')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power [dB]')
plt.ylim([0, room.fs / 2])
plot_filename = f"images/spotformed_spectrogram_peak_0.png"
plt.savefig(plot_filename)
print(f"  -> {plot_filename} にスペクトログラムを保存しました。")
plt.close()