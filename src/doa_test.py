# ライブラリのインポート
# *************************************************************************************************
import pyaudio
import numpy as np
import pyroomacoustics as pra
import time

# *************************************************************************************************
# 定数の定義
# *************************************************************************************************
# device = "ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:1,0)" # マイク4つのマイクロフォンアレイを使用
device = "TAMAGO-03: USB Audio (hw:4,0)" # 卵型のマイクロフォンアレイを使用
form_1 = pyaudio.paInt16
chans = 8
samp_rate = 16000
chunk = 8192
record_secs = 20
c = 343.
nfft = 256
freq_range = [300, 3500]
# *************************************************************************************************
# クラスの定義
# *************************************************************************************************
class DoaTest:
    def __init__(self):
        p = pyaudio.PyAudio()
        index = None
        for i in range(p.get_device_count()):
            if p.get_device_info_by_index(i)['name'] == device: # デバイスが見つかったらindexを取得
                print(f"デバイス発見 [ID:{i}], {p.get_device_info_by_index(i)['name']}")
                index = i
                break
            else:
                print(p.get_device_info_by_index(i)['name'])
            if i == p.get_device_count() - 1: # デバイスが見つからなかったら終了
                print("デバイスが見つかりません")
                exit()
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=form_1, rate=samp_rate, channels=chans, input_device_index=index, input=True, frames_per_buffer=chunk)
        print("録音開始")
        radius = None
        if device == "ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:1,0)":
            radius = 0.065
        elif device == "TAMAGO-03: USB Audio (hw:4,0)":
            radius = 0.065
        self.mic_locs = pra.circular_2D_array(center=[0,0], M=chans, phi0=0, radius=radius)

    def timer_callback(self):
        start_time = time.time()
        data = self.stream.read(chunk)
        data = np.frombuffer(data, dtype='int16')
        data = data.reshape(-1, chans)
        X = pra.transform.stft.analysis(data, nfft, nfft // 2)
        X = X.transpose([2, 1, 0])
        doa = pra.doa.algorithms['MUSIC'](self.mic_locs, samp_rate, nfft, c=c, num_src=1, max_four=4)
        doa.locate_sources(X, freq_range=freq_range)
        spatial_resp = doa.grid.values
        # 0-1に正規化
        spatial_resp = (spatial_resp - spatial_resp.min())/(spatial_resp.max() - spatial_resp.min())
        print("経過時間: {}".format(time.time() - start_time))
        # spatial_respはnumpy配列
        return spatial_resp

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        print("録音終了")
# *************************************************************************************************
# メイン関数
# *************************************************************************************************
def main():
    doa = DoaTest()
    try:
        for _ in range(int(samp_rate * record_secs / chunk)):
            spatial_resp = doa.timer_callback()
            # 必要に応じてspatial_respを処理
            time.sleep(0.1)
    finally:
        doa.close()

if __name__ == '__main__':
    main()
