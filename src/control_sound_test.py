#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sounddevice as sd
import pulsectl
import argparse
import time

class SoundController:
    def __init__(self, left_volume=50, right_volume=50, device_id=None):
        # Pulseaudioのクライアントを初期化
        self.pulse = pulsectl.Pulse('sound-controller')
        self.sample_rate = 44100  # サンプリングレート
        self.left_volume = left_volume  # 左チャンネルの音量
        self.right_volume = right_volume  # 右チャンネルの音量
        self.device_id = device_id  # 指定されたデバイスID
        self.is_playing = False   # 再生状態
        self.stream = None        # 音声ストリーム

    def list_playback_devices(self):
        devices = sd.query_devices()
        print("再生可能なデバイス:")
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                print(f"デバイス {i}: {device['name']}, サンプリングレート: {device['default_samplerate']} Hz")

    def validate_device(self):
        """指定されたデバイスIDが有効か確認"""
        if self.device_id is not None:
            devices = sd.query_devices()
            if self.device_id < 0 or self.device_id >= len(devices):
                raise ValueError(f"無効なデバイスID: {self.device_id}。デバイスIDは0から{len(devices)-1}の範囲で指定してください。")
            if devices[self.device_id]['max_output_channels'] == 0:
                raise ValueError(f"デバイス {self.device_id} ({devices[self.device_id]['name']}) は再生をサポートしていません。")

    def generate_gaussian_noise(self, duration):
        """
        ガウスノイズを生成する
        
        Args:
            duration (float): 生成する音声の長さ（秒）
        
        Returns:
            numpy.ndarray: ガウスノイズデータ（stereo）
        """
        # ガウスノイズを生成（平均0、標準偏差0.1）
        samples = np.random.normal(0, 0.1, int(self.sample_rate * duration)).astype(np.float32)
        
        # ステレオに変換（左右チャンネル）
        stereo_samples = np.column_stack((samples, samples))
        
        return stereo_samples

    def set_volume(self, left=None, right=None):
        """
        左右の音量を設定する（0-100の範囲）
        
        Args:
            left (int, optional): 左チャンネルの音量（0-100）
            right (int, optional): 右チャンネルの音量（0-100）
        """
        if left is not None:
            self.left_volume = max(0, min(100, left))
        
        if right is not None:
            self.right_volume = max(0, min(100, right))
        
        print(f"音量設定: 左={self.left_volume}%, 右={self.right_volume}%")
        
        # すべてのシンクを取得
        sink_list = self.pulse.sink_list()
        
        if sink_list:
            # デフォルトのシンク（通常はメインの出力デバイス）を使用
            default_sink = self.pulse.get_sink_by_name(self.pulse.server_info().default_sink_name)
            
            # チャンネルマップを取得
            channel_map = default_sink.channel_list
            channel_count = len(channel_map)
            
            # 音量調整の配列を作成（通常はステレオで[左, 右]）
            if channel_count >= 2:
                # 0.0-1.0の範囲に正規化
                volumes = [self.left_volume / 100.0, self.right_volume / 100.0]
                
                # 追加のチャンネルがある場合は右チャンネルの値で埋める
                while len(volumes) < channel_count:
                    volumes.append(self.right_volume / 100.0)
                
                # 音量を設定
                self.pulse.volume_set_all_chans(default_sink, volumes[0])
                
                # チャンネル個別に音量を調整
                for i, vol in enumerate(volumes):
                    try:
                        # チャンネル別の音量設定（左右独立）
                        cvol = default_sink.volume
                        cvol.values[i] = vol
                        self.pulse.volume_set(default_sink, cvol)
                    except Exception as e:
                        print(f"チャンネル{i}の音量設定に失敗: {e}")

    def play_noise(self, duration=5.0):
        """
        ガウスノイズを再生する
        
        Args:
            duration (float, optional): 再生時間（秒）, デフォルト5秒
        """
        # デバイスIDの検証
        self.validate_device()
        
        # ノイズを生成
        noise = self.generate_gaussian_noise(duration)
        
        # 音量を設定
        self.set_volume(self.left_volume, self.right_volume)
        
        print(f"ガウスノイズを再生します（{duration}秒, デバイスID: {self.device_id if self.device_id is not None else 'デフォルト'}）")
        
        # 前回のストリームが残っていれば停止
        if self.stream is not None and self.is_playing:
            self.stop_noise()
            time.sleep(0.1)  # 完全に停止するまで少し待つ
        
        # 再生（デバイスIDを指定）
        self.is_playing = True
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            dtype='float32',
            device=self.device_id
        )
        self.stream.start()
        self.stream.write(noise)
        
        # 再生終了後
        self.is_playing = False
        self.stream.stop()
        self.stream.close()
        self.stream = None
        
        print("再生終了")

    def stop_noise(self):
        """
        再生中のノイズを停止する
        """
        if self.stream is not None and self.is_playing:
            print("ノイズを停止します")
            self.is_playing = False
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def close(self):
        """
        リソースを解放する
        """
        self.stop_noise()
        self.pulse.close()


def main():
    parser = argparse.ArgumentParser(description='ガウスノイズを生成して左右の音量を制御するプログラム')
    parser.add_argument('--left', type=int, default=50, help='左チャンネルの音量（0-100）')
    parser.add_argument('--right', type=int, default=50, help='右チャンネルの音量（0-100）')
    parser.add_argument('--duration', type=float, default=5.0, help='再生時間（秒）')
    parser.add_argument('--device', type=int, default=None, help='再生デバイスのID（list_playback_devicesで確認）')
    args = parser.parse_args()

    controller = SoundController(left_volume=args.left, right_volume=args.right, device_id=args.device)
    controller.list_playback_devices()
    
    try:
        # ノイズ再生
        controller.play_noise(args.duration)
        
    except KeyboardInterrupt:
        print("\n中断されました")
    except ValueError as e:
        print(f"エラー: {e}")
    finally:
        controller.close()


if __name__ == "__main__":
    main()