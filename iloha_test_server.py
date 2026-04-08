#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unity Communicator用のダミー通信ノード
WebSocket接続、UDP関節角度受信をシミュレート
リアルタイムプロット機能付き
"""

import asyncio
import websockets
import socket
import json
import struct
import time
import threading
from typing import Optional
import numpy as np
from collections import deque

# matplotlibの設定（Qt5バックエンドを使用）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DummyCommunicationNode:
    def __init__(self):
        # 通信設定
        self.websocket_port = 8080
        self.unity_joint_port: Optional[int] = None
        
        # 通信状態
        self.is_connected = False
        self.is_receiving_joints = False
        
        # スレッド制御
        self.joint_thread: Optional[threading.Thread] = None
        self.websocket_thread: Optional[threading.Thread] = None
        self.stop_threads = False
        
        # データ受信周期
        self.receive_interval = 0.01
        
        # プロット設定（ユーザーが変更可能）
        self.plot_joint_indices = [7]  # プロットする関節のインデックスリスト
        
        # プロット用データキュー（スレッドセーフ）
        # (timestamp, joint_angles)のタプルを保存
        max_data_points = 1000  # 最大データ点数
        self.data_queue = deque(maxlen=max_data_points)
        
        # プロット用の変数
        self.fig = None
        self.ax = None
        self.scatters = []
        self.plot_output_path = "iloha_joint_plot.png"
        self._threads_stopped = False
        self._plot_saved = False

    async def websocket_handler(self, websocket):
        """WebSocket接続ハンドラー"""
        print(f"WebSocket接続: {websocket.remote_address}")
        try:
            print("Unity側からのメッセージを待機中...")
            message = await websocket.recv()
            data = json.loads(message)
            self.unity_joint_port = data.get('joint_send_port')
            self.is_connected = True
            self.start_udp_communication()
            await asyncio.sleep(0.5)
            # 接続確認レスポンスを送信
            response = {"status": "connected", "message": "接続情報受信完了"}
            await websocket.send(json.dumps(response))
            await self.handle_websocket_messages(websocket)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket接続がクライアント側から閉じられました")
        except Exception as e:
            print(f"WebSocketエラー: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("WebSocket接続終了")
            self.cleanup()

    async def handle_websocket_messages(self, websocket):
        """WebSocketメッセージの継続受信処理"""
        try:
            async for message in websocket:
                print(f"WebSocketメッセージ受信: {message}")
                try:
                    data = json.loads(message)
                    command = data.get('command')
                    if command == 'reset_robot':
                        print("ロボットリセット要求を受信しました")
                        await self.handle_reset_request(websocket)
                    elif command == 'save_data':
                        print("データ保存要求を受信しました")
                    else:
                        print(f"不明なコマンド: {command}")
                except json.JSONDecodeError:
                    print(f"JSON解析エラー: {message}")
                except Exception as e:
                    print(f"メッセージ処理エラー: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket接続が閉じられました（メッセージ受信中）")
        except Exception as e:
            print(f"WebSocketメッセージ受信エラー: {e}")

    async def handle_reset_request(self, websocket):
        """リセット要求処理"""
        try:
            await asyncio.sleep(2.0)  # 2秒間のリセット処理をシミュレート
            print("ロボットリセット処理が完了しました")
            response = {"status": "reset_complete", "message": "ロボットリセットが完了しました"}
            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"リセット処理エラー: {e}")
            error_response = {"status": "reset_error", "message": f"リセット処理でエラーが発生しました: {e}"}
            try:
                await websocket.send(json.dumps(error_response))
            except:
                print("エラー通知の送信に失敗しました")

    def start_udp_communication(self):
        """UDP通信スレッドを開始"""
        if self.unity_joint_port:
            self.stop_threads = False
            self.is_receiving_joints = True
            self.joint_thread = threading.Thread(target=self.joint_receiver_thread)
            self.joint_thread.daemon = True
            self.joint_thread.start()
            print("UDP通信スレッドを開始しました")

    def joint_receiver_thread(self):
        """関節角度受信スレッド"""
        print(f"関節角度受信開始: ポート{self.unity_joint_port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', self.unity_joint_port))
        sock.settimeout(1.0)
        sock.setblocking(False)

        start_time = time.time()

        try:
            while self.is_receiving_joints and not self.stop_threads:
                latest_data_packet = None
                addr = None
                try:
                    # --- 修正点2： バッファが空になるまでループで読み出す ---
                    while True:
                        # バッファからデータを読み出す
                        # 成功するたびに latest_data_packet が上書きされていく
                        latest_data_packet, addr = sock.recvfrom(1024)
                except (BlockingIOError, socket.error):
                    # BlockingIOError (Windows) または socket.error (Linux) は、
                    # 「バッファが空で、もう読み出すデータがない」ことを示す
                    # これはエラーではなく、正常な動作
                    pass
                if latest_data_packet and len(latest_data_packet) >= 57:
                    mode = latest_data_packet[0]
                    joint_angles = []
                    for i in range(14):
                        offset = 1 + i * 4
                        angle = struct.unpack('<f', latest_data_packet[offset:offset+4])[0]
                        joint_angles.append(np.degrees(angle))

                    timestamp = time.time()
                    self.data_queue.append((timestamp, joint_angles))
                time.sleep(max(0.0, self.receive_interval - (time.time() - start_time)))
                start_time = time.time()
        except Exception as e:
            print(f"関節角度受信スレッドエラー: {e}")
        finally:
            sock.close()
            print("関節角度受信終了")

    def cleanup(self):
        """リソースクリーンアップ"""
        if not self._threads_stopped:
            self.is_connected = False
            self.is_receiving_joints = False
            self.stop_threads = True

            if self.joint_thread and self.joint_thread.is_alive():
                self.joint_thread.join(timeout=2)
            self.joint_thread = None
            self._threads_stopped = True

        # WebSocket側のcleanupが先に呼ばれてfig未初期化だった場合でも、
        # メインスレッド側のcleanupで再試行して画像保存できるようにする
        if not self._plot_saved:
            try:
                if self.fig is None or self.ax is None:
                    print("プロット未初期化のため、画像保存を次のcleanupで再試行します")
                    return
                if threading.current_thread() is not threading.main_thread():
                    print("メインスレッドで画像保存するため、次のcleanupで再試行します")
                    return
                self.save_plot()
                self._plot_saved = True
            except Exception as e:
                print(f"プロット画像保存エラー: {e}")
                import traceback
                traceback.print_exc()

        print("クリーンアップ完了")

    async def start_server(self):
        """WebSocketサーバーを開始"""
        print(f"WebSocketサーバーを開始: ポート{self.websocket_port}")
        try:
            async with websockets.serve(
                self.websocket_handler,
                "0.0.0.0",
                self.websocket_port
            ):
                await asyncio.Future()
        except KeyboardInterrupt:
            print("\nサーバー停止中...")
        except Exception as e:
            print(f"サーバーエラー: {e}")
        finally:
            self.cleanup()

    def start_server_thread(self):
        """WebSocketサーバーを別スレッドで開始"""
        def run_server():
            asyncio.run(self.start_server())
        
        self.websocket_thread = threading.Thread(target=run_server)
        self.websocket_thread.daemon = True
        self.websocket_thread.start()
        print("WebSocketサーバースレッドを開始しました")

    def setup_plot(self):
        """プロットの初期設定"""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Angle (degree)')
        self.ax.set_title('Real-time Joint Angle Plot')
        self.ax.grid(True)
        
        # 各関節に対してscatterオブジェクトを作成
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        self.scatters = []
        for i, joint_idx in enumerate(self.plot_joint_indices):
            color = colors[i % len(colors)]
            scatter = self.ax.scatter([], [], c=color, label=f'Joint {joint_idx}', s=20, alpha=0.6)
            self.scatters.append(scatter)
        
        self.ax.legend()

    def update_plot(self, frame):
        """プロット更新関数（FuncAnimationから呼ばれる）"""
        if len(self.data_queue) == 0:
            return self.scatters

        # 受信開始時刻を基準に全期間をプロット
        first_timestamp = self.data_queue[0][0]
        
        # 各関節のデータを準備
        plot_data = {idx: {'times': [], 'angles': []} for idx in self.plot_joint_indices}
        
        # キューからデータを取得
        for timestamp, joint_angles in self.data_queue:
            elapsed_time = timestamp - first_timestamp
            for joint_idx in self.plot_joint_indices:
                if joint_idx < len(joint_angles):
                    plot_data[joint_idx]['times'].append(elapsed_time)
                    plot_data[joint_idx]['angles'].append(joint_angles[joint_idx])
        
        # 各scatterを更新
        for i, joint_idx in enumerate(self.plot_joint_indices):
            times = plot_data[joint_idx]['times']
            angles = plot_data[joint_idx]['angles']
            if len(times) > 0:
                # scatterの位置を更新
                offsets = np.column_stack([times, angles])
                self.scatters[i].set_offsets(offsets)
            else:
                # データがない場合は空にする
                self.scatters[i].set_offsets(np.empty((0, 2)))
        
        # 軸範囲を更新
        if any(len(plot_data[idx]['times']) > 0 for idx in self.plot_joint_indices):
            max_elapsed = max(
                max(plot_data[idx]['times']) for idx in self.plot_joint_indices if plot_data[idx]['times']
            )
            self.ax.set_xlim(0, max(1.0, max_elapsed))
            
            # Y軸の範囲を自動調整
            all_angles = []
            for joint_idx in self.plot_joint_indices:
                all_angles.extend(plot_data[joint_idx]['angles'])
            
            if all_angles:
                min_angle = min(all_angles)
                max_angle = max(all_angles)
                margin = (max_angle - min_angle) * 0.1 if max_angle != min_angle else 10
                self.ax.set_ylim(min_angle - margin, max_angle + margin)
        
        return self.scatters

    def start_plot(self):
        """プロットを保存できるように初期化"""
        self.setup_plot()

    def save_plot(self):
        """現在の受信データをもとにプロット画像を保存"""
        if self.fig is None or self.ax is None:
            return

        self.update_plot(frame=0)
        self.fig.tight_layout()
        self.fig.savefig(self.plot_output_path, dpi=150)
        print(f"プロット画像を保存しました: {self.plot_output_path}")

    def run(self):
        """メインの実行関数"""
        print("=" * 60)
        print("UDP Test Server Starting")
        print(f"Plot target joints: {self.plot_joint_indices}")
        print("Plot range: full capture duration")
        print("=" * 60)
        
        # WebSocketサーバーを別スレッドで開始
        self.start_server_thread()
        
        # matplotlibをメインスレッドで実行
        try:
            self.start_plot()
            while not self.stop_threads:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping server...")
        finally:
            self.cleanup()
            plt.close('all')

if __name__ == "__main__":
    node = DummyCommunicationNode()
    
    # Plot settings (change as needed)
    node.plot_joint_indices = [10]  # Joint indices to plot
    node.run()
