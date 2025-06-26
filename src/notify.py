#!/usr/bin/env python3
"""
コマンド実行完了時の通知システム
Discord Webhook経由で通知を送信
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

try:
    import requests
except ImportError:
    print("警告: requestsがインストールされていません。通知機能が無効になります。")
    requests = None


class NotificationSystem:
    """通知システムクラス"""
    
    def __init__(self):
        self.webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        self.enabled = os.getenv('NOTIFICATIONS_ENABLED', 'true').lower() == 'true'
        self.hostname = os.getenv('HOSTNAME', 'unknown')
        
    def send_discord_message(self, message: str, color: int = 0x00ff00) -> bool:
        """Discord Webhookにメッセージを送信"""
        if not self.enabled or not self.webhook_url or not requests:
            return False
            
        embed = {
            "embeds": [{
                "description": message,
                "color": color,  # デフォルト: 緑色
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "footer": {
                    "text": f"ホスト: {self.hostname}"
                }
            }]
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=embed,
                timeout=10
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Discord通知の送信に失敗: {e}")
            return False
    
    def notify_start(self, script_name: str, additional_info: str = "") -> None:
        """実行開始通知"""
        message = f"🚀 **[sound_dp]** {script_name} 開始\n"
        message += f"⏰ 開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        if additional_info:
            message += f"📊 {additional_info}\n"
        
        self.send_discord_message(message, color=0x0099ff)  # 青色
        
    def notify_success(self, script_name: str, duration: Optional[float] = None, additional_info: str = "") -> None:
        """成功完了通知"""
        message = f"✅ **[sound_dp]** {script_name} 完了！\n"
        message += f"⏰ 完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if duration:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            message += f"📈 実行時間: {hours}時間{minutes}分{seconds}秒\n"
            
        if additional_info:
            message += f"📋 {additional_info}\n"
            
        self.send_discord_message(message, color=0x00ff00)  # 緑色
        
    def notify_error(self, script_name: str, error_message: str = "", duration: Optional[float] = None) -> None:
        """エラー通知"""
        message = f"❌ **[sound_dp]** {script_name} エラー\n"
        message += f"⏰ エラー時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if duration:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            message += f"📈 実行時間: {hours}時間{minutes}分{seconds}秒\n"
            
        if error_message:
            message += f"💬 エラー内容: {error_message[:500]}...\n"  # 長すぎる場合は切り捨て
            
        self.send_discord_message(message, color=0xff0000)  # 赤色


def main():
    """コマンドライン使用例"""
    if len(sys.argv) < 3:
        print("使用方法: python notify.py <type> <script_name> [additional_info]")
        print("type: start, success, error")
        sys.exit(1)
        
    notifier = NotificationSystem()
    notification_type = sys.argv[1]
    script_name = sys.argv[2]
    additional_info = sys.argv[3] if len(sys.argv) > 3 else ""
    
    if notification_type == "start":
        notifier.notify_start(script_name, additional_info)
    elif notification_type == "success":
        notifier.notify_success(script_name, additional_info=additional_info)
    elif notification_type == "error":
        notifier.notify_error(script_name, error_message=additional_info)
    else:
        print(f"不明な通知タイプ: {notification_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
