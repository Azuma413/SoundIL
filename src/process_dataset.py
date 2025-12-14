import argparse
import re
import os
import sys
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from PIL import Image
import torch

# LeRobotDatasetのインポート
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# SoundConfigのインポート (env.tasks.sound_cameraから)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.tasks.sound_camera import SoundConfig

class DatasetProcessor:
    def __init__(self, input_path: str, target_task: str):
        self.input_path = Path(input_path)
        self.target_task = target_task
        self.config = self._parse_config(target_task)
        
        # LeRobotDatasetの読み込み (ローカル)
        repo_id = f"local/{self.input_path.name}"
        self.dataset = LeRobotDataset(repo_id, root=self.input_path)
        
        # 時間的平滑化用の過去フレーム
        self.past_frame = None
        if self.config.use_temporal_smoothing:
            # チャンネル数はFeatureなら3, それ以外はmic_array_num
            # ただし、入力データセットのmic_array_numに依存する...
            # いや、ターゲットのmic_array_numに合わせる必要がある
            # ここでは処理後のチャンネル数で確保する
            # 入力データは常にm=N (例えばm=6など) あると仮定し、そこから間引くか、
            # あるいは入力もConfigをパースして整合性を取る必要がある
            # User Request: "<task名>-m<マイクロフォンアレイ数>-f30-s2-p0のデータセットが与えられた際に"
            # なので、入力のmと出力のmは異なる可能性があるが、
            # 基本的にはmは減らす方向か、あるいは同じで処理だけ変えるか。
            # 今回は単純化のため、Feature変換後のチャンネル数 or 出力Mic数でバッファを持つ
            num_channels = 3 if self.config.use_feature else self.config.mic_array_num
            self.past_frame = np.ones(
                (self.config.observation_height, self.config.observation_width, num_channels),
                dtype=np.float32
            ) * 127.5

        # 更新頻度管理
        # config.update_freq は Interval (frames) になっている
        self.update_interval = self.config.update_freq
        self.frame_count = 0
        
        # 保持用キャッシュ
        self.cached_sound_map0 = None
        self.cached_sound_map1 = None
        self.cached_spectrogram = None

    def _parse_config(self, task_name: str) -> SoundConfig:
        """タスク名からSoundConfigを生成 (genesis_env.pyと同様のロジック)"""
        pattern = r"-m(\d+)-f(\d+)-s(\d+)-p(\d+)"
        match = re.search(pattern, task_name)
        
        if match:
            m, f, s, p = map(int, match.groups())
            
            mic_array_num = m
            # f: Target Freq (Hz) -> Interval (frames)
            update_freq = max(1, int(30 / f))
            
            use_spectrogram = False
            if s == 0:
                mic_array_num = 0
            elif s == 1:
                use_spectrogram = False
            elif s == 2:
                use_spectrogram = True
                
            use_gaussian_filter = False
            use_temporal_smoothing = False
            use_feature = False
            
            if p == 1:
                use_gaussian_filter = True
            elif p == 2:
                use_temporal_smoothing = True
            elif p == 3:
                use_gaussian_filter = True
                use_temporal_smoothing = True
            elif p == 4:
                use_feature = True
                
            # デフォルト値などは適宜設定
            return SoundConfig(
                mic_array_num=mic_array_num,
                update_freq=update_freq,
                use_spectrogram=use_spectrogram,
                use_gaussian_filter=use_gaussian_filter,
                use_temporal_smoothing=use_temporal_smoothing,
                use_feature=use_feature,
                observation_height=224, # 仮定
                observation_width=224   # 仮定
            )
        else:
            raise ValueError(f"Task name {task_name} does not match format ...-mN-fN-sN-pN")

    def _reconstruct_sound_map(self, sound0, sound1, total_channels):
        """3ch+3chの画像から元のマルチチャンネルマップを復元"""
        # sound0: ch 0-2, sound1: ch 3-5
        # total_channelsに合わせて結合
        # 入力データのmic_array_numを知る必要があるが、
        # ここではsound0, sound1の有効データから推測するか、
        # あるいは単に結合してtargetのmic_array_numだけ使うか。
        # 入力データセットが f30-s2-p0 なら、p0=加工なし なので、
        # sound0, sound1 にはrawのsound map (mic array毎の強度) が入っているはず。
        
        h, w, _ = sound0.shape
        # 最大6チャンネルまで対応 (SoundCameraの実装依存)
        full_map = np.zeros((h, w, 6), dtype=sound0.dtype)
        full_map[:, :, 0:3] = sound0
        full_map[:, :, 3:6] = sound1
        
        # ターゲットのチャンネル数分だけ切り出す
        # もし入力のチャンネル数がターゲットより少なかったら...?
        # User Requestにより、入力は適切なもの(加工用元データ)が与えられると仮定
        return full_map[:, :, :total_channels]

    def _convert_to_feature_image(self, sound_map: np.ndarray) -> np.ndarray:
        """SoundCamera._convert_to_feature_image と同等"""
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
        """SoundCamera._normalize_to_uint8 と同等"""
        array_normalized = np.zeros_like(array, dtype=np.float32)
        # feature imageの場合は処理済みではないか？
        # SoundCameraを見ると、feature image作成後にnormalizeしている
        # しかしfeature image作成内でも255を使っている。
        # SoundCameraの実装順序：
        # 1. sound_map generation (float?) -> No, SoundCamera generates plain float spec map
        # 2. Gaussian Filter
        # 3. Convert to Feature (returns float stack with 0-255 range likely)
        # 4. Normalize to Uint8 (scales min-max to 0-255)
        # Wait, if Feature Image has 0 and 255 values, normalizing it again might be weird if min/max are already 0/255.
        # But _normalize_to_uint8 does min-max normalization per channel.
        # If channel connects 0 and 255, min=0, max=255 -> (x-0)/255*255 = x. Correct.
        
        for i in range(array.shape[2]):
            channel = array[:, :, i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val > min_val:
                array_normalized[:, :, i] = (channel - min_val) / (max_val - min_val) * 255
            else:
                array_normalized[:, :, i] = 127.5 # sound_mapの場合はこれだが、featureの場合は？
                # Feature mapのbinary mapが全部0の場合、max=min=0 -> 127.5になる.
                # これはまずいのでは？ Feature mapの背景は黒(0)であるべき。
                # SoundCameraの実装を再確認すると、_normalize_to_uint8 は汎用的。
                # Featureの場合、binary_mapは0 or 255。もし全部0なら、ここを通るとグレーになる。
                # SoundCameraの実装通りにするなら、そのままコピーする。
                # SoundCamera.py Line 512: array_normalized[:, :, i] = 127.5
                # はい、全部0ならグレーになります。
                # 特徴量変換p=4の場合、音が無い(全部0)ときはグレーになる挙動で正しいか？
                # SoundCameraの挙動を再現するならYES。
        
        return np.clip(array_normalized, 0, 255).astype(np.uint8)

    def _apply_temporal_smoothing(self, current_frame: np.ndarray) -> np.ndarray:
        """SoundCamera._apply_temporal_smoothing と同等"""
        if self.past_frame is None:
            self.past_frame = current_frame.astype(np.float32)
            return current_frame
        current_normalized = current_frame.astype(np.float32) / 255.0
        weight = self.config.temporal_smoothing_weight
        self.past_frame = self.past_frame * (1 - weight) + current_normalized * weight * 255
        self.past_frame = np.clip(self.past_frame, 0, 255)
        return self.past_frame.astype(np.uint8)

    def _split_channels(self, array: np.ndarray, start_ch: int, end_ch: int) -> np.ndarray:
        """SoundCamera._split_channels と同等"""
        height, width, total_channels = array.shape
        result = np.zeros((height, width, 3), dtype=array.dtype)
        available_channels = min(end_ch, total_channels) - start_ch
        available_channels = max(0, available_channels)
        for i in range(min(available_channels, 3)):
            if start_ch + i < total_channels:
                result[:, :, i] = array[:, :, start_ch + i]
        return result

    def process(self, output_name=None):
        if output_name is None:
            # 入力名 + target_suffix (e.g. kitcut_1ep_newtask)
            # しかし target_task 自体がsuffix的な情報を含んでいる
            # 出力Path: datasets/<target_task>
            # 既存のmake_sim_datasetの命名規則に従うなら datasets/taskName_idx
            # ここでは単純に datasets/<target_task>_<idx> とする
            idx = 0
            while True:
                out_path = Path("datasets") / f"{self.target_task}_{idx}"
                if not out_path.exists():
                    break
                idx += 1
            output_name = str(out_path)
            
        print(f"Processing dataset: {self.input_path} -> {output_name}")
        
        # 出力用Datasetの準備
        # featuresは入力からコピーしつつ、specの有無などを調整
        features = self.dataset.features.copy()
        
        # s=0 (mic=0) の場合、specは削除、sound0, sound1は残すが中身は0
        # s=1 (no spec) の場合、specは削除
        # s=2 (spec) の場合、specは残す (入力にあれば)
        
        if not self.config.use_spectrogram:
            if "observation.images.spec" in features:
                del features["observation.images.spec"]
                
        # LeRobotDataset作成
        # image_writer_threadsなどを適切に設定
        output_dataset = LeRobotDataset.create(
            repo_id=None,
            fps=30,
            root=output_name,
            robot_type="so-101",
            use_videos=True,
            features=features,
            batch_encoding_size=1, # 安全のため1
        )
        
        # エピソードループ
        for ep_idx in range(self.dataset.num_episodes):
            print(f"Processing Episode {ep_idx+1}/{self.dataset.num_episodes}")
            # LeRobotDataset (v3.0) might use meta.episodes list
            # meta.episodes[i] contains "dataset_from_index" and "dataset_to_index"
            episode_meta = self.dataset.meta.episodes[ep_idx]
            from_idx = episode_meta["dataset_from_index"]
            to_idx = episode_meta["dataset_to_index"]
            
            # Reset state for new episode
            self.frame_count = 0
            if self.config.use_temporal_smoothing:
                num_channels = 3 if self.config.use_feature else self.config.mic_array_num
                self.past_frame = np.ones(
                    (self.config.observation_height, self.config.observation_width, num_channels),
                    dtype=np.float32
                ) * 127.5
            self.cached_sound_map0 = None
            self.cached_sound_map1 = None
            self.cached_spectrogram = None
            
            # フレームループ
            for frame_idx in range(from_idx, to_idx):
                item = self.dataset[frame_idx]
                
                # アイテム辞書を作成 (Modify用に)
                # itemはTensor等かもしれないのでnumpyへ
                frame_data = {}
                for k, v in item.items():
                    if isinstance(v, torch.Tensor):
                        v = v.cpu().numpy()
                    frame_data[k] = v
                
                # 画像のProcessing
                # 1. 取得
                # 入力データセットには sound0, sound1 があると仮定
                # (s=0, p=0のデータセットでも sound0,1 はゼロ埋めで存在するはずだが、
                #  加工元となるデータセットは音情報を含んでいる前提 (s=2, p=0など))
                # アイテム辞書を作成 (Modify用に)
                frame_data = {}
                for k, v in item.items():
                    if isinstance(v, torch.Tensor):
                        v = v.cpu().numpy()
                    frame_data[k] = v

                # Remove system keys that LeRobotDataset adds automatically or retrieves
                system_keys = ["index", "episode_index", "frame_index", "timestamp", "task_index", "dataset_index"]
                for sk in system_keys:
                    if sk in frame_data:
                        del frame_data[sk]

                # Transpose ALL image/video features from CHW to HWC
                # output_dataset.features にあるものだけで、かつ画像/動画タイプのもの
                for k, feat_info in output_dataset.features.items():
                    if k in frame_data and feat_info["dtype"] in ["image", "video"]:
                        arr = frame_data[k]
                        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[0] == 3:
                            # (3, H, W) -> (H, W, 3) assumption
                            # H, W check?
                            # shape mismatch error said: '(3, 224, 224)' does not have expected shape '(224, 224, 3)'
                            frame_data[k] = arr.transpose(1, 2, 0)

                # Sound Processing Preparation
                src_sound0 = frame_data.get("observation.images.sound0")
                src_sound1 = frame_data.get("observation.images.sound1")
                src_spec = frame_data.get("observation.images.spec")
                
                # ... (rest of processing uses frame_data in place) ...
                
                if src_sound0 is None:
                     pass

                # 更新判定
                should_update = (self.frame_count % self.update_interval == 0)
                self.frame_count += 1
                
                if should_update or self.cached_sound_map0 is None:
                    # 加工計算
                    if self.config.mic_array_num == 0:
                        h, w = self.config.observation_height, self.config.observation_width
                        sound_map0 = np.zeros((h, w, 3), dtype=np.uint8)
                        sound_map1 = np.zeros((h, w, 3), dtype=np.uint8)
                        spectrogram = None
                    else:
                        full_map = self._reconstruct_sound_map(src_sound0, src_sound1, self.config.mic_array_num)
                        
                        if self.config.use_gaussian_filter:
                            for c in range(full_map.shape[2]):
                                full_map[:, :, c] = gaussian_filter(
                                    full_map[:, :, c],
                                    sigma=self.config.gaussian_sigma
                                )
                        
                        if self.config.use_feature:
                            full_map = self._convert_to_feature_image(full_map)
                            
                        # Normalize
                        full_map = self._normalize_to_uint8(full_map)
                        
                        if self.config.use_temporal_smoothing:
                            full_map = self._apply_temporal_smoothing(full_map)
                            
                        sound_map0 = self._split_channels(full_map, 0, 3)
                        sound_map1 = self._split_channels(full_map, 3, 6)
                        
                        if self.config.use_spectrogram and src_spec is not None:
                            spectrogram = src_spec
                        else:
                            spectrogram = None
                            
                    self.cached_sound_map0 = sound_map0
                    self.cached_sound_map1 = sound_map1
                    self.cached_spectrogram = spectrogram
                else:
                    sound_map0 = self.cached_sound_map0
                    sound_map1 = self.cached_sound_map1
                    spectrogram = self.cached_spectrogram
                
                # Ensure outputs are updated in frame_data
                frame_data["observation.images.sound0"] = sound_map0
                frame_data["observation.images.sound1"] = sound_map1
                
                if self.config.use_spectrogram and spectrogram is not None:
                    frame_data["observation.images.spec"] = spectrogram
                elif "observation.images.spec" in frame_data:
                    del frame_data["observation.images.spec"]
                    
                # task description
                frame_data["task"] = self.config.get_task_description(self.target_task) if hasattr(self.config, "get_task_description") else "Sound Task"
                
                # Filter only valid keys for add_frame
                final_frame = {}
                for k in output_dataset.features:
                    if k in frame_data:
                        final_frame[k] = frame_data[k]
                final_frame["task"] = frame_data["task"]
                
                output_dataset.add_frame(final_frame)
                
            output_dataset.save_episode()
            
        return output_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset (e.g. datasets/soundShake-m4-f30-s2-p0_0)")
    parser.add_argument("--target", type=str, required=True, help="Target task string (e.g. soundShake-m4-f5-s2-p4)")
    
    args = parser.parse_args()
    
    processor = DatasetProcessor(args.input, args.target)
    out_path = processor.process()
    print(f"Dataset processing complete. Saved to {out_path}")
