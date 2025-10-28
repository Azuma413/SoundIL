"""
新しいSoundCameraクラスのテストスクリプト
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sound_camera import SoundCamera, SoundConfig


class MockTarget:
    """テスト用のモック音源オブジェクト"""
    def __init__(self, position):
        self.position = torch.tensor(position, dtype=torch.float32)
    
    def get_pos(self):
        return self.position


def test_basic_soundmap():
    """基本的なSoundMap生成のテスト"""
    print("=" * 50)
    print("Test 1: 基本的なSoundMap生成")
    print("=" * 50)
    
    # 設定
    config = SoundConfig(
        observation_height=128,
        observation_width=128,
        mic_array_num=3,
        use_spectrogram=False,
        use_feature=False
    )
    
    # モック音源（10x10x3の部屋に合わせた位置）
    target = MockTarget([5.5, 5.5, 0.3])
    
    # SoundCameraの初期化
    sound_cam = SoundCamera(target, config)
    
    # レンダリング
    sound_map, spectrogram = sound_cam.render()
    
    # 結果の確認
    print(f"SoundMap shape: {sound_map.shape}")
    print(f"SoundMap dtype: {sound_map.dtype}")
    print(f"SoundMap range: [{sound_map.min()}, {sound_map.max()}]")
    print(f"Spectrogram: {spectrogram}")
    
    assert sound_map.shape == (128, 128, 3), f"Expected shape (128, 128, 3), got {sound_map.shape}"
    assert sound_map.dtype == np.uint8, f"Expected dtype uint8, got {sound_map.dtype}"
    assert spectrogram is None, "Spectrogram should be None when use_spectrogram=False"
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axes[i].imshow(sound_map[:, :, i], cmap='viridis')
        axes[i].set_title(f'Mic Array {i+1}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('images/test_basic_soundmap.png')
    plt.close()
    
    print("✓ Test 1 passed\n")


def test_feature_image():
    """特徴画像生成のテスト"""
    print("=" * 50)
    print("Test 2: 特徴画像生成")
    print("=" * 50)
    
    # 設定
    config = SoundConfig(
        observation_height=128,
        observation_width=128,
        mic_array_num=3,
        use_feature=True,
        feature_threshold=0.8,
        marker_size=3
    )
    
    # モック音源（10x10x3の部屋に合わせた位置）
    target = MockTarget([5.5, 5.5, 0.3])
    
    # SoundCameraの初期化
    sound_cam = SoundCamera(target, config)
    
    # レンダリング
    sound_map, spectrogram = sound_cam.render()
    
    # 結果の確認
    print(f"Feature image shape: {sound_map.shape}")
    print(f"Feature image dtype: {sound_map.dtype}")
    
    assert sound_map.shape == (128, 128, 3), f"Expected shape (128, 128, 3), got {sound_map.shape}"
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Mean', 'Marker', 'Binary']
    for i in range(3):
        axes[i].imshow(sound_map[:, :, i], cmap='viridis')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('images/test_feature_image.png')
    plt.close()
    
    print("✓ Test 2 passed\n")


def test_gaussian_filter():
    """ガウシアンフィルタのテスト"""
    print("=" * 50)
    print("Test 3: ガウシアンフィルタ")
    print("=" * 50)
    
    # 設定
    config = SoundConfig(
        observation_height=128,
        observation_width=128,
        mic_array_num=3,
        use_gaussian_filter=True,
        gaussian_sigma=2.0
    )
    
    # モック音源（10x10x3の部屋に合わせた位置）
    target = MockTarget([5.5, 5.5, 0.3])
    
    # SoundCameraの初期化
    sound_cam = SoundCamera(target, config)
    
    # レンダリング
    sound_map, spectrogram = sound_cam.render()
    
    print(f"Filtered SoundMap shape: {sound_map.shape}")
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axes[i].imshow(sound_map[:, :, i], cmap='viridis')
        axes[i].set_title(f'Filtered Mic Array {i+1}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('images/test_gaussian_filter.png')
    plt.close()
    
    print("✓ Test 3 passed\n")


def test_multiple_mic_arrays():
    """複数のマイクアレイのテスト"""
    print("=" * 50)
    print("Test 4: 複数のマイクアレイ")
    print("=" * 50)
    
    # 設定（6個のマイクアレイ）
    config = SoundConfig(
        observation_height=128,
        observation_width=128,
        mic_array_num=6,
        use_feature=False
    )
    
    # モック音源（10x10x3の部屋に合わせた位置）
    target = MockTarget([5.5, 5.5, 0.3])
    
    # SoundCameraの初期化
    sound_cam = SoundCamera(target, config)
    
    # レンダリング
    sound_map, spectrogram = sound_cam.render()
    
    # 結果の確認
    print(f"SoundMap shape with 6 arrays: {sound_map.shape}")
    
    assert sound_map.shape == (128, 128, 6), f"Expected shape (128, 128, 6), got {sound_map.shape}"
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(6):
        row, col = i // 3, i % 3
        axes[row, col].imshow(sound_map[:, :, i], cmap='viridis')
        axes[row, col].set_title(f'Mic Array {i+1}')
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig('images/test_multiple_arrays.png')
    plt.close()
    
    print("✓ Test 4 passed\n")


def test_temporal_smoothing():
    """時間的平滑化のテスト"""
    print("=" * 50)
    print("Test 5: 時間的平滑化")
    print("=" * 50)
    
    # 設定
    config = SoundConfig(
        observation_height=128,
        observation_width=128,
        mic_array_num=3,
        use_temporal_smoothing=True,
        temporal_smoothing_weight=0.3
    )
    
    # モック音源（移動する、10x10x3の部屋に合わせた位置）
    positions = [
        [5.5, 5.5, 0.3],
        [5.8, 5.2, 0.3],
        [6.1, 4.9, 0.3]
    ]
    
    # SoundCameraの初期化
    target = MockTarget(positions[0])
    sound_cam = SoundCamera(target, config)
    
    # 複数フレームのレンダリング
    sound_cam.start_recording()
    
    for i, pos in enumerate(positions):
        target.position = torch.tensor(pos, dtype=torch.float32)
        sound_map, _ = sound_cam.render()
        print(f"Frame {i+1} rendered, shape: {sound_map.shape}")
    
    print(f"Total frames recorded: {len(sound_cam.frames)}")
    
    assert len(sound_cam.frames) == 3, "Should have 3 frames"
    
    print("✓ Test 5 passed\n")


def test_with_beamforming():
    """ビームフォーミングとスペクトログラムのテスト（test.pyと同じ条件）"""
    print("=" * 50)
    print("Test 6: ビームフォーミングとスペクトログラム（test.pyと同じ条件）")
    print("=" * 50)
    
    # 設定（test.pyと完全に同じ条件）
    config = SoundConfig(
        observation_height=100,  # test.pyのmap_scale=10でN=100に相当
        observation_width=100,
        mic_array_num=6,  # test.pyのmic_num=6
        mics_per_array=8,  # test.pyのnum_microphones=8
        mic_radius=0.035,  # 修正済み
        use_spectrogram=True,
        num_peaks=2,  # test.pyのk=2
        gaussian_sigma=1.0,  # test.pyのgaussian_filter(X, 1)
        audio_file_path="sounds/1.wav",
        fs=16000,
        nfft=512,
        music_num_src=3,
        room_max_order=17
    )
    
    # モック音源（test.pyと完全に同じ位置）
    target = MockTarget([5.5, 5.5, 0.3])
    
    # SoundCameraの初期化
    sound_cam = SoundCamera(target, config)
    
    # レンダリング
    sound_map, spectrogram = sound_cam.render()
    
    # 結果の確認
    print(f"SoundMap shape: {sound_map.shape}")
    print(f"Spectrogram shape: {spectrogram.shape if spectrogram is not None else 'None'}")
    
    if spectrogram is not None:
        plt.imsave('images/test_spec.png', spectrogram)
        print("✓ Spectrogram generated successfully")
        print(f"✓ Spectrogram saved to images/test_beamformed_spectrogram.png")
    else:
        print("⚠ Spectrogram generation failed or returned None")
    
    print("✓ Test 6 passed\n")

def test_combined_audio_and_noise():
    """音声ファイルとノイズの組み合わせテスト"""
    print("=" * 50)
    print("Test 9: 音声ファイル + ノイズ")
    print("=" * 50)

    # 設定（音声ファイル + ノイズ）
    config = SoundConfig(
        observation_height=128,
        observation_width=128,
        mic_array_num=3,
        audio_file_path="sounds/1.wav",
        noise_intensity=0.5,
        use_spectrogram=True,
    )
    
    # モック音源（10x10x3の部屋に合わせた位置）
    target = MockTarget([5.5, 5.5, 0.3])
    
    # SoundCameraの初期化
    sound_cam = SoundCamera(target, config)
    
    # レンダリング
    sound_map, spectrogram = sound_cam.render()
    print(spectrogram)
    plt.imsave('images/test_noise_spec.png', spectrogram)
    print("✓ Test 9 passed\n")

def main():
    """すべてのテストを実行"""
    print("\n" + "=" * 50)
    print("SoundCamera クラステスト開始")
    print("=" * 50 + "\n")
    
    try:
        test_basic_soundmap()
        test_feature_image()
        test_gaussian_filter()
        test_multiple_mic_arrays()
        test_temporal_smoothing()
        test_with_beamforming()
        test_combined_audio_and_noise()
        
        print("\n" + "=" * 50)
        print("すべてのテストが成功しました！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
