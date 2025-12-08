import sys
import os
import numpy as np
import time
from unittest.mock import MagicMock

# Add path to allow imports
sys.path.append(os.getcwd())

from env.tasks.sound_camera import SoundCamera, SoundConfig

def test_signal_logic():
    print("Testing SoundCamera Signal Logic...")
    
    # Mock target
    mock_target = MagicMock()
    mock_target.get_pos.return_value = np.array([0.0, 0.0, 0.0])
    
    # Config
    config = SoundConfig(
        fs=16000,
        processing_time=1.0, # 1 second buffer
        shake_mode=True,
        velocity_threshold=0.1,
        update_freq=1,
        mic_array_num=1 # Simplify
    )
    
    # Initialize Camera
    camera = SoundCamera(mock_target, config)
    
    # Mock _simulate_all_arrays to avoid heavy computation and check signal
    mock_doa = MagicMock()
    mock_doa.Pssl = np.ones((360, 1)) 
    camera._simulate_all_arrays = MagicMock()
    camera._simulate_all_arrays.return_value = ([np.zeros((8, 100))], [mock_doa])
    
    # Mock audio signal (constant 1.0 for easy tracking)
    camera.audio_signal = np.ones(16000 * 10, dtype=np.float32) # 10 seconds of 1s
    
    # Test Step 1: Initial State
    print("Step 1: Initial State")
    assert np.all(camera.signal_buffer == 0), "Buffer should be initialized to zeros"
    
    # Test Step 2: Update with low velocity (0.0)
    print("Step 2: Update with low velocity (0.0)")
    # Manually set prev_time to simulate dt
    camera.prev_time = time.time() - 0.033 # 30FPS
    camera.prev_pos = np.array([0.0, 0.0, 0.0])
    
    # Call render
    camera.render()
    
    # Check signal passed to simulate
    args, _ = camera._simulate_all_arrays.call_args
    signal_passed = args[1]
    
    # With velocity 0, mask should be 0, BUT we added noise injection.
    # So signal should NOT be all zeros.
    print(f"Signal max: {np.max(np.abs(signal_passed))}")
    assert not np.all(signal_passed == 0), "Signal should NOT be all zeros (noise injected)"
    assert np.max(np.abs(signal_passed)) < 1e-3, "Noise should be small"
    
    # Test Step 3: Update with high velocity (1.0)
    print("Step 3: Update with high velocity (1.0)")
    # Move target to create velocity
    # dist = vel * dt = 1.0 * 0.033 = 0.033
    # We need to update prev_time manually because we mocked time.time() implicitly by setting prev_time
    # Actually render calls time.time().
    # Let's just rely on the fact that render updates prev_time.
    # But we need to make sure dt is calculated correctly.
    # In render: dt = self.prev_time - last_time.
    # We need to sleep or mock time.
    
    # Let's just force update_state directly for precise control if needed, 
    # but render logic is what we want to test.
    # We can just set prev_time to (now - 0.033) before calling render.
    camera.prev_time = time.time() - 0.033
    mock_target.get_pos.return_value = np.array([0.033, 0.0, 0.0])
    
    camera.render()
    
    args, _ = camera._simulate_all_arrays.call_args
    signal_passed = args[1]
    
    # Newest part should be 1.0 (unmasked)
    # Old part (from Step 2) was masked+noise.
    
    n_samples = int(0.033 * 16000)
    print(f"Signal passed tail (newest): {signal_passed[-10:]}")
    
    assert np.all(signal_passed[-n_samples:] == 1.0), "Newest signal should be unmasked"
    
    print("Test Passed!")

if __name__ == "__main__":
    test_signal_logic()
