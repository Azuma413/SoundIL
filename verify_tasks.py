import genesis as gs
import numpy as np
import cv2
import os
from env.genesis_env import GenesisEnv
from env.tasks.sound_camera import SoundConfig

def verify_task(task_name, description):
    print(f"\n=== Verifying {task_name}: {description} ===")
    
    # SoundConfig (using defaults or specific if needed)
    sound_config = SoundConfig(
        observation_height=224,
        observation_width=224,
        mic_array_num=6,
        use_spectrogram=True,
        audio_file_path="sounds/1.wav"
    )
    
    try:
        env = GenesisEnv(
            task=task_name,
            observation_height=224,
            observation_width=224,
            show_viewer=False,
            sound_config=sound_config
        )
    except Exception as e:
        print(f"Failed to initialize env: {e}")
        return

    print("Task Description:", env.get_task_description())
    
    obs, info = env.reset()
    print("Reset successful.")
    
    # Check number of cubes/boxes based on internal state if possible
    # We can access env._env which is the SoundTask instance
    task_instance = env._env
    print(f"Task Type: {getattr(task_instance, 'task_type', 'Unknown')}")
    print(f"Num Cubes: {getattr(task_instance, 'num_cubes', 'Unknown')}")
    print(f"Use Two Boxes: {getattr(task_instance, 'use_two_boxes', 'Unknown')}")
    
    if task_name.startswith("soundDiff"):
        print(f"Current Sound Type: {getattr(task_instance, 'current_sound_type', 'Unknown')}")
        if hasattr(task_instance, 'target_box'):
             print(f"Target Box Pos: {task_instance.target_box.get_pos().cpu().numpy()}")
    
    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check sound images
        s0 = obs["observation.images.sound0"]
        s1 = obs["observation.images.sound1"]
        print(f"Step {i}: Sound0 max={s0.max()}, Sound1 max={s1.max()}")
        
        if task_name.startswith("soundShake") and i == 0:
            # In shake mode, initially should be silent if not moving?
            # But random action might move it.
            # Let's check velocity if possible
            vel = task_instance.sound_cam.get_target_velocity()
            print(f"Target Velocity: {vel}")

    env.close()
    print(f"=== {task_name} Verified ===\n")

if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32")
    
    # Verify sound (default)
    verify_task("sound-m3-fo-so", "2 cubes, pick sounding one")
    
    # Verify soundDiff
    verify_task("soundDiff-m3-fo-so", "1 cube, 2 boxes, sound A/B")
    
    # Verify soundShake
    verify_task("soundShake-m3-fo-so", "2 cubes, shake to hear")
