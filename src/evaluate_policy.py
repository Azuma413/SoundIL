from pathlib import Path
import gymnasium as gym
import imageio
import numpy
import torch
import genesis as gs
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.genesis_env import GenesisEnv

# Create a directory to store the video of the evaluation
output_directory = Path("outputs/eval/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)
device = "cuda"
pretrained_policy_path = "lerobot/diffusion_pusht"
policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

gs.init(backend=gs.gpu, precision="32")
env = GenesisEnv(task="test", observation_height=480, observation_width=640, show_viewer=True)
print(policy.config.input_features)
print(env.observation_space)
print(policy.config.output_features)
print(env.action_space)
policy.reset()
numpy_observation, info = env.reset(seed=42)
rewards = []
frames = []
frames.append(env.render())
step = 0
done = False
limit = 600
while not done:
    # Prepare observation for the policy running in Pytorch
    state = torch.from_numpy(numpy_observation["agent_pos"])
    image = torch.from_numpy(numpy_observation["pixels"])
    state = state.to(torch.float32)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)

    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    # Create the policy input dictionary
    observation = {
        "observation.state": state,
        "observation.image": image,
    }

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(observation)

    # Prepare the action for the environment
    numpy_action = action.squeeze(0).to("cpu").numpy()

    # Step through the environment and receive a new observation
    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"{step=} {reward=} {terminated=}")

    # Keep track of all the rewards and frames
    rewards.append(reward)
    frames.append(env.render())

    # The rollout is considered done when the success state is reach (i.e. terminated is True),
    # or the maximum number of iterations is reached (i.e. truncated is True)
    done = terminated | truncated | done
    step += 1
    if step > 600:
        break

if terminated:
    print("Success!")
else:
    print("Failure!")

# Get the speed of environment (i.e. its number of frames per second).
fps = env.metadata["render_fps"]

# Encode all frames into a mp4 video.
video_path = output_directory / "rollout.mp4"
imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

print(f"Video of the evaluation is available in '{video_path}'.")
