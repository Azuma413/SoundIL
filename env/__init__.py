from gymnasium.envs.registration import register

register(
    id="gym_genesis/SoundPick-v0",
    entry_point="gym_genesis.env:GenesisEnv",
    max_episode_steps=200,
    nondeterministic=False,
    kwargs={
        "task": "sound",
        "observation_height": 480,
        "observation_width": 640,
        "show_viewer": False,
    }
)
register(
    id="gym_genesis/TestPick-v0",
    entry_point="gym_genesis.env:GenesisEnv",
    max_episode_steps=200,
    nondeterministic=False,
    kwargs={
        "task": "test",
        "observation_height": 480,
        "observation_width": 640,
        "show_viewer": False,
    }
)