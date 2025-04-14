# sound_dp
音環境認識DiffusionPolicy

Python3.10

## Setup
```bash
git clone --recurse-submodules https://github.com/Azuma413/sound_dp.git
cd sound_dp
uv sync
uv pip install torch torchvision torchaudio
cd Genesis
uv pip install -e ".[dev]"
cd ../lerobot
uv pip install -e .
```