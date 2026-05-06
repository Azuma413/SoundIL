import argparse
import csv
import os
from contextlib import nullcontext
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_STR
from lerobot.utils.utils import get_safe_torch_device

from src.eval_policy import (
    fill_missing_image_observations,
    infer_dataset_name,
    infer_task_name,
    normalize_checkpoint_step,
)
from env.genesis_env import GenesisEnv


COLOR_BY_OPTIONS = ["sound_type", "sound_coordinate", "success", "episode_step"]


def load_policy(training_name, pretrained_policy_path, device):
    model_type = training_name.split("_")[0]
    if model_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    elif model_type == "act":
        policy = ACTPolicy.from_pretrained(pretrained_policy_path)
        policy.config.n_action_steps = 1
        policy.config.temporal_ensemble_coeff = 0.01
        policy.temporal_ensembler = ACTTemporalEnsembler(
            policy.config.temporal_ensemble_coeff, policy.config.chunk_size
        )
        policy.reset()
        print(
            "Overriding ACT eval config: "
            f"n_action_steps={policy.config.n_action_steps}, "
            f"temporal_ensemble_coeff={policy.config.temporal_ensemble_coeff}"
        )
    elif model_type == "pi0":
        policy = PI0Policy.from_pretrained(pretrained_policy_path)
    elif model_type == "vqbet":
        policy = VQBeTPolicy.from_pretrained(pretrained_policy_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    policy.to(device)
    policy.eval()
    return policy


def resolve_module(root_module, module_path):
    module = root_module
    for part in module_path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


class HiddenStateRecorder:
    def __init__(self, policy, hidden_layer):
        self.policy = policy
        self.hidden_layer = hidden_layer
        self.current = None
        self.handle = None

    def __enter__(self):
        module, module_path = self._find_module()
        print(f"Recording hidden states from: {module_path}")

        def hook(_module, inputs, _output):
            hidden = inputs[0] if inputs else _output
            if isinstance(hidden, (tuple, list)):
                hidden = hidden[0]
            self.current = hidden.detach().float().cpu()

        self.handle = module.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None

    def pop(self):
        hidden = self.current
        self.current = None
        return hidden

    def _find_module(self):
        if self.hidden_layer != "auto":
            return resolve_module(self.policy, self.hidden_layer), self.hidden_layer

        model_name = getattr(self.policy, "name", "")
        auto_paths = {
            "act": ("model.action_head",),
            "diffusion": ("diffusion.unet.final_conv",),
            "vqbet": ("vqbet.action_head",),
            "pi0": ("model.action_out_proj",),
        }

        for module_path in auto_paths.get(model_name, ()):
            try:
                return resolve_module(self.policy, module_path), module_path
            except (AttributeError, TypeError, IndexError, KeyError):
                continue

        raise ValueError(
            "Could not infer a hidden layer automatically. "
            "Specify one with --hidden-layer, e.g. model.action_head."
        )


def convert_observation(numpy_observation, dataset):
    converted_obs = {}
    for key, value in numpy_observation.items():
        if key.startswith("observation.images."):
            converted_obs[key.replace("observation.images.", "")] = (
                value.copy() if isinstance(value, np.ndarray) else value
            )
        elif key == "observation.state":
            if "observation.state" in dataset.features:
                for i, name in enumerate(dataset.features["observation.state"]["names"]):
                    converted_obs[name] = value[i]
        else:
            converted_obs[key] = value.copy() if isinstance(value, np.ndarray) else value

    fill_missing_image_observations(converted_obs, dataset.features)
    return build_dataset_frame(dataset.features, converted_obs, prefix=OBS_STR)


def predict_action_and_record_hidden(
    observation_frame,
    policy,
    device,
    preprocessor,
    postprocessor,
    use_amp,
    task,
    recorder,
):
    observation = copy(observation_frame)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        observation = prepare_observation_for_inference(observation, device, task, robot_type=None)
        observation = preprocessor(observation)
        action = policy.select_action(observation)
        hidden = recorder.pop()
        action = postprocessor(action)
    return action, hidden


def infer_hidden_layout(policy):
    if getattr(policy, "name", "") == "diffusion":
        return "channel_first"
    return "sequence_last"


def infer_hidden_index(policy):
    model_name = getattr(policy, "name", "")
    if model_name == "vqbet":
        return max(0, getattr(policy.config, "n_obs_steps", 1) - 1)
    if model_name == "diffusion":
        return max(0, getattr(policy.config, "n_obs_steps", 1) - 1)
    return 0


def hidden_to_points(hidden, reduction, layout, hidden_index):
    if hidden is None:
        return None
    hidden = hidden.numpy()
    if hidden.ndim == 3:
        if hidden.shape[0] == 1:
            hidden = hidden[0]
        elif hidden.shape[1] == 1:
            hidden = np.swapaxes(hidden, 0, 1)[0]
        else:
            hidden = hidden.reshape(-1, hidden.shape[-1])

    if hidden.ndim == 1:
        return hidden[None, :]

    if hidden.ndim != 2:
        hidden = hidden.reshape(-1, hidden.shape[-1])

    if layout == "channel_first":
        hidden = hidden.T

    if reduction == "auto":
        index = min(hidden_index, hidden.shape[0] - 1)
        return hidden[index : index + 1]
    if reduction == "none":
        return hidden
    if reduction == "first":
        return hidden[:1]
    if reduction == "last":
        return hidden[-1:]
    if reduction == "mean":
        return hidden.mean(axis=0, keepdims=True)
    raise ValueError(f"Unknown hidden reduction: {reduction}")


def get_sound_metadata(env):
    inner_env = getattr(env, "_env", None)
    sound_type = getattr(inner_env, "current_sound_type", "Unknown")
    coord = np.full(3, np.nan, dtype=np.float32)
    try:
        target = getattr(getattr(inner_env, "sound_cam", None), "target", None)
        if target is not None:
            coord = target.get_pos().detach().cpu().numpy().astype(np.float32)
    except Exception:
        pass
    return sound_type, coord


def append_hidden_records(
    records,
    hidden,
    episode,
    env_step,
    sound_type,
    sound_coord,
    reduction,
    layout,
    hidden_index,
):
    hidden_rows = hidden_to_points(hidden, reduction, layout, hidden_index)
    if hidden_rows is None:
        return
    for chunk_index, vector in enumerate(hidden_rows):
        records.append(
            {
                "hidden": vector,
                "episode": episode,
                "env_step": env_step,
                "chunk_index": chunk_index,
                "sound_type": sound_type,
                "sound_x": float(sound_coord[0]),
                "sound_y": float(sound_coord[1]),
                "sound_z": float(sound_coord[2]),
                "success": False,
                "episode_progress": 0.0,
            }
        )


def assign_episode_outcomes(records, episode_lengths, episode_successes):
    for record in records:
        episode = record["episode"]
        length = max(1, episode_lengths.get(episode, 1) - 1)
        record["success"] = episode_successes.get(episode, False)
        record["episode_progress"] = min(1.0, record["env_step"] / length)


def sample_records(records, max_points, seed):
    if max_points is None or len(records) <= max_points:
        return records
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(records), size=max_points, replace=False))
    return [records[i] for i in indices]


def success_colors(records):
    cmap = LinearSegmentedColormap.from_list("failure_progress", ["#1f77b4", "#ffffff", "#d62728"])
    colors = []
    for record in records:
        if record["success"]:
            colors.append("#1f77b4")
        else:
            colors.append(cmap(record["episode_progress"]))
    return colors


def plot_embedding(embedding, records, color_by, coordinate_axis, output_path):
    fig, ax = plt.subplots(figsize=(9, 7))

    if color_by == "sound_type":
        labels = np.array([record["sound_type"] for record in records])
        unique_labels = sorted(set(labels))
        cmap = plt.get_cmap("tab10", max(1, len(unique_labels)))
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=8,
                alpha=0.75,
                color=cmap(idx),
                label=label,
                linewidths=0,
            )
        ax.legend(title="Sound type", markerscale=2)
    elif color_by == "sound_coordinate":
        axis_to_key = {"x": "sound_x", "y": "sound_y", "z": "sound_z"}
        if coordinate_axis == "radius":
            values = np.array(
                [
                    np.linalg.norm([record["sound_x"], record["sound_y"], record["sound_z"]])
                    for record in records
                ]
            )
        else:
            values = np.array([record[axis_to_key[coordinate_axis]] for record in records])
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=8,
            alpha=0.75,
            c=values,
            cmap="viridis",
            linewidths=0,
        )
        fig.colorbar(scatter, ax=ax, label=f"Sound coordinate: {coordinate_axis}")
    elif color_by == "success":
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=8,
            alpha=0.8,
            c=success_colors(records),
            linewidths=0,
        )
    elif color_by == "episode_step":
        values = np.array([record["env_step"] for record in records])
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=8,
            alpha=0.75,
            c=values,
            cmap="viridis",
            linewidths=0,
        )
        fig.colorbar(scatter, ax=ax, label="Episode step")
    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_metadata_csv(records, embedding, output_path):
    fieldnames = [
        "tsne_x",
        "tsne_y",
        "episode",
        "env_step",
        "chunk_index",
        "sound_type",
        "sound_x",
        "sound_y",
        "sound_z",
        "success",
        "episode_progress",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point, record in zip(embedding, records, strict=False):
            row = {key: record[key] for key in fieldnames if key in record}
            row["tsne_x"] = float(point[0])
            row["tsne_y"] = float(point[1])
            writer.writerow(row)


def run_evaluation(args):
    checkpoint_step = normalize_checkpoint_step(args.checkpoint_step)
    output_directory = Path(args.output_dir) if args.output_dir else Path(
        f"outputs/tsne/{args.training_name}_{checkpoint_step}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_policy_path = Path(
        f"outputs/train/{args.training_name}/checkpoints/{checkpoint_step}/pretrained_model"
    )
    if not pretrained_policy_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_policy_path}")

    print(f"Using device: {device}")
    print(f"Loading policy from: {pretrained_policy_path}")
    policy = load_policy(args.training_name, pretrained_policy_path, device)

    dataset_name = args.dataset_name or infer_dataset_name(args.training_name)
    task_name = args.task_name or infer_task_name(dataset_name)
    dataset_path = Path(f"datasets/{dataset_name}").resolve()
    print(f"Loading dataset from: {dataset_path}")
    dataset = LeRobotDataset(str(dataset_path))

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(pretrained_policy_path),
        dataset_stats=dataset.meta.stats,
    )

    env = GenesisEnv(
        task=task_name,
        observation_height=args.observation_height,
        observation_width=args.observation_width,
        show_viewer=args.show_viewer,
    )

    records = []
    episode_lengths = {}
    episode_successes = {}
    torch_device = get_safe_torch_device(policy.config.device)
    hidden_layout = getattr(args, "hidden_layout", "auto")
    if hidden_layout == "auto":
        hidden_layout = infer_hidden_layout(policy)
    hidden_index = getattr(args, "hidden_index", None)
    if hidden_index is None:
        hidden_index = infer_hidden_index(policy)
    hidden_reduction = getattr(args, "hidden_reduction", "auto")
    print(
        "Hidden point selection: "
        f"reduction={hidden_reduction}, layout={hidden_layout}, index={hidden_index}"
    )

    with HiddenStateRecorder(policy, args.hidden_layer) as recorder:
        ep = 0
        while ep < args.episode_num:
            try:
                print(f"\n=== Episode {ep + 1}/{args.episode_num} ===")
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                numpy_observation, _ = env.reset()

                rewards = []
                done = False
                step = 0
                while not done:
                    observation_frame = convert_observation(numpy_observation, dataset)
                    task_description = env.get_task_description()
                    sound_type, sound_coord = get_sound_metadata(env)
                    action_dict, hidden = predict_action_and_record_hidden(
                        observation_frame=observation_frame,
                        policy=policy,
                        device=torch_device,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=task_description,
                        recorder=recorder,
                    )
                    append_hidden_records(
                        records,
                        hidden,
                        ep,
                        step,
                        sound_type,
                        sound_coord,
                        hidden_reduction,
                        hidden_layout,
                        hidden_index,
                    )

                    action_tensor = action_dict["action"] if isinstance(action_dict, dict) else action_dict
                    numpy_action = action_tensor.squeeze(0).cpu().numpy()
                    numpy_observation, reward, terminated, truncated, _info = env.step(numpy_action)
                    rewards.append(reward)
                    done = terminated or truncated or (reward > 0)
                    step += 1

                    if args.max_eval_steps is not None and step >= args.max_eval_steps:
                        done = True

                total_reward = sum(rewards)
                episode_lengths[ep] = step
                episode_successes[ep] = total_reward > 0
                print(
                    f"Episode {ep + 1}: steps={step}, reward={total_reward:.4f}, "
                    f"success={episode_successes[ep]}"
                )
                ep += 1
            except Exception as exc:
                print(f"Error during episode {ep + 1}: {exc}")
                print("Retrying this episode with a fresh environment.")
                env.close()
                env = GenesisEnv(
                    task=task_name,
                    observation_height=args.observation_height,
                    observation_width=args.observation_width,
                    show_viewer=args.show_viewer,
                )

    env.close()
    assign_episode_outcomes(records, episode_lengths, episode_successes)
    if not records:
        raise RuntimeError("No hidden states were recorded. Check --hidden-layer and the policy type.")

    records = sample_records(records, args.max_points, args.seed)
    if len(records) < 2:
        raise RuntimeError("t-SNE needs at least 2 hidden-state points. Increase --episode-num or --max-eval-steps.")

    hidden_matrix = np.stack([record["hidden"] for record in records]).astype(np.float32)
    hidden_path = output_directory / "hidden_states.npz"
    np.savez_compressed(
        hidden_path,
        hidden=hidden_matrix,
        episode=np.array([record["episode"] for record in records]),
        env_step=np.array([record["env_step"] for record in records]),
        chunk_index=np.array([record["chunk_index"] for record in records]),
        sound_type=np.array([record["sound_type"] for record in records]),
        sound_coord=np.array(
            [[record["sound_x"], record["sound_y"], record["sound_z"]] for record in records],
            dtype=np.float32,
        ),
        success=np.array([record["success"] for record in records]),
        episode_progress=np.array([record["episode_progress"] for record in records], dtype=np.float32),
    )

    perplexity = min(args.perplexity, max(1, len(records) - 1))
    print(f"Running t-SNE on {len(records)} points with perplexity={perplexity}...")
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(hidden_matrix)

    color_by_options = [args.color_by] if args.color_by is not None else COLOR_BY_OPTIONS
    plot_paths = []
    for color_by in color_by_options:
        if color_by == "sound_coordinate":
            for coordinate_axis in ("x", "y"):
                plot_path = output_directory / f"tsne_{color_by}_{coordinate_axis}.png"
                plot_embedding(embedding, records, color_by, coordinate_axis, plot_path)
                plot_paths.append(plot_path)
        else:
            plot_path = output_directory / f"tsne_{color_by}.png"
            plot_embedding(embedding, records, color_by, "y", plot_path)
            plot_paths.append(plot_path)
    save_metadata_csv(records, embedding, output_directory / "tsne_metadata.csv")

    success_count = sum(episode_successes.values())
    with open(output_directory / "summary.txt", "w") as f:
        f.write(f"training_name: {args.training_name}\n")
        f.write(f"checkpoint_step: {checkpoint_step}\n")
        f.write(f"dataset_name: {dataset_name}\n")
        f.write(f"task_name: {task_name}\n")
        f.write(f"episodes: {args.episode_num}\n")
        f.write(f"success: {success_count}/{args.episode_num}\n")
        f.write(f"hidden_layer: {args.hidden_layer}\n")
        f.write(f"hidden_reduction: {hidden_reduction}\n")
        f.write(f"hidden_layout: {hidden_layout}\n")
        f.write(f"hidden_index: {hidden_index}\n")
        f.write(f"points: {len(records)}\n")
        for plot_path in plot_paths:
            f.write(f"plot: {plot_path}\n")
        f.write(f"hidden_states: {hidden_path}\n")

    for plot_path in plot_paths:
        print(f"Saved plot: {plot_path}")
    print(f"Saved hidden states: {hidden_path}")
    print(f"Saved metadata: {output_directory / 'tsne_metadata.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation for a checkpoint and plot t-SNE of model hidden states."
    )
    parser.add_argument("--training-name", default="act_soundDiff-m4-f10-s2-p0_0")
    parser.add_argument("--checkpoint-step", default="100000")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--episode-num", type=int, default=10)
    parser.add_argument("--observation-height", type=int, default=224)
    parser.add_argument("--observation-width", type=int, default=224)
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument(
        "--color-by",
        choices=COLOR_BY_OPTIONS,
        default=None,
        help="Coloring mode. If omitted, plots are generated for all modes.",
    )
    parser.add_argument(
        "--hidden-layer",
        default="auto",
        help=(
            "Module to hook. The hook records the module input. "
            "auto supports act, diffusion, vqbet, and pi0."
        ),
    )
    parser.add_argument(
        "--hidden-reduction",
        choices=["auto", "first", "last", "mean", "none"],
        default="auto",
        help=(
            "How to convert a sequence hidden state into t-SNE points. "
            "auto selects one current-step point per model invocation."
        ),
    )
    parser.add_argument(
        "--hidden-layout",
        choices=["auto", "sequence_last", "channel_first"],
        default="auto",
        help="Use channel_first for Conv1d hidden tensors shaped as channels x steps.",
    )
    parser.add_argument(
        "--hidden-index",
        type=int,
        default=None,
        help="Sequence index used by --hidden-reduction=auto. Defaults to the model's current action index.",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run_evaluation(parse_args())
