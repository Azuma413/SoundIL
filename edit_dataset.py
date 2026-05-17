from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OPENCV_GUI_PLUGIN_PATH", "")

import numpy as np
from PIL import Image, ImageDraw



ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
LEROBOT_SRC = ROOT / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

LeRobotDataset: Any | None = None


SYSTEM_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


@dataclass
class EpisodeData:
    features: dict[str, list[np.ndarray]]
    tasks: list[str]

    @property
    def size(self) -> int:
        if self.features:
            return len(next(iter(self.features.values())))
        return len(self.tasks)


def list_dataset_dirs() -> list[Path]:
    if not DATASETS_DIR.exists():
        return []
    return sorted(path for path in DATASETS_DIR.iterdir() if (path / "meta" / "info.json").exists())


def load_info(dataset_root: Path) -> dict[str, Any]:
    with (dataset_root / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def user_features(features: dict[str, dict]) -> list[str]:
    return [name for name in features if name not in SYSTEM_FEATURES]


def normalize_feature_spec(info: dict[str, Any]) -> dict[str, Any]:
    spec = dict(info)
    if isinstance(spec.get("shape"), list):
        spec["shape"] = tuple(spec["shape"])
    if isinstance(spec.get("names"), list):
        spec["names"] = tuple(spec["names"])
    return spec


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def get_lerobot_dataset_class() -> Any:
    global LeRobotDataset
    if LeRobotDataset is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset

        LeRobotDataset = _LeRobotDataset
    return LeRobotDataset


def normalize_image(value: Any) -> np.ndarray:
    arr = tensor_to_numpy(value)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[0] < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max(initial=0) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def normalize_numeric(value: Any) -> np.ndarray:
    arr = tensor_to_numpy(value)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def infer_output_name(source_name: str) -> str:
    base = f"{source_name}-edited"
    candidate = base
    idx = 1
    while (DATASETS_DIR / candidate).exists():
        candidate = f"{base}-{idx}"
        idx += 1
    return candidate


def make_plot_image(
    values: list[np.ndarray],
    frame_index: int,
    feature: str,
    names: list[str] | tuple[str] | None,
    width: int = 520,
    height: int = 340,
) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 52, 30, width - 14, height - 34
    draw.rectangle((left, top, right, bottom), outline=(190, 190, 190))
    draw.text((8, 8), feature, fill=(20, 20, 20))
    if not values:
        return image

    data = np.stack(values).astype(np.float32)
    if data.ndim == 1:
        data = data[:, None]
    y_min = float(np.nanmin(data))
    y_max = float(np.nanmax(data))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return image
    if y_max <= y_min:
        y_max = y_min + 1.0

    palette = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (148, 103, 189),
        (255, 127, 14),
        (23, 190, 207),
        (127, 127, 127),
        (188, 189, 34),
    ]
    plot_w = max(right - left, 1)
    plot_h = max(bottom - top, 1)
    denom = max(len(data) - 1, 1)
    for dim in range(data.shape[1]):
        points = []
        for idx, value in enumerate(data[:, dim]):
            x = left + int(round(idx / denom * plot_w))
            y = bottom - int(round((float(value) - y_min) / (y_max - y_min) * plot_h))
            points.append((x, y))
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=palette[dim % len(palette)])
        else:
            draw.line(points, fill=palette[dim % len(palette)], width=2)

    x_bar = left + int(round(max(0, min(frame_index, len(data) - 1)) / denom * plot_w))
    draw.line((x_bar, top, x_bar, bottom), fill=(220, 0, 0), width=2)
    draw.text((8, bottom - 10), f"{y_min:.3g}", fill=(80, 80, 80))
    draw.text((8, top - 4), f"{y_max:.3g}", fill=(80, 80, 80))
    if names:
        legend = "  ".join(str(name) for name in list(names)[: min(data.shape[1], 4)])
        draw.text((left, height - 22), legend, fill=(70, 70, 70))
    return image


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class WebDatasetEditor:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.dataset_root: Path | None = None
        self.dataset: Any | None = None
        self.info: dict[str, Any] = {}
        self.features_info: dict[str, dict] = {}
        self.available_features: list[str] = []
        self.deleted_features: set[str] = set()
        self.deleted_episodes: set[int] = set()
        self.episode_cache: dict[int, EpisodeData] = {}
        self.frame_item_cache: dict[int, dict[str, Any]] = {}
        self.current_episode = 0
        self.current_frame = 0
        self.output_name = ""
        self.status = "Select a dataset"

    def datasets(self) -> list[str]:
        return [path.name for path in list_dataset_dirs()]

    def select_dataset(self, name: str) -> dict[str, Any]:
        with self.lock:
            dataset_root = DATASETS_DIR / name
            self.status = f"Loading {name} ..."
            self.info = load_info(dataset_root)
            dataset_class = get_lerobot_dataset_class()
            self.dataset = dataset_class(f"local/{name}", root=dataset_root, video_backend="pyav")
            self.dataset_root = dataset_root
            self.features_info = {
                feature_name: normalize_feature_spec(feature_info)
                for feature_name, feature_info in self.info.get("features", {}).items()
            }
            self.available_features = user_features(self.features_info)
            self.deleted_features.clear()
            self.deleted_episodes.clear()
            self.episode_cache.clear()
            self.frame_item_cache.clear()
            self.current_episode = 0
            self.current_frame = 0
            self.output_name = infer_output_name(name)
            self.load_current_episode()
            self.status = "Ready"
            return self.state()

    def state(self) -> dict[str, Any]:
        total = 0 if self.dataset is None else int(self.dataset.meta.total_episodes)
        data = self.current_data()
        size = 0 if data is None else data.size
        features = [name for name in self.available_features if name not in self.deleted_features]
        return {
            "datasets": self.datasets(),
            "selectedDataset": None if self.dataset_root is None else self.dataset_root.name,
            "features": features,
            "featureInfo": {
                name: {
                    "dtype": self.features_info.get(name, {}).get("dtype"),
                    "names": self.features_info.get(name, {}).get("names"),
                }
                for name in features
            },
            "episode": self.current_episode,
            "totalEpisodes": total,
            "deletedEpisodes": sorted(self.deleted_episodes),
            "frame": self.current_frame,
            "frames": size,
            "selectionStart": 0,
            "selectionEnd": max(size - 1, 0),
            "outputName": self.output_name,
            "status": self.status,
        }

    def current_data(self) -> EpisodeData | None:
        return self.episode_cache.get(self.current_episode)

    def valid_episode_count(self) -> int:
        return 0 if self.dataset is None else int(self.dataset.meta.total_episodes)

    def load_current_episode(self) -> None:
        if self.dataset is None:
            return
        if self.current_episode in self.deleted_episodes:
            return
        if self.current_episode not in self.episode_cache:
            self.status = f"Loading episode {self.current_episode} ..."
            self.episode_cache[self.current_episode] = self.read_episode(self.current_episode)
        data = self.episode_cache[self.current_episode]
        self.current_frame = max(0, min(self.current_frame, max(data.size - 1, 0)))

    def read_episode(self, episode_index: int) -> EpisodeData:
        assert self.dataset is not None
        ep_meta = self.dataset.meta.episodes[episode_index]
        from_idx = int(ep_meta["dataset_from_index"])
        to_idx = int(ep_meta["dataset_to_index"])
        features = [
            name
            for name in self.available_features
            if name not in self.deleted_features and self.features_info[name].get("dtype") not in {"video", "image"}
        ]
        data = {feature: [] for feature in features}
        tasks: list[str] = []
        for idx in range(from_idx, to_idx):
            item = self.dataset.hf_dataset[idx]
            task_index = int(np.asarray(tensor_to_numpy(item["task_index"])).reshape(-1)[0])
            tasks.append(str(self.dataset.meta.tasks.iloc[task_index].name))
            for feature in features:
                if feature not in item:
                    continue
                data[feature].append(normalize_numeric(item[feature]))
        return EpisodeData(data, tasks)

    def episode_dataset_index(self, frame: int) -> int:
        assert self.dataset is not None
        ep_meta = self.dataset.meta.episodes[self.current_episode]
        from_idx = int(ep_meta["dataset_from_index"])
        to_idx = int(ep_meta["dataset_to_index"])
        return max(from_idx, min(from_idx + frame, to_idx - 1))

    def get_decoded_frame_item(self, dataset_index: int) -> dict[str, Any]:
        if dataset_index not in self.frame_item_cache:
            assert self.dataset is not None
            self.frame_item_cache[dataset_index] = self.dataset[dataset_index]
        return self.frame_item_cache[dataset_index]

    def ensure_current_episode_fully_loaded(self) -> EpisodeData | None:
        data = self.current_data()
        if data is None or self.dataset is None:
            return data
        missing_visual_features = [
            name
            for name in self.available_features
            if name not in self.deleted_features
            and self.features_info[name].get("dtype") in {"video", "image"}
            and name not in data.features
        ]
        if not missing_visual_features:
            return data

        self.status = f"Loading episode {self.current_episode} videos ..."
        ep_meta = self.dataset.meta.episodes[self.current_episode]
        from_idx = int(ep_meta["dataset_from_index"])
        to_idx = int(ep_meta["dataset_to_index"])
        for feature in missing_visual_features:
            data.features[feature] = []
        for dataset_index in range(from_idx, to_idx):
            item = self.get_decoded_frame_item(dataset_index)
            for feature in missing_visual_features:
                data.features[feature].append(normalize_image(item[feature]))
        return data

    def change_episode(self, delta: int) -> dict[str, Any]:
        with self.lock:
            if self.dataset is None:
                return self.state()
            total = self.valid_episode_count()
            idx = self.current_episode + delta
            while 0 <= idx < total and idx in self.deleted_episodes:
                idx += delta
            if 0 <= idx < total:
                self.current_episode = idx
                self.current_frame = 0
                self.load_current_episode()
            self.status = "Ready"
            return self.state()

    def set_frame(self, frame: int) -> dict[str, Any]:
        with self.lock:
            data = self.current_data()
            max_frame = 0 if data is None else max(data.size - 1, 0)
            self.current_frame = max(0, min(int(frame), max_frame))
            return self.state()

    def panel_png(self, feature: str, frame: int) -> bytes:
        with self.lock:
            data = self.current_data()
            if data is None:
                return image_to_png_bytes(Image.new("RGB", (520, 340), "white"))
            dtype = self.features_info.get(feature, {}).get("dtype")
            if dtype in {"video", "image"}:
                if feature in data.features:
                    values = data.features[feature]
                    if not values:
                        return image_to_png_bytes(Image.new("RGB", (520, 340), "white"))
                    frame = max(0, min(frame, len(values) - 1))
                    image_array = values[frame]
                else:
                    frame = max(0, min(frame, max(data.size - 1, 0)))
                    item = self.get_decoded_frame_item(self.episode_dataset_index(frame))
                    image_array = normalize_image(item[feature])
                image = Image.fromarray(image_array).convert("RGB")
                canvas = Image.new("RGB", (520, 340), "white")
                image.thumbnail((520, 310), Image.Resampling.BILINEAR)
                canvas.paste(image, ((520 - image.width) // 2, 28 + (312 - image.height) // 2))
                ImageDraw.Draw(canvas).text((8, 8), f"{feature} frame {frame}", fill=(20, 20, 20))
                return image_to_png_bytes(canvas)
            if feature not in data.features:
                return image_to_png_bytes(Image.new("RGB", (520, 340), "white"))
            values = data.features[feature]
            if not values:
                return image_to_png_bytes(Image.new("RGB", (520, 340), "white"))
            frame = max(0, min(frame, len(values) - 1))
            if dtype == "float32":
                return image_to_png_bytes(
                    make_plot_image(values, frame, feature, self.features_info.get(feature, {}).get("names"))
                )
            return image_to_png_bytes(Image.new("RGB", (520, 340), "white"))

    def delete_feature(self, feature: str) -> dict[str, Any]:
        with self.lock:
            if feature:
                self.deleted_features.add(feature)
                self.available_features = [name for name in self.available_features if name != feature]
                for episode in self.episode_cache.values():
                    episode.features.pop(feature, None)
                self.status = f"Feature marked for deletion: {feature}"
            return self.state()

    def delete_episode(self) -> dict[str, Any]:
        with self.lock:
            if self.dataset is None:
                return self.state()
            self.deleted_episodes.add(self.current_episode)
            self.episode_cache.pop(self.current_episode, None)
            total = self.valid_episode_count()
            candidates = [idx for idx in range(total) if idx not in self.deleted_episodes]
            if candidates:
                self.current_episode = min(candidates, key=lambda idx: abs(idx - self.current_episode))
                self.current_frame = 0
                self.load_current_episode()
            self.status = "Episode marked for deletion"
            return self.state()

    def delete_range(self, start: int, end: int) -> dict[str, Any]:
        with self.lock:
            data = self.current_data()
            if data is None or data.size == 0:
                return self.state()
            start, end = self.clean_range(start, end, data.size)
            if start == 0 and end == data.size - 1:
                return self.delete_episode()
            data = self.ensure_current_episode_fully_loaded()
            if data is None:
                return self.state()
            keep = [idx for idx in range(data.size) if not (start <= idx <= end)]
            self.apply_index_mapping(data, keep)
            self.current_frame = min(start, max(data.size - 1, 0))
            self.status = f"Deleted frames {start}-{end}"
            return self.state()

    def resample(self, start: int, end: int, factor: float, whole: bool = False) -> dict[str, Any]:
        with self.lock:
            data = self.current_data()
            if data is None or data.size == 0:
                return self.state()
            if whole:
                start, end = 0, data.size - 1
            start, end = self.clean_range(start, end, data.size)
            data = self.ensure_current_episode_fully_loaded()
            if data is None:
                return self.state()
            self.resample_range(data, start, end, factor)
            self.current_frame = min(start, max(data.size - 1, 0))
            self.status = f"Resampled frames {start}-{end}"
            return self.state()

    @staticmethod
    def clean_range(start: int, end: int, size: int) -> tuple[int, int]:
        start = max(0, min(int(start), max(size - 1, 0)))
        end = max(0, min(int(end), max(size - 1, 0)))
        return (start, end) if start <= end else (end, start)

    @staticmethod
    def apply_index_mapping(data: EpisodeData, indices: list[int]) -> None:
        for feature, values in list(data.features.items()):
            data.features[feature] = [values[idx] for idx in indices]
        data.tasks = [data.tasks[idx] for idx in indices]

    def resample_range(self, data: EpisodeData, start: int, end: int, factor: float) -> None:
        factor = float(factor)
        if factor <= 0:
            raise ValueError("Length factor must be greater than 0.")
        old_len = end - start + 1
        new_len = max(1, int(round(old_len * factor)))
        positions = np.arange(new_len, dtype=np.float32) / factor
        positions = np.clip(positions, 0, old_len - 1)
        nearest = np.clip(np.rint(positions).astype(int), 0, old_len - 1)
        for feature, values in list(data.features.items()):
            dtype = self.features_info.get(feature, {}).get("dtype")
            before = values[:start]
            segment = values[start : end + 1]
            after = values[end + 1 :]
            if dtype == "float32":
                arr = np.stack(segment).astype(np.float32)
                src_x = np.arange(old_len, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[:, None]
                new_arr = np.stack(
                    [np.interp(positions, src_x, arr[:, dim]) for dim in range(arr.shape[1])],
                    axis=1,
                ).astype(np.float32)
                resampled = [row.reshape(segment[0].shape) for row in new_arr]
            else:
                resampled = [segment[idx].copy() for idx in nearest]
            data.features[feature] = before + resampled + after
        task_segment = data.tasks[start : end + 1]
        data.tasks = data.tasks[:start] + [task_segment[idx] for idx in nearest] + data.tasks[end + 1 :]

    def export(self, output: str) -> dict[str, Any]:
        with self.lock:
            if self.dataset is None:
                raise ValueError("No dataset selected.")
            output = output.strip()
            if not output:
                raise ValueError("Output name is empty.")
            output_root = DATASETS_DIR / output
            if output_root.exists():
                shutil.rmtree(output_root)
            self.output_name = output
            self._export_dataset(output, output_root)
            self.status = f"Exported: datasets/{output}"
            return self.state()

    def make_output_frame(self, data: EpisodeData, frame_index: int) -> dict[str, Any]:
        frame = {}
        for feature in self.available_features:
            if feature in self.deleted_features or feature not in data.features:
                continue
            frame[feature] = data.features[feature][frame_index]
        task = data.tasks[frame_index] if frame_index < len(data.tasks) and data.tasks[frame_index] else "task"
        frame["task"] = task
        return frame

    def _export_dataset(self, output: str, output_root: Path) -> None:
        assert self.dataset is not None
        source_features = {
            name: info
            for name, info in self.features_info.items()
            if name not in SYSTEM_FEATURES and name not in self.deleted_features
        }
        dataset_class = get_lerobot_dataset_class()
        output_dataset = dataset_class.create(
            repo_id=f"local/{output}",
            fps=int(self.info.get("fps", self.dataset.fps)),
            root=output_root,
            robot_type=self.info.get("robot_type"),
            use_videos=any(info.get("dtype") == "video" for info in source_features.values()),
            features=source_features,
            video_backend="pyav",
            batch_encoding_size=1,
        )
        total = self.valid_episode_count()
        exported = 0
        old_available = self.available_features
        self.available_features = list(source_features.keys())
        try:
            for episode_index in range(total):
                if episode_index in self.deleted_episodes:
                    continue
                data = self.episode_cache.get(episode_index)
                if data is None:
                    data = self.read_episode(episode_index)
                    self.episode_cache[episode_index] = data
                old_episode = self.current_episode
                self.current_episode = episode_index
                data = self.ensure_current_episode_fully_loaded()
                self.current_episode = old_episode
                if data.size == 0:
                    continue
                for frame_index in range(data.size):
                    output_dataset.add_frame(self.make_output_frame(data, frame_index))
                output_dataset.save_episode()
                exported += 1
        finally:
            self.available_features = old_available
        output_dataset.finalize()
        if exported == 0:
            raise ValueError("No episodes were exported.")


WEB_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeRobot Dataset Editor</title>
<style>
body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f7f8;color:#17202a}
.toolbar{display:flex;gap:8px;align-items:center;padding:10px;background:#fff;border-bottom:1px solid #d7dce1;position:sticky;top:0;z-index:2}
select,input,button{font:inherit;padding:5px 8px}
button{cursor:pointer}.status{margin-left:auto;color:#52606d}
.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;padding:8px;height:calc(100vh - 185px);box-sizing:border-box}
.panel{background:#fff;border:1px solid #d7dce1;display:flex;flex-direction:column;min-width:0}
.panel select{margin:6px}.panel img{width:100%;height:100%;object-fit:contain;min-height:0}
.timeline{background:#fff;border-top:1px solid #d7dce1;padding:10px;display:grid;grid-template-columns:110px 1fr 120px;gap:8px;align-items:center}
.actions{grid-column:1/4;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.anchor{display:inline-flex;align-items:center;gap:4px;background:#eef2f5;border:1px solid #ccd3da;border-radius:4px;padding:4px 6px}
.anchor button{padding:0 5px}
input[type=range]{width:100%}
</style>
</head>
<body>
<div class="toolbar">
  <label>Dataset</label><select id="dataset"></select>
  <button id="prev">&lt;</button><button id="next">&gt;</button><span id="episode">episode -/-</span>
  <button id="deleteEpisode">Delete episode</button>
  <label>Feature</label><select id="deleteFeature"></select><button id="deleteFeatureBtn">Delete feature</button>
  <label>Output</label><input id="output" size="28"><button id="export">Export</button>
  <span class="status" id="status"></span>
</div>
<div class="panels">
  <div class="panel"><select id="feature0"></select><img id="panel0"></div>
  <div class="panel"><select id="feature1"></select><img id="panel1"></div>
  <div class="panel"><select id="feature2"></select><img id="panel2"></div>
</div>
<div class="timeline">
  <label>Frame</label><input id="frame" type="range" min="0" max="0" step="1"><span id="frameLabel">0 / 0</span>
  <div class="actions">
    <button id="setStart">Set Start</button>
    <button id="setEnd">Set End</button>
    <span class="anchor">Start <strong id="startLabel">unset</strong><button id="clearStart" title="Clear start anchor">x</button></span>
    <span class="anchor">End <strong id="endLabel">unset</strong><button id="clearEnd" title="Clear end anchor">x</button></span>
    <button id="all">Select all</button>
    <button id="deleteRange">Delete selected range</button>
    <label>Length factor</label><input id="factor" value="1.5" size="6">
    <button id="speedRange">Change selected speed</button>
    <button id="speedEpisode">Change episode speed</button>
  </div>
</div>
<script>
let state=null;
let anchorStart=null;
let anchorEnd=null;
let panelGeneration=0;
const ids = x => document.getElementById(x);
async function api(path, body=null){
  const opt = body ? {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {};
  const r = await fetch(path,opt);
  if(!r.ok){ throw new Error(await r.text()); }
  return await r.json();
}
async function busy(message, fn){
  ids('status').textContent = message;
  await new Promise(requestAnimationFrame);
  try {
    return await fn();
  } catch (e) {
    ids('status').textContent = 'Error';
    throw e;
  }
}
function fill(sel, values, keep){
  const old = keep ?? sel.value;
  sel.innerHTML = '';
  values.forEach(v => { const o=document.createElement('option'); o.value=v; o.textContent=v; sel.appendChild(o); });
  if(values.includes(old)) sel.value=old;
}
function update(s){
  state=s;
  fill(ids('dataset'), s.datasets, s.selectedDataset);
  fill(ids('deleteFeature'), s.features, ids('deleteFeature').value);
  for(let i=0;i<3;i++) fill(ids('feature'+i), s.features, ids('feature'+i).value || s.features[Math.min(i,s.features.length-1)]);
  ids('episode').textContent = `episode ${s.episode} / ${Math.max(s.totalEpisodes-1,0)}`;
  ids('output').value = s.outputName || ids('output').value;
  ids('status').textContent = s.status || '';
  const max = Math.max((s.frames||0)-1,0);
  ids('frame').max=max;
  ids('frame').value=Math.min(s.frame||0,max);
  if(anchorStart !== null) anchorStart = Math.min(anchorStart, max);
  if(anchorEnd !== null) anchorEnd = Math.min(anchorEnd, max);
  ids('frameLabel').textContent = `${ids('frame').value} / ${max}`;
  renderAnchors();
  refreshPanels();
}
function currentRange(){
  const max = +(ids('frame').max || 0);
  const start = anchorStart === null ? 0 : anchorStart;
  const end = anchorEnd === null ? max : anchorEnd;
  return start <= end ? [start, end] : [end, start];
}
function renderAnchors(){
  ids('startLabel').textContent = anchorStart === null ? 'unset' : anchorStart;
  ids('endLabel').textContent = anchorEnd === null ? 'unset' : anchorEnd;
}
function clearAnchors(){
  anchorStart=null;
  anchorEnd=null;
  renderAnchors();
}
function refreshPanels(){
  if(!state || !state.features.length) return;
  const generation = ++panelGeneration;
  const frame = ids('frame').value;
  for(let i=0;i<3;i++){
    const f = ids('feature'+i).value;
    const img = ids('panel'+i);
    const url = `/api/panel.png?feature=${encodeURIComponent(f)}&frame=${frame}&slot=${i}&t=${Date.now()}`;
    fetch(url)
      .then(r => r.blob())
      .then(blob => {
        if (generation !== panelGeneration) return;
        const old = img.dataset.objectUrl;
        const objectUrl = URL.createObjectURL(blob);
        img.dataset.objectUrl = objectUrl;
        img.src = objectUrl;
        if (old) URL.revokeObjectURL(old);
      })
      .catch(() => {});
  }
}
async function loadInitial(){ update(await api('/api/state')); }
ids('dataset').onchange=async()=>{clearAnchors(); update(await busy('Loading dataset ...',()=>api('/api/select',{name:ids('dataset').value})));};
ids('prev').onclick=async()=>{clearAnchors(); update(await busy('Loading episode ...',()=>api('/api/episode',{delta:-1})));};
ids('next').onclick=async()=>{clearAnchors(); update(await busy('Loading episode ...',()=>api('/api/episode',{delta:1})));};
ids('frame').oninput=async()=>{ ids('frameLabel').textContent=`${ids('frame').value} / ${ids('frame').max}`; await api('/api/frame',{frame:+ids('frame').value}); refreshPanels(); };
for(let i=0;i<3;i++) ids('feature'+i).onchange=refreshPanels;
ids('setStart').onclick=()=>{anchorStart=+ids('frame').value; renderAnchors();};
ids('setEnd').onclick=()=>{anchorEnd=+ids('frame').value; renderAnchors();};
ids('clearStart').onclick=()=>{anchorStart=null; renderAnchors();};
ids('clearEnd').onclick=()=>{anchorEnd=null; renderAnchors();};
ids('all').onclick=()=>{anchorStart=0; anchorEnd=+ids('frame').max; renderAnchors();};
ids('deleteFeatureBtn').onclick=async()=>{if(confirm('Delete feature?')) update(await api('/api/delete_feature',{feature:ids('deleteFeature').value}));};
ids('deleteEpisode').onclick=async()=>{if(confirm('Delete episode?')){clearAnchors(); update(await busy('Loading episode ...',()=>api('/api/delete_episode',{})));}};
ids('deleteRange').onclick=async()=>{const [start,end]=currentRange(); if(confirm(`Delete frames ${start}-${end}?`)){clearAnchors(); update(await busy('Loading episode videos ...',()=>api('/api/delete_range',{start,end})));}};
ids('speedRange').onclick=async()=>{const [start,end]=currentRange(); clearAnchors(); update(await busy('Loading episode videos ...',()=>api('/api/resample',{start,end,factor:+ids('factor').value,whole:false})));};
ids('speedEpisode').onclick=async()=>{clearAnchors(); update(await busy('Loading episode videos ...',()=>api('/api/resample',{factor:+ids('factor').value,whole:true})));};
ids('export').onclick=async()=>{update(await busy('Exporting ...',()=>api('/api/export',{output:ids('output').value})));};
loadInitial().catch(e=>alert(e.message));
</script>
</body>
</html>
"""


def make_web_handler(editor: WebDatasetEditor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def send_bytes(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
            self.send_bytes(json.dumps(data).encode("utf-8"), "application/json", status)

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self.send_bytes(WEB_HTML.encode("utf-8"), "text/html; charset=utf-8")
                elif parsed.path == "/api/state":
                    self.send_json(editor.state())
                elif parsed.path == "/api/panel.png":
                    query = parse_qs(parsed.query)
                    feature = query.get("feature", [""])[0]
                    frame = int(query.get("frame", ["0"])[0])
                    self.send_bytes(editor.panel_png(feature, frame), "image/png")
                else:
                    self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:
            try:
                body = self.read_json()
                if self.path == "/api/select":
                    self.send_json(editor.select_dataset(str(body.get("name", ""))))
                elif self.path == "/api/episode":
                    self.send_json(editor.change_episode(int(body.get("delta", 0))))
                elif self.path == "/api/frame":
                    self.send_json(editor.set_frame(int(body.get("frame", 0))))
                elif self.path == "/api/delete_feature":
                    self.send_json(editor.delete_feature(str(body.get("feature", ""))))
                elif self.path == "/api/delete_episode":
                    self.send_json(editor.delete_episode())
                elif self.path == "/api/delete_range":
                    self.send_json(editor.delete_range(int(body.get("start", 0)), int(body.get("end", 0))))
                elif self.path == "/api/resample":
                    self.send_json(
                        editor.resample(
                            int(body.get("start", 0)),
                            int(body.get("end", 0)),
                            float(body.get("factor", 1.0)),
                            bool(body.get("whole", False)),
                        )
                    )
                elif self.path == "/api/export":
                    self.send_json(editor.export(str(body.get("output", ""))))
                else:
                    self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def run_web(host: str = "127.0.0.1", port: int = 8765) -> None:
    editor = WebDatasetEditor()
    server = ThreadingHTTPServer((host, port), make_web_handler(editor))
    url = f"http://{host}:{server.server_port}"
    print(f"Dataset editor: {url}", flush=True)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_web(args.host, args.port)


if __name__ == "__main__":
    main()
