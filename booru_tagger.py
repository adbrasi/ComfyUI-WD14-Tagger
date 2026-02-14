import csv
import gc
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import comfy.utils
import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

import folder_paths
from .pysssss import get_ext_dir, get_extension_config, log

try:
    import timm
    import torch
    import torchvision.transforms as transforms
except Exception:  # pragma: no cover - optional at import time
    timm = None
    torch = None
    transforms = None


DEFAULT_WD14_REPO_PREFIX = "SmilingWolf"
DEFAULT_CAMIE_REPO = "Camais03/camie-tagger-v2"
DEFAULT_PIXAI_REPO = "pixai-labs/pixai-tagger-v0.9"

WD14_CSV_FILE = "selected_tags.csv"
WD14_ONNX_FILE = "model.onnx"

CAMIE_ONNX_FILE = "camie-tagger-v2.onnx"
CAMIE_META_FILE = "camie-tagger-v2-metadata.json"

PIXAI_PTH_FILE = "model_v0.9.pth"
PIXAI_TAGS_JSON_FILE = "tags_v0.9_13k.json"
PIXAI_CHAR_IP_MAP_FILE = "char_ip_map.json"


if "wd14_tagger" in folder_paths.folder_names_and_paths:
    MODELS_DIR = folder_paths.get_folder_paths("wd14_tagger")[0]
    os.makedirs(MODELS_DIR, exist_ok=True)
else:
    MODELS_DIR = get_ext_dir("models", mkdir=True)

CONFIG = get_extension_config()
DEFAULTS = {
    "model": "wd-v1-4-moat-tagger-v2",
    "threshold": 0.35,
    "character_threshold": 0.85,
    "exclude_tags": "",
    "replace_underscore": False,
    "ortProviders": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}
DEFAULTS.update(CONFIG.get("settings", {}))

KNOWN_WD14_MODELS = list(CONFIG.get("models", {}).keys())

WD14_SESSION_CACHE: Dict[str, Tuple[ort.InferenceSession, str, str, int]] = {}
WD14_TAG_CACHE: Dict[str, Tuple[List[str], List[int], List[int], List[int]]] = {}
CAMIE_CACHE: Dict[str, Tuple[ort.InferenceSession, str, Dict[int, str], Dict[str, str], int]] = {}
PIXAI_CACHE: Dict[str, Tuple[torch.nn.Module, Dict[int, str], int, int, Dict[str, List[str]], str]] = {}


def _cleanup_memory() -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_ort_providers() -> List[str]:
    available = ort.get_available_providers()
    preferred = []
    for provider in DEFAULTS["ortProviders"]:
        if provider in available:
            preferred.append(provider)
    if "CPUExecutionProvider" not in preferred:
        preferred.append("CPUExecutionProvider")
    return preferred


def _normalize_models(selected_models: str) -> List[str]:
    aliases = {
        "wd": "wd14",
        "wd14": "wd14",
        "camie": "camie",
        "pixai": "pixai",
    }
    out: List[str] = []
    for token in selected_models.split(","):
        model = aliases.get(token.strip().lower())
        if model and model not in out:
            out.append(model)
    return out or ["wd14"]


def _tags_exclude_set(exclude_tags: str) -> set:
    return {tag.strip().lower() for tag in exclude_tags.split(",") if tag.strip()}


def _tensor_to_pil_batch(image_tensor) -> List[Image.Image]:
    tensor = image_tensor.detach().cpu().numpy()
    tensor = np.clip(tensor * 255.0, 0, 255).astype(np.uint8)
    return [Image.fromarray(tensor[idx]) for idx in range(tensor.shape[0])]


def _iterate_batches(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _ensure_wd14_assets(model_name: str, force_download: bool) -> Tuple[str, str]:
    onnx_path = os.path.join(MODELS_DIR, f"{model_name}.onnx")
    csv_path = os.path.join(MODELS_DIR, f"{model_name}.csv")

    if not force_download and os.path.exists(onnx_path) and os.path.exists(csv_path):
        return onnx_path, csv_path

    if model_name not in KNOWN_WD14_MODELS:
        missing = []
        if not os.path.exists(onnx_path):
            missing.append(onnx_path)
        if not os.path.exists(csv_path):
            missing.append(csv_path)
        raise FileNotFoundError(
            "WD14 local model not found. For custom models, place both .onnx and .csv files manually. "
            f"Missing: {', '.join(missing)}"
        )

    repo_id = f"{DEFAULT_WD14_REPO_PREFIX}/{model_name}"
    cache_dir = os.path.join(MODELS_DIR, "_downloads", repo_id.replace("/", "_"))
    os.makedirs(cache_dir, exist_ok=True)

    log(f"Downloading WD14 model: {repo_id}", "INFO", True)
    downloaded_onnx = hf_hub_download(
        repo_id=repo_id,
        filename=WD14_ONNX_FILE,
        local_dir=cache_dir,
        force_download=force_download,
    )
    downloaded_csv = hf_hub_download(
        repo_id=repo_id,
        filename=WD14_CSV_FILE,
        local_dir=cache_dir,
        force_download=force_download,
    )

    with open(downloaded_onnx, "rb") as src, open(onnx_path, "wb") as dst:
        dst.write(src.read())
    with open(downloaded_csv, "rb") as src, open(csv_path, "wb") as dst:
        dst.write(src.read())

    return onnx_path, csv_path


def _load_wd14_tags(csv_path: str, replace_underscore: bool) -> Tuple[List[str], List[int], List[int], List[int]]:
    cache_key = f"{csv_path}:{replace_underscore}"
    if cache_key in WD14_TAG_CACHE:
        return WD14_TAG_CACHE[cache_key]

    tags: List[str] = []
    rating_indices: List[int] = []
    general_indices: List[int] = []
    character_indices: List[int] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for idx, row in enumerate(reader):
            tag = row[1].replace("_", " ") if replace_underscore else row[1]
            category = row[2]
            tags.append(tag)
            if category == "9":
                rating_indices.append(idx)
            elif category == "0":
                general_indices.append(idx)
            elif category == "4":
                character_indices.append(idx)

    WD14_TAG_CACHE[cache_key] = (tags, rating_indices, general_indices, character_indices)
    return WD14_TAG_CACHE[cache_key]


def _get_wd14_session(model_name: str, onnx_path: str) -> Tuple[ort.InferenceSession, str, str, int]:
    cache_key = f"{model_name}:{onnx_path}"
    if cache_key in WD14_SESSION_CACHE:
        return WD14_SESSION_CACHE[cache_key]

    providers = _get_ort_providers()
    model = onnx.load(onnx_path)
    input_name = model.graph.input[0].name
    try:
        input_size = int(model.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
    except Exception:
        input_size = 448
    del model

    session = ort.InferenceSession(onnx_path, providers=providers)
    output_name = session.get_outputs()[0].name
    WD14_SESSION_CACHE[cache_key] = (session, input_name, output_name, input_size)
    return WD14_SESSION_CACHE[cache_key]


def _preprocess_wd14(img: Image.Image, size: int) -> np.ndarray:
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        img = img.convert("RGBA")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    ratio = float(size) / max(img.size)
    new_size = (max(1, int(img.size[0] * ratio)), max(1, int(img.size[1] * ratio)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    square = Image.new("RGB", (size, size), (255, 255, 255))
    square.paste(img, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))

    arr = np.array(square, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB -> BGR
    return arr


def _run_wd14(
    images: List[Image.Image],
    model_name: str,
    batch_size: int,
    threshold: float,
    character_threshold: float,
    replace_underscore: bool,
    include_rating: bool,
    force_download: bool,
) -> List[List[str]]:
    onnx_path, csv_path = _ensure_wd14_assets(model_name, force_download)
    if force_download:
        WD14_SESSION_CACHE.pop(f"{model_name}:{onnx_path}", None)
    tags, rating_indices, general_indices, character_indices = _load_wd14_tags(csv_path, replace_underscore)
    session, input_name, output_name, input_size = _get_wd14_session(model_name, onnx_path)

    outputs: List[List[str]] = []
    pbar = comfy.utils.ProgressBar(len(images))
    for batch in _iterate_batches(images, batch_size):
        batch_arr = np.stack([_preprocess_wd14(img, input_size) for img in batch]).astype(np.float32)
        probs = session.run([output_name], {input_name: batch_arr})[0]

        for row in probs:
            general_tags = [tags[i] for i in general_indices if row[i] >= threshold]
            character_tags = [tags[i] for i in character_indices if row[i] >= character_threshold]
            sample_tags = character_tags + general_tags

            if include_rating and rating_indices:
                rating_scores = [(i, row[i]) for i in rating_indices]
                best_rating_idx = max(rating_scores, key=lambda x: x[1])[0]
                sample_tags.insert(0, tags[best_rating_idx])

            outputs.append(sample_tags)
            pbar.update(1)

    return outputs


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _ensure_camie_assets(repo_id: str, force_download: bool) -> Tuple[str, str]:
    model_dir = os.path.join(MODELS_DIR, repo_id.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, CAMIE_ONNX_FILE)
    meta_path = os.path.join(model_dir, CAMIE_META_FILE)

    if force_download or not (os.path.exists(onnx_path) and os.path.exists(meta_path)):
        log(f"Downloading Camie model: {repo_id}", "INFO", True)
        hf_hub_download(repo_id=repo_id, filename=CAMIE_ONNX_FILE, local_dir=model_dir, force_download=force_download)
        hf_hub_download(repo_id=repo_id, filename=CAMIE_META_FILE, local_dir=model_dir, force_download=force_download)

    return onnx_path, meta_path


def _load_camie_meta(meta_path: str) -> Tuple[Dict[int, str], Dict[str, str], int]:
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    tag_mapping = metadata["dataset_info"]["tag_mapping"]
    idx_to_tag = {int(k): v for k, v in tag_mapping["idx_to_tag"].items()}
    tag_to_category = tag_mapping["tag_to_category"]
    img_size = int(metadata.get("model_info", {}).get("img_size", 448))
    return idx_to_tag, tag_to_category, img_size


def _preprocess_imagenet(img: Image.Image, img_size: int) -> np.ndarray:
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        img = img.convert("RGBA")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    width, height = img.size
    aspect_ratio = width / max(1, height)
    if aspect_ratio > 1:
        new_width = img_size
        new_height = max(1, int(new_width / aspect_ratio))
    else:
        new_height = img_size
        new_width = max(1, int(new_height * aspect_ratio))

    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    padded = Image.new("RGB", (img_size, img_size), (124, 116, 104))
    padded.paste(resized, ((img_size - new_width) // 2, (img_size - new_height) // 2))

    arr = np.array(padded).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)


def _get_camie_session(repo_id: str, force_download: bool) -> Tuple[ort.InferenceSession, str, Dict[int, str], Dict[str, str], int]:
    if repo_id in CAMIE_CACHE and not force_download:
        return CAMIE_CACHE[repo_id]

    onnx_path, meta_path = _ensure_camie_assets(repo_id, force_download)
    idx_to_tag, tag_to_category, img_size = _load_camie_meta(meta_path)

    providers = _get_ort_providers()
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    CAMIE_CACHE[repo_id] = (session, input_name, idx_to_tag, tag_to_category, img_size)
    return CAMIE_CACHE[repo_id]


def _run_camie(
    images: List[Image.Image],
    repo_id: str,
    batch_size: int,
    general_threshold: float,
    character_threshold: float,
    min_confidence: float,
    force_download: bool,
) -> List[List[str]]:
    session, input_name, idx_to_tag, tag_to_category, img_size = _get_camie_session(repo_id, force_download)

    outputs: List[List[str]] = []
    pbar = comfy.utils.ProgressBar(len(images))
    for batch in _iterate_batches(images, batch_size):
        arr = np.stack([_preprocess_imagenet(img, img_size) for img in batch]).astype(np.float32)
        raw = session.run(None, {input_name: arr})
        logits = raw[1] if len(raw) >= 2 else raw[0]
        probs = _sigmoid(logits)

        for row in probs:
            general_tags: List[str] = []
            character_tags: List[str] = []
            for idx, confidence in enumerate(row):
                if confidence < min_confidence:
                    continue
                tag = idx_to_tag.get(idx)
                if tag is None:
                    continue
                category = tag_to_category.get(tag, "general").lower()
                threshold = character_threshold if category == "character" else general_threshold
                if confidence < threshold:
                    continue
                if category == "character":
                    character_tags.append(tag)
                elif category != "rating":
                    general_tags.append(tag)
            outputs.append(character_tags + general_tags)
            pbar.update(1)

    return outputs


class PixAITaggingHead(torch.nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        if torch is None:
            return
        super().__init__()
        self.head = torch.nn.Sequential(torch.nn.Linear(input_dim, num_classes))

    def forward(self, x):
        logits = self.head(x)
        return torch.sigmoid(logits)


def _require_pixai_deps() -> None:
    if torch is None or timm is None or transforms is None:
        raise RuntimeError(
            "PixAI requires torch, torchvision and timm. Install dependencies from requirements.txt"
        )


def _normalize_hf_token(hf_token: str) -> Optional[str]:
    token = hf_token.strip() if hf_token else ""
    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        return token

    env_token = (
        os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    if env_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = env_token
    return env_token


def _ensure_pixai_assets(repo_id: str, hf_token: Optional[str], force_download: bool) -> Tuple[str, str, str]:
    model_dir = os.path.join(MODELS_DIR, repo_id.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)

    required_files = [PIXAI_PTH_FILE, PIXAI_TAGS_JSON_FILE, PIXAI_CHAR_IP_MAP_FILE]
    for filename in required_files:
        file_path = os.path.join(model_dir, filename)
        if force_download or not os.path.exists(file_path):
            log(f"Downloading PixAI asset: {repo_id}/{filename}", "INFO", True)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=model_dir,
                force_download=force_download,
                token=hf_token,
            )

    return (
        os.path.join(model_dir, PIXAI_PTH_FILE),
        os.path.join(model_dir, PIXAI_TAGS_JSON_FILE),
        os.path.join(model_dir, PIXAI_CHAR_IP_MAP_FILE),
    )


def _pixai_pil_to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        img.load()
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    if img.mode == "P":
        return _pixai_pil_to_rgb(img.convert("RGBA"))
    return img.convert("RGB")


def _pixai_transform():
    _require_pixai_deps()
    return transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _load_pixai_maps(tags_path: str, ip_map_path: str) -> Tuple[Dict[int, str], int, int, Dict[str, List[str]]]:
    with open(tags_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    tag_map = payload["tag_map"]
    split = payload["tag_split"]

    idx_to_tag = {int(v): k for k, v in tag_map.items()}
    gen_count = int(split["gen_tag_count"])
    char_count = int(split["character_tag_count"])

    with open(ip_map_path, "r", encoding="utf-8") as f:
        ip_map = json.load(f)

    return idx_to_tag, gen_count, char_count, ip_map


def _resolve_pixai_device(device: str) -> str:
    _require_pixai_deps()
    if device == "auto":
        if torch.cuda.is_available():
            try:
                torch.zeros(1).to("cuda")
                return "cuda"
            except Exception:
                return "cpu"
        return "cpu"
    return device


def _build_pixai_model(weights_path: str, num_classes: int, device: str):
    _require_pixai_deps()
    encoder = timm.create_model("hf_hub:SmilingWolf/wd-eva02-large-tagger-v3", pretrained=False)
    encoder.reset_classifier(0)
    decoder = PixAITaggingHead(1024, num_classes)
    model = torch.nn.Sequential(encoder, decoder)
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _get_pixai_model(
    repo_id: str,
    hf_token: Optional[str],
    force_download: bool,
    device: str,
) -> Tuple[torch.nn.Module, Dict[int, str], int, int, Dict[str, List[str]], str]:
    cache_key = f"{repo_id}:{device}"
    if cache_key in PIXAI_CACHE and not force_download:
        return PIXAI_CACHE[cache_key]

    weights_path, tags_path, ip_map_path = _ensure_pixai_assets(repo_id, hf_token, force_download)
    idx_to_tag, gen_count, char_count, ip_map = _load_pixai_maps(tags_path, ip_map_path)

    model = _build_pixai_model(weights_path, len(idx_to_tag), device)
    PIXAI_CACHE[cache_key] = (model, idx_to_tag, gen_count, char_count, ip_map, device)
    return PIXAI_CACHE[cache_key]


def _run_pixai(
    images: List[Image.Image],
    repo_id: str,
    batch_size: int,
    general_threshold: float,
    character_threshold: float,
    pixai_no_ip: bool,
    pixai_device: str,
    hf_token: Optional[str],
    force_download: bool,
) -> List[List[str]]:
    _require_pixai_deps()
    device = _resolve_pixai_device(pixai_device)
    model, idx_to_tag, gen_count, char_count, char_ip_map, device = _get_pixai_model(
        repo_id, hf_token, force_download, device
    )
    transform = _pixai_transform()

    outputs: List[List[str]] = []
    pbar = comfy.utils.ProgressBar(len(images))
    for batch in _iterate_batches(images, batch_size):
        tensors = [transform(_pixai_pil_to_rgb(img)) for img in batch]
        batch_tensor = torch.stack(tensors)
        if device == "cuda":
            batch_tensor = batch_tensor.pin_memory().to(device, non_blocking=True)
        else:
            batch_tensor = batch_tensor.to(device)

        with torch.inference_mode():
            probs = model(batch_tensor)

        for row in probs:
            general_idx = (row[:gen_count] > general_threshold).nonzero(as_tuple=True)[0]
            character_idx = (row[gen_count : gen_count + char_count] > character_threshold).nonzero(as_tuple=True)[0]

            general_tags = [idx_to_tag[int(i)] for i in general_idx.cpu().tolist() if int(i) in idx_to_tag]
            character_tags = [
                idx_to_tag[int(i + gen_count)]
                for i in character_idx.cpu().tolist()
                if int(i + gen_count) in idx_to_tag
            ]

            ip_tags: List[str] = []
            if not pixai_no_ip:
                for character_tag in character_tags:
                    if character_tag in char_ip_map:
                        ip_tags.extend(char_ip_map[character_tag])
                ip_tags = sorted(set(ip_tags))

            outputs.append(character_tags + ip_tags + general_tags)
            pbar.update(1)

    return outputs


class BooruTagger:
    @classmethod
    def INPUT_TYPES(cls):
        installed_wd14 = [
            os.path.splitext(filename)[0]
            for filename in os.listdir(MODELS_DIR)
            if filename.endswith(".onnx") and os.path.exists(os.path.join(MODELS_DIR, os.path.splitext(filename)[0] + ".csv"))
        ]
        extra = [name for name in installed_wd14 if name not in KNOWN_WD14_MODELS]
        wd14_models = KNOWN_WD14_MODELS + extra
        if not wd14_models:
            wd14_models = [DEFAULTS["model"]]

        return {
            "required": {
                "image": ("IMAGE",),
                "selected_models": (
                    "STRING",
                    {"default": "wd14,camie,pixai", "multiline": False},
                ),
                "wd14_model": (wd14_models, {"default": DEFAULTS["model"]}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 256, "step": 1}),
                "wd14_threshold": ("FLOAT", {"default": DEFAULTS["threshold"], "min": 0.0, "max": 1.0, "step": 0.01}),
                "wd14_character_threshold": (
                    "FLOAT",
                    {"default": DEFAULTS["character_threshold"], "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "camie_threshold": ("FLOAT", {"default": 0.492, "min": 0.0, "max": 1.0, "step": 0.01}),
                "camie_character_threshold": ("FLOAT", {"default": 0.492, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pixai_threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pixai_character_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "wd14_separator": ("STRING", {"default": ", ", "multiline": False}),
                "camie_separator": ("STRING", {"default": ", ", "multiline": False}),
                "pixai_separator": ("STRING", {"default": ", ", "multiline": False}),
                "model_separator": ("STRING", {"default": " | ", "multiline": False}),
                "exclude_tags": ("STRING", {"default": DEFAULTS.get("exclude_tags", ""), "multiline": False}),
                "replace_underscore": ("BOOLEAN", {"default": bool(DEFAULTS.get("replace_underscore", False))}),
                "dedupe": ("BOOLEAN", {"default": True}),
                "include_rating": ("BOOLEAN", {"default": False}),
                "pixai_no_ip": ("BOOLEAN", {"default": False}),
                "force_download": ("BOOLEAN", {"default": False}),
                "pixai_device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "camie_repo_id": ("STRING", {"default": DEFAULT_CAMIE_REPO, "multiline": False}),
                "pixai_repo_id": ("STRING", {"default": DEFAULT_PIXAI_REPO, "multiline": False}),
                "hf_token": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "tag"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def tag(
        self,
        image,
        selected_models,
        wd14_model,
        batch_size,
        wd14_threshold,
        wd14_character_threshold,
        camie_threshold,
        camie_character_threshold,
        pixai_threshold,
        pixai_character_threshold,
        wd14_separator,
        camie_separator,
        pixai_separator,
        model_separator,
        exclude_tags,
        replace_underscore,
        dedupe,
        include_rating,
        pixai_no_ip,
        force_download,
        pixai_device,
        camie_repo_id,
        pixai_repo_id,
        hf_token,
    ):
        models = _normalize_models(selected_models)
        images = _tensor_to_pil_batch(image)
        exclude = _tags_exclude_set(exclude_tags)
        token = _normalize_hf_token(hf_token)

        results_by_model: Dict[str, List[List[str]]] = {}

        for model_name in models:
            if model_name == "wd14":
                results_by_model[model_name] = _run_wd14(
                    images=images,
                    model_name=wd14_model,
                    batch_size=batch_size,
                    threshold=wd14_threshold,
                    character_threshold=wd14_character_threshold,
                    replace_underscore=replace_underscore,
                    include_rating=include_rating,
                    force_download=force_download,
                )
            elif model_name == "camie":
                results_by_model[model_name] = _run_camie(
                    images=images,
                    repo_id=camie_repo_id,
                    batch_size=batch_size,
                    general_threshold=camie_threshold,
                    character_threshold=camie_character_threshold,
                    min_confidence=0.1,
                    force_download=force_download,
                )
            elif model_name == "pixai":
                results_by_model[model_name] = _run_pixai(
                    images=images,
                    repo_id=pixai_repo_id,
                    batch_size=batch_size,
                    general_threshold=pixai_threshold,
                    character_threshold=pixai_character_threshold,
                    pixai_no_ip=pixai_no_ip,
                    pixai_device=pixai_device,
                    hf_token=token,
                    force_download=force_download,
                )
            else:
                raise ValueError(f"Unknown model selected: {model_name}")

        separator_by_model = {
            "wd14": wd14_separator,
            "camie": camie_separator,
            "pixai": pixai_separator,
        }

        merged_strings: List[str] = []
        for idx in range(len(images)):
            blocks: List[str] = []
            merged_seen = set()

            for model_name in models:
                tags = results_by_model.get(model_name, [[]])[idx]
                if exclude:
                    tags = [tag for tag in tags if tag.lower() not in exclude]
                if dedupe:
                    tags = [tag for tag in tags if tag not in merged_seen]
                    merged_seen.update(tags)

                joined = separator_by_model[model_name].join(tags)
                if joined:
                    blocks.append(joined)

            merged_strings.append(model_separator.join(blocks))

        _cleanup_memory()
        return {"ui": {"tags": merged_strings}, "result": (merged_strings, image)}


NODE_CLASS_MAPPINGS = {
    "BooruTagger|pysssss": BooruTagger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BooruTagger|pysssss": "booru tagger",
}
