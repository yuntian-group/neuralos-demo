#!/usr/bin/env python3
# render_sessions.py
import os
import glob
import json
import ast
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import imageio.v2 as imageio

# Your utilities / model imports
from utils import initialize_model
from ldm.models.diffusion.ddpm import DDIMSampler  # LatentDiffusion is pulled via initialize_model configs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ----------------------------
# Defaults / constants
# ----------------------------
SCREEN_WIDTH = 512
SCREEN_HEIGHT = 384
LATENT_CHANNELS = 16  # matches your LATENT_DIMS first dim
TIMESTEPS_DEFAULT = 1000  # full DDPM chain unless overridden by model config

# Key name normalization (subset from your worker)
KEYMAPPING = {
    'arrowup': 'up',
    'arrowdown': 'down',
    'arrowleft': 'left',
    'arrowright': 'right',
    'meta': 'command',
    'contextmenu': 'apps',
    'control': 'ctrl',
}

# Valid keys copied from your worker (minus invalids)
KEYS = ['\t', '\n', '\r', ' ', '!', '"', '#', '$', '%', '&', "'", '(',
    ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
    '8', '9', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~',
    'accept', 'add', 'alt', 'altleft', 'altright', 'apps', 'backspace',
    'browserback', 'browserfavorites', 'browserforward', 'browserhome',
    'browserrefresh', 'browsersearch', 'browserstop', 'capslock', 'clear',
    'convert', 'ctrl', 'ctrlleft', 'ctrlright', 'decimal', 'del', 'delete',
    'divide', 'down', 'end', 'enter', 'esc', 'escape', 'execute', 'f1', 'f10',
    'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f2', 'f20',
    'f21', 'f22', 'f23', 'f24', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
    'final', 'fn', 'hanguel', 'hangul', 'hanja', 'help', 'home', 'insert', 'junja',
    'kana', 'kanji', 'launchapp1', 'launchapp2', 'launchmail',
    'launchmediaselect', 'left', 'modechange', 'multiply', 'nexttrack',
    'nonconvert', 'num0', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6',
    'num7', 'num8', 'num9', 'numlock', 'pagedown', 'pageup', 'pause', 'pgdn',
    'pgup', 'playpause', 'prevtrack', 'print', 'printscreen', 'prntscrn',
    'prtsc', 'prtscr', 'return', 'right', 'scrolllock', 'select', 'separator',
    'shift', 'shiftleft', 'shiftright', 'sleep', 'space', 'stop', 'subtract', 'tab',
    'up', 'volumedown', 'volumemute', 'volumeup', 'win', 'winleft', 'winright', 'yen',
    'command', 'option', 'optionleft', 'optionright']

INVALID_KEYS = ['f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'select', 'separator', 'execute']
VALID_KEYS = [k for k in KEYS if k not in INVALID_KEYS]
STOI = {k: i for i, k in enumerate(VALID_KEYS)}

# ----------------------------
# Helpers
# ----------------------------
def load_latent_stats(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    with open(path, 'r') as f:
        stats = json.load(f)
    mean = torch.tensor(stats['mean'], device=device)
    std = torch.tensor(stats['std'], device=device)
    return {'mean': mean, 'std': std}

def prepare_model(
    model_name: str,
    config: str,
    device: torch.device,
    screen_w: int,
    screen_h: int,
    latent_channels: int,
    latent_stats_path: str
):
    model = initialize_model(config, model_name).to(device)
    model.eval()
    stats = load_latent_stats(latent_stats_path, device)
    latent_dims = (latent_channels, screen_h // 8, screen_w // 8)

    # initial previous_frame latent (normalized zeros)
    padding = torch.zeros(*latent_dims, device=device).unsqueeze(0)
    padding = (padding - stats['mean'].view(1, -1, 1, 1)) / stats['std'].view(1, -1, 1, 1)
    return model, stats, latent_dims, padding

def clamp_xy(x: Optional[int], y: Optional[int], w: int, h: int) -> Tuple[int,int]:
    x = 0 if x is None else max(0, min(w-1, int(x)))
    y = 0 if y is None else max(0, min(h-1, int(y)))
    return x, y

def build_inputs(
    prev_latent: torch.Tensor,
    hidden_states: Optional[Any],
    x: int, y: int,
    left_click: bool, right_click: bool,
    keys_down: List[str],
    t: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    ki = torch.zeros(len(VALID_KEYS), dtype=torch.long, device=device)
    for k in keys_down:
        k = (k or '').lower()
        k = KEYMAPPING.get(k, k)
        if k in STOI:
            ki[STOI[k]] = 1
    d: Dict[str, torch.Tensor] = {
        'image_features': prev_latent.to(device),
        'is_padding': torch.BoolTensor([t == 0]).to(device),
        'x': torch.LongTensor([x]).unsqueeze(0).to(device),
        'y': torch.LongTensor([y]).unsqueeze(0).to(device),
        'is_leftclick': torch.BoolTensor([left_click]).unsqueeze(0).to(device),
        'is_rightclick': torch.BoolTensor([right_click]).unsqueeze(0).to(device),
        'key_events': ki
    }
    if hidden_states is not None:
        d['hidden_states'] = hidden_states
    return d

@torch.no_grad()
def step_model(
    model,
    inputs: Dict[str, torch.Tensor],
    latent_dims: Tuple[int,int,int],
    stats: Dict[str, torch.Tensor],
    device: torch.device,
    sampler_type: str,
    steps: int,
    ddim_discr_method: str,
    timesteps_full: int
):
    # Temporal encoder
    out_from_rnn, hidden_states = model.temporal_encoder.forward_step(inputs)

    # Sampling
    model.clip_denoised = False

    if sampler_type == 'rnn':
        sample_latent = out_from_rnn[:, :LATENT_CHANNELS]
    elif sampler_type == 'ddpm':
        if steps >= timesteps_full:
            sample_latent = model.p_sample_loop(
                cond={'c_concat': out_from_rnn},
                shape=[1, *latent_dims],
                return_intermediates=False,
                verbose=False
            )
        elif steps == 1:
            x = torch.randn([1, *latent_dims], device=device)
            t = torch.full((1,), timesteps_full - 1, device=device, dtype=torch.long)
            sample_latent = model.apply_model(x, t, {'c_concat': out_from_rnn})
        else:
            sampler = DDIMSampler(model)
            sample_latent, _ = sampler.sample(
                S=steps, conditioning={'c_concat': out_from_rnn},
                batch_size=1, shape=latent_dims, verbose=False
            )
    else:  # 'ddim'
        sampler = DDIMSampler(model)
        sample_latent, _ = sampler.sample(
            S=steps, conditioning={'c_concat': out_from_rnn},
            ddim_discretize=ddim_discr_method,
            batch_size=1, shape=latent_dims, verbose=False
        )

    # Decode
    sample = sample_latent * stats['std'].view(1, -1, 1, 1) + stats['mean'].view(1, -1, 1, 1)
    sample = model.decode_first_stage(sample).squeeze(0).clamp(-1, 1)
    img = ((sample[:3].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).astype(np.uint8)
    return sample_latent, hidden_states, img

def safe_literal_list(s: str) -> List[Tuple[str,str]]:
    s = (s or "").strip()
    if not s or s == "[]":
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    return []

def csv_iter_rows(csv_path: str):
    with open(csv_path, 'r') as f:
        _ = f.readline()  # header
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(",", 6)
            if len(parts) < 7:
                continue
            ts = float(parts[0])
            x = int(float(parts[2])); y = int(float(parts[3]))
            left = parts[4].strip().lower() == 'true'
            right = parts[5].strip().lower() == 'true'
            key_str = parts[6]
            yield ts, x, y, left, right, key_str

def median_dt_to_fps(timestamps: List[float], default_fps: int = 10) -> int:
    if len(timestamps) < 2:
        return default_fps
    dts = np.diff(np.array(timestamps))
    med = float(np.median(dts))
    if med <= 1e-6:
        return default_fps
    fps = max(1, int(round(1.0 / med)))
    return min(60, fps)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ---------- Teacher forcing helpers: frames in RAM + single-frame encode ----------
def _to_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = np.stack([frame]*3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    elif frame.shape[2] != 3:
        frame = frame[:, :, :3]
    if frame.shape[0] != SCREEN_HEIGHT or frame.shape[1] != SCREEN_WIDTH:
        frame = np.array(Image.fromarray(frame).resize((SCREEN_WIDTH, SCREEN_HEIGHT), Image.BILINEAR))
    return frame

def read_gt_frames_rgb_uint8(gt_mp4: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Read all frames from GT mp4 into RAM as RGB uint8 at model resolution."""
    try:
        reader = imageio.get_reader(gt_mp4)
    except Exception as e:
        print(f"[TF] Unable to open GT video: {gt_mp4} :: {e}")
        return []
    frames: List[np.ndarray] = []
    try:
        for i, frame in enumerate(reader):
            if max_frames is not None and i >= max_frames:
                break
            frames.append(_to_rgb_uint8(frame))
    except Exception as e:
        if frames:
            print(f"[TF] Partial read of GT video {gt_mp4}: kept {len(frames)} frames :: {e}")
        else:
            print(f"[TF] Failed to read {gt_mp4}: {e}")
            frames = []
    finally:
        try:
            reader.close()
        except Exception:
            pass
    return frames

@torch.no_grad()
def encode_frame_to_norm_latent(
    model,
    frame_rgb_uint8: np.ndarray,
    stats: Dict[str, torch.Tensor],
    device: torch.device
) -> torch.Tensor:
    """
    Single-frame encoder to normalized latent:
      - convert to [-1,1] tensor [1,3,H,W]
      - posterior = first_stage_model.encode(x) (or fallback)
      - z = posterior.sample()
      - z_norm = (z - mean) / std
      Returns CPU tensor [1, C, H/8, W/8]
    """
    # [-1, 1] normalization
    arr = (frame_rgb_uint8.astype(np.float32) / 127.5) - 1.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # Prefer first_stage_model.encode -> posterior
    if hasattr(model, "first_stage_model") and hasattr(model.first_stage_model, "encode"):
        posterior = model.first_stage_model.encode(x)
        z = posterior.sample()
    #elif hasattr(model, "encode_first_stage"):
    #    z = model.encode_first_stage(x)
    #elif hasattr(model, "encode"):
    #    posterior = model.encode(x)
    #    z = posterior.sample() if hasattr(posterior, "sample") else posterior
    else:
        assert False
        raise RuntimeError("No encoder found on model for teacher forcing.")

    z_norm = (z - stats['mean'].view(1, -1, 1, 1)) / stats['std'].view(1, -1, 1, 1)
    return z_norm.detach().cpu()

# ----------------------------
# Core rendering per CSV
# ----------------------------
def render_csv_to_mp4(
    csv_path: str,
    out_path: str,
    model,
    stats: Dict[str, torch.Tensor],
    latent_dims: Tuple[int,int,int],
    padding_latent: torch.Tensor,
    sampler_type: str,
    steps: int,
    timesteps_full: int,
    device: torch.device,
    ddim_discr_method: str,
    fps_override: Optional[int] = None,
    teacher_forcing: bool = False
):
    rows = list(csv_iter_rows(csv_path))
    if not rows:
        print(f"[WARN] No rows in {csv_path}, skipping.")
        return

    timestamps = [r[0] for r in rows]
    fps = fps_override if fps_override and fps_override > 0 else median_dt_to_fps(timestamps, default_fps=10)

    # Teacher forcing: preload all GT frames into CPU RAM
    gt_frames: List[np.ndarray] = []
    if teacher_forcing:
        gt_mp4 = os.path.splitext(csv_path)[0] + ".mp4"
        if os.path.exists(gt_mp4):
            gt_frames = read_gt_frames_rgb_uint8(gt_mp4, max_frames=len(rows))
            if not gt_frames:
                print(f"[TF] Falling back to autoregressive for {os.path.basename(csv_path)} (no GT frames).")
                teacher_forcing = False
        else:
            print(f"[TF] GT mp4 missing for {os.path.basename(csv_path)} -> {gt_mp4}. Falling back.")
            teacher_forcing = False

    down_keys = set()
    prev_latent = padding_latent
    hidden_states = None

    ensure_dir(os.path.dirname(out_path))
    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', quality=8)

    try:
        #import pdb; pdb.set_trace()
        for t_idx, (ts, x, y, l, r, key_str) in enumerate(rows):
            # Use GT previous-frame latent when TF is enabled and available
            if teacher_forcing and t_idx > 0 and (t_idx - 1) < len(gt_frames):
                prev_latent = encode_frame_to_norm_latent(model, gt_frames[t_idx - 1], stats, device)
            elif t_idx == 0:
                prev_latent = padding_latent

            # key events
            for ev_type, key_name in safe_literal_list(key_str):
                key_name = (key_name or "").lower()
                if key_name in KEYMAPPING:
                    key_name = KEYMAPPING[key_name]
                if ev_type == 'keydown':
                    down_keys.add(key_name)
                elif ev_type == 'keyup':
                    down_keys.discard(key_name)

            x, y = clamp_xy(x, y, SCREEN_WIDTH, SCREEN_HEIGHT)
            inputs = build_inputs(
                prev_latent, hidden_states, x, y, l, r,
                sorted(list(down_keys)), t_idx, device
            )

            sample_latent, hidden_states, frame_img = step_model(
                model, inputs, latent_dims, stats, device, ddim_discr_method=ddim_discr_method,
                sampler_type=sampler_type, steps=steps, timesteps_full=timesteps_full
            )

            # Only carry model output forward if not doing teacher forcing
            if not teacher_forcing:
                prev_latent = sample_latent

            writer.append_data(frame_img)
    finally:
        writer.close()

# ----------------------------
# Batch over train/test
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Render NeuralOS evaluation CSVs to MP4 using your model.")
    ap.add_argument("--eval-root", type=str, default="evaluation_frames",
                    help="Root folder that contains train/ and test/ with CSVs.")
    ap.add_argument("--split", type=str, default="both", choices=["train", "test", "both"],
                    help="Which split(s) to render.")
    ap.add_argument("--out-root", type=str, default="model_generated",
                    help="Output root folder (e.g., model_generated_ddim16 / model_generated_ddpm).")
    ap.add_argument("--model-name", type=str, default='yuntian-deng/computer-model-s-origunet-nospatial-online-x0-joint-onlineonly-222222k722n222-146k',
                    help="HuggingFace or local model name/path for initialize_model.")
    ap.add_argument("--config", type=str, default='config_final_model_origunet_nospatial_x0.yaml',
                    help="Config file used by initialize_model (e.g., config_final_model_origunet_nospatial_x0.yaml).")
    ap.add_argument("--latent-stats", type=str, default="latent_stats.json",
                    help="Path to latent_stats.json (with mean/std).")
    ap.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm", "rnn"],
                    help="Sampling strategy.")
    ap.add_argument("--steps", type=int, default=32,
                    help="Sampling steps (DDIM or accelerated DDPM). Use >=1000 to run full DDPM chain.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="cuda or cpu")
    ap.add_argument("--prefix", type=str, default='')
    ap.add_argument("--fps", type=int, default=15,
                    help="If >0, override FPS. Otherwise auto from median Δtimestamp.")
    ap.add_argument("--timesteps-full", type=int, default=TIMESTEPS_DEFAULT,
                    help="Total diffusion timesteps for full DDPM chain.")
    ap.add_argument("--teacher-forcing", action="store_true",
                    help="If set, condition each frame on the GT previous-frame latent instead of the model output.")
    args = ap.parse_args()
    assert args.teacher_forcing

    device = torch.device(args.device)
    ddim_discr_method = 'uniform'
    if args.prefix == '2_074k':
        print('using newer model')
        args.model_name = 'yuntian-deng/computer-model-s-origunet-nospatial-online-x0-joint-onlineonly-222222k722n2222-074k'
    if args.prefix == 'n_046k':
        print('using newer 1gpu model')
        args.model_name = 'yuntian-deng/computer-model-s-origunet-nospatial-online-x0-joint-onlineonly-222222k722n22nnn-046k'
    if args.prefix == 'distill_8':
        print ('using distill 8')
        ddim_discr_method = 'progressivedistillation_64'
        args.model_name = 'yuntian-deng/computer-model-distill-8'
    if args.prefix == 'distill_16':
        print ('using distill 16')
        ddim_discr_method = 'progressivedistillation_64'
        args.model_name = 'yuntian-deng/computer-model-distill-16'

    model, stats, latent_dims, padding_latent = prepare_model(
        model_name=args.model_name,
        config=args.config,
        device=device,
        screen_w=SCREEN_WIDTH,
        screen_h=SCREEN_HEIGHT,
        latent_channels=LATENT_CHANNELS,
        latent_stats_path=args.latent_stats
    )

    splits = ["train","test"] if args.split == "both" else [args.split]
    for split in splits:
        in_dir = os.path.join(args.eval_root, split)
        out_dir = os.path.join(args.out_root, split)
        ensure_dir(out_dir)

        csvs = sorted(glob.glob(os.path.join(in_dir, "*.csv")))
        csvs = [p for p in csvs if os.path.basename(p).startswith('0')]
        if not csvs:
            print(f"[WARN] No CSVs found in {in_dir}")
            continue

        print(f"Rendering {len(csvs)} files from {in_dir} -> {out_dir} "
              f"(sampler={args.sampler}, steps={args.steps}, teacher_forcing={args.teacher_forcing})")

        for csv_path in tqdm(csvs):
            base = os.path.splitext(os.path.basename(csv_path))[0]
            subdir = f'{args.sampler}_{args.steps}' + ('_tf' if args.teacher_forcing else '')
            if args.prefix != '':
                subdir = f'{args.prefix}_{subdir}'
            out_path = os.path.join(out_dir, subdir, f"{base}.mp4")

            render_csv_to_mp4(
                csv_path=csv_path,
                out_path=out_path,
                model=model,
                stats=stats,
                latent_dims=latent_dims,
                padding_latent=padding_latent,
                sampler_type=args.sampler,
                steps=args.steps,
                timesteps_full=args.timesteps_full,
                device=device,
                ddim_discr_method=ddim_discr_method,
                fps_override=args.fps,
                teacher_forcing=args.teacher_forcing
            )
    print("Done.")

if __name__ == "__main__":
    main()

