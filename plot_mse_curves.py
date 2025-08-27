#!/usr/bin/env python3
# plot_mse_curves.py
import os
import glob
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def list_settings(gen_root: str, split: str, include: Optional[List[str]]) -> List[str]:
    base = os.path.join(gen_root, split)
    if include:
        return include
    settings = []
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            if os.path.isdir(os.path.join(base, name)):
                settings.append(name)
    return settings


def prepare_frame(array: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Ensure frame is RGB uint8 and resized to target (H, W)."""
    h, w = array.shape[:2]
    # Convert to RGB 3ch
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.shape[2] == 4:
        array = array[:, :, :3]
    elif array.shape[2] != 3:
        array = array[:, :, :3]

    th, tw = target_hw
    if (h, w) != (th, tw):
        array = np.array(Image.fromarray(array).resize((tw, th), Image.BILINEAR))
    return array


def per_frame_mse(gen_path: str, gt_path: str, normalize_01: bool = True,
                  max_frames: int = 0) -> Optional[List[float]]:
    """
    Return list of MSEs for frames 0..min_len-1 between gen and gt videos.
    If normalize_01=True, MSE is computed on [0,1] floats; otherwise on uint8 [0..255].
    max_frames > 0 limits frames for quick runs.

    If an MP4 is still being written (or cannot be opened), return None to skip gracefully.
    If an error occurs mid-iteration, return any frames computed so far (or None if none).
    """
    try:
        gen_r = imageio.get_reader(gen_path)
        gt_r  = imageio.get_reader(gt_path)
    except Exception as e:
        print(f"[SKIP open] {e} :: {gen_path} vs {gt_path}")
        return None

    mses: List[float] = []
    count = 0
    try:
        for gen_frame, gt_frame in zip(gen_r, gt_r):
            if max_frames and count >= max_frames:
                break

            H, W = gt_frame.shape[0], gt_frame.shape[1]
            gen_frame = prepare_frame(gen_frame, (H, W))
            gt_frame  = prepare_frame(gt_frame,  (H, W))

            if normalize_01:
                a = gen_frame.astype(np.float32) / 255.0
                b = gt_frame.astype(np.float32) / 255.0
            else:
                a = gen_frame.astype(np.float32)
                b = gt_frame.astype(np.float32)

            mse = math.sqrt(float(np.mean((a - b) ** 2)))
            mses.append(mse)
            count += 1
    except Exception as e:
        if mses:
            print(f"[PARTIAL read] {e} :: {gen_path} vs {gt_path} (kept {len(mses)} frames)")
        else:
            print(f"[SKIP read] {e} :: {gen_path} vs {gt_path}")
            mses = None
    finally:
        try:
            gen_r.close()
            gt_r.close()
        except Exception:
            pass

    return mses


def accumulate_curves(gen_root: str, gt_root: str, split: str, setting: str,
                      normalize_01: bool, max_files: int, max_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate per-frame MSEs over all matched (gen, gt) videos for one (split, setting).
    Returns (cum_mean, per_frame_mean).
    """
    gen_dir = os.path.join(gen_root, split, setting)
    gt_dir  = os.path.join(gt_root,  split)

    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.mp4")))
    if max_files > 0:
        gen_files = gen_files[:max_files]

    if not gen_files:
        print(f"[{split}/{setting}] No generated MP4s found in {gen_dir}")
        return np.array([]), np.array([])

    sum_per_t: List[float] = []
    cnt_per_t: List[int]   = []
    matched = 0

    for gpath in tqdm(gen_files, desc=f"[{split}] {setting}", leave=False):
        base = os.path.splitext(os.path.basename(gpath))[0]
        gtpath = os.path.join(gt_dir, f"{base}.mp4")
        if not os.path.exists(gtpath):
            continue

        mses = per_frame_mse(gpath, gtpath, normalize_01=normalize_01, max_frames=max_frames)
        if not mses:
            continue

        matched += 1
        if len(sum_per_t) < len(mses):
            grow = len(mses) - len(sum_per_t)
            sum_per_t.extend([0.0] * grow)
            cnt_per_t.extend([0]   * grow)
        for t, v in enumerate(mses):
            sum_per_t[t] += v
            cnt_per_t[t] += 1

    if matched == 0 or sum(cnt_per_t) == 0:
        print(f"[{split}/{setting}] No matched pairs with frames; skipping.")
        return np.array([]), np.array([])

    per_frame_mean = np.array([
        (sum_per_t[t] / cnt_per_t[t]) if cnt_per_t[t] > 0 else np.nan
        for t in range(len(sum_per_t))
    ], dtype=np.float64)

    cum_num = np.cumsum([sum_per_t[t] for t in range(len(sum_per_t))], dtype=np.float64)
    cum_den = np.cumsum([cnt_per_t[t] for t in range(len(cnt_per_t))], dtype=np.float64)
    cum_mean = np.divide(cum_num, cum_den, out=np.full_like(cum_num, np.nan), where=cum_den > 0)

    print(f"[{split}/{setting}] matched files: {matched}, max frames used: {len(cum_mean)}")
    return cum_mean, per_frame_mean


def compute_baseline_curve(gt_root: str, split: str,
                           normalize_01: bool, max_pairs: int, max_frames: int) -> np.ndarray:
    """
    Baseline: compare GT runs 1_*, 2_*, ... against the corresponding 0_* file.
    Aggregate across all available pairs to produce a single baseline curve (cum_mean).
    """
    gt_dir = os.path.join(gt_root, split)
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.mp4")))
    if not gt_files:
        print(f"[{split}/baseline] No GT MP4s in {gt_dir}")
        return np.array([])

    # Build suffix -> {run_index: path}
    suffix_map: Dict[str, Dict[int, str]] = {}
    for p in gt_files:
        name = os.path.basename(p)
        if "_" not in name:
            continue
        idx_str, rest = name.split("_", 1)
        if not idx_str.isdigit():
            continue
        run = int(idx_str)
        suffix_map.setdefault(rest, {})[run] = p

    sum_per_t: List[float] = []
    cnt_per_t: List[int]   = []
    pairs_used = 0

    # For each suffix with a 0_* file, pair it with all run>0 versions
    for rest, runs in tqdm(suffix_map.items(), desc=f"[{split}] baseline", leave=False):
        if 0 not in runs:
            continue
        base0 = runs[0]
        for run_idx, rp in sorted(runs.items()):
            if run_idx == 0:
                continue
            if max_pairs and pairs_used >= max_pairs:
                break
            mses = per_frame_mse(rp, base0, normalize_01=normalize_01, max_frames=max_frames)
            if not mses:
                continue

            pairs_used += 1
            if len(sum_per_t) < len(mses):
                grow = len(mses) - len(sum_per_t)
                sum_per_t.extend([0.0] * grow)
                cnt_per_t.extend([0]   * grow)
            for t, v in enumerate(mses):
                sum_per_t[t] += v
                cnt_per_t[t] += 1

    if pairs_used == 0 or sum(cnt_per_t) == 0:
        print(f"[{split}/baseline] No valid 0_* vs i_* pairs found.")
        return np.array([])

    cum_num = np.cumsum([sum_per_t[t] for t in range(len(sum_per_t))], dtype=np.float64)
    cum_den = np.cumsum([cnt_per_t[t] for t in range(len(cnt_per_t))], dtype=np.float64)
    baseline_cum = np.divide(cum_num, cum_den, out=np.full_like(cum_num, np.nan), where=cum_den > 0)

    print(f"[{split}/baseline] pairs used: {pairs_used}, max frames used: {len(baseline_cum)}")
    return baseline_cum


def plot_split(split: str,
               curves: Dict[str, np.ndarray],
               out_png: str,
               title_suffix: str = "",
               y_label: str = "Cumulative MSE (0–1 scale)",
               baseline: Optional[np.ndarray] = None):
    plt.figure(figsize=(9, 6))

    # Plot settings first
    for setting, y in sorted(curves.items()):
        if y.size == 0:
            continue
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, label=setting)

    # Add baseline last, dashed
    if baseline is not None and baseline.size > 0:
        x_b = np.arange(1, len(baseline) + 1)
        plt.plot(x_b, baseline, linestyle="--", linewidth=2.0, label="baseline (GT runs)")

    plt.title(f"{split.capitalize()} MSE vs Frames {title_suffix}".strip())
    plt.xlabel("Frames so far")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved: {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Plot per-setting MSE curves over frames for train/test, with GT baseline.")
    ap.add_argument("--gen-root", type=str, default="model_generated",
                    help="Root of generated videos: model_generated/<split>/<setting>/*.mp4")
    ap.add_argument("--gt-root", type=str, default="evaluation_frames",
                    help="Root of ground-truth videos: evaluation_frames/<split>/*.mp4")
    ap.add_argument("--splits", type=str, default="train,test",
                    help="Comma-separated: train,test or just train")
    ap.add_argument("--settings", type=str, default="",
                    help="Comma-separated list to include (default: auto-discover subfolders of gen-root/<split>)")
    ap.add_argument("--out-dir", type=str, default="mse_plots",
                    help="Output directory for PNGs (one per split).")
    ap.add_argument("--normalize01", action="store_true",
                    help="Compute MSE on [0,1] floats (recommended).")
    ap.add_argument("--max-files", type=int, default=0,
                    help="Cap on generated-vs-GT files per setting (0 = no cap).")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Cap frames per pair (0 = no cap).")
    ap.add_argument("--no-baseline", action="store_true",
                    help="Disable plotting the GT baseline curve.")
    ap.add_argument("--max-baseline-pairs", type=int, default=0,
                    help="Cap on number of GT baseline pairs (0 = no cap).")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    include_settings = [s.strip() for s in args.settings.split(",") if s.strip()] if args.settings else None

    for split in splits:
        settings = list_settings(args.gen_root, split, include_settings)
        if not settings:
            print(f"[{split}] No settings found under {os.path.join(args.gen_root, split)}")
            continue

        split_curves: Dict[str, np.ndarray] = {}
        for setting in settings:
            if 'ddim_4' in setting or 'ddim_8' in setting:
                continue
            cum_mean, _ = accumulate_curves(
                args.gen_root, args.gt_root, split, setting,
                normalize_01=args.normalize01,
                max_files=args.max_files,
                max_frames=args.max_frames
            )
            split_curves[setting] = cum_mean

        baseline = None
        if not args.no_baseline:
            baseline = compute_baseline_curve(
                args.gt_root, split,
                normalize_01=args.normalize01,
                max_pairs=args.max_baseline_pairs,
                max_frames=args.max_frames
            )

        out_png = os.path.join(args.out_dir, f"mse_{split}.png")
        title_suffix = "(normalized 0–1)" if args.normalize01 else "(pixel 0–255)"
        ylabel = "Cumulative MSE (0–1 scale)" if args.normalize01 else "Cumulative MSE (0–255 scale)"
        plot_split(split, split_curves, out_png, title_suffix=title_suffix, y_label=ylabel, baseline=baseline)


if __name__ == "__main__":
    main()

