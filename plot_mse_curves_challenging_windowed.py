#!/usr/bin/env python3
# plot_mse_curves.py
import os
import glob
import argparse
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------
# Utility helpers
# -------------------
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

def safe_open_readers(gen_path: str, gt_path: str):
    try:
        return imageio.get_reader(gen_path), imageio.get_reader(gt_path)
    except Exception as e:
        print(f"[SKIP open] {e} :: {gen_path} vs {gt_path}")
        return None, None

# -------------------
# RMSE computation
# -------------------
def per_frame_rmse(gen_path: str, gt_path: str, normalize_01: bool = True,
                   max_frames: int = 0) -> Optional[List[float]]:
    """RMSE for frames 0..min_len-1. Returns None if unreadable."""
    gen_r, gt_r = safe_open_readers(gen_path, gt_path)
    if gen_r is None:
        return None

    vals: List[float] = []
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
            rmse = math.sqrt(float(np.mean((a - b) ** 2)))
            vals.append(rmse)
            count += 1
    except Exception as e:
        if vals:
            print(f"[PARTIAL read] {e} :: {gen_path} vs {gt_path} (kept {len(vals)} frames)")
        else:
            print(f"[SKIP read] {e} :: {gen_path} vs {gt_path}")
            vals = None
    finally:
        try:
            gen_r.close()
            gt_r.close()
        except Exception:
            pass
    return vals

def read_video_frames_standardized(path: str, target_hw: Optional[Tuple[int,int]]=None,
                                   max_frames: int = 0) -> Optional[List[np.ndarray]]:
    """Read all (or up to max_frames) frames, convert to RGB uint8, resize to target_hw if given."""
    try:
        r = imageio.get_reader(path)
    except Exception as e:
        print(f"[SKIP open] {e} :: {path}")
        return None
    frames: List[np.ndarray] = []
    try:
        for i, frame in enumerate(r):
            if max_frames and i >= max_frames:
                break
            if target_hw is not None:
                frame = prepare_frame(frame, target_hw)
            else:
                h, w = frame.shape[:2]
                frame = prepare_frame(frame, (h, w))
            frames.append(frame)
    except Exception as e:
        if frames:
            print(f"[PARTIAL read std] {e} :: {path} (kept {len(frames)} frames)")
        else:
            print(f"[SKIP read std] {e} :: {path}")
            frames = None
    finally:
        try:
            r.close()
        except Exception:
            pass
    return frames

def per_event_rmse_on_challenging_window(gen_path: str, gt_path: str,
                                         threshold: float = 0.1,
                                         window: int = 0,
                                         normalize_01: bool = True,
                                         max_frames: int = 0,
                                         reduction: str = "best") -> Optional[List[float]]:
    """
    For each 'challenging' GT event at index t (mean |ΔGT| on 0–1 > threshold),
    compute RMSE(gen[s], GT[t]) for s ∈ [t-window, t+window] ∩ [0, n_gen-1].
    - reduction='best': take min over the window
    - reduction='mean': take average over the window
    Returns the per-event RMSE list (ordered by event time). None if unreadable.
    """
    gt_frames = read_video_frames_standardized(gt_path, target_hw=None, max_frames=max_frames)
    if gt_frames is None or len(gt_frames) < 2:
        return None
    H, W = gt_frames[0].shape[:2]
    gt_frames = [prepare_frame(f, (H, W)) for f in gt_frames]

    gen_frames = read_video_frames_standardized(gen_path, target_hw=(H, W), max_frames=max_frames)
    if gen_frames is None or len(gen_frames) == 0:
        return None

    n_gt  = len(gt_frames)
    n_gen = len(gen_frames)

    # find challenging indices on 0–1
    gt01_prev = gt_frames[0].astype(np.float32) / 255.0
    indices: List[int] = []
    for t in range(1, n_gt):
        gt01 = gt_frames[t].astype(np.float32) / 255.0
        trans_mag = float(np.mean(np.abs(gt01 - gt01_prev)))
        if trans_mag > threshold:
            indices.append(t)
        gt01_prev = gt01
    if not indices:
        return []

    ev_vals: List[float] = []
    for t in indices:
        t0 = max(0, t - window)
        t1 = min(n_gen - 1, t + window)
        if t0 > t1 or t >= n_gt:
            continue

        if normalize_01:
            gt_t = gt_frames[t].astype(np.float32) / 255.0
        else:
            gt_t = gt_frames[t].astype(np.float32)

        window_vals: List[float] = []
        for s in range(t0, t1 + 1):
            gen_s = gen_frames[s].astype(np.float32) / 255.0 if normalize_01 else gen_frames[s].astype(np.float32)
            rmse = math.sqrt(float(np.mean((gen_s - gt_t) ** 2)))
            window_vals.append(rmse)

        if not window_vals:
            continue

        if reduction == "best":
            ev_vals.append(min(window_vals))
        elif reduction == "mean":
            ev_vals.append(float(np.mean(window_vals)))
        else:
            raise ValueError(f"Unknown reduction: {reduction}. Use 'best' or 'mean'.")
    return ev_vals

# -------------------
# Aggregation
# -------------------
def accumulate_curves(gen_root: str, gt_root: str, split: str, setting: str,
                      normalize_01: bool, max_files: int, max_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate per-frame RMSEs; returns (cum_mean, per_frame_mean)."""
    gen_dir = os.path.join(gen_root, split, setting)
    gt_dir  = os.path.join(gt_root,  split)
    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.mp4")))
    if max_files > 0:
        gen_files = gen_files[:max_files]
    if not gen_files:
        print(f"[{split}/{setting}] No generated MP4s found in {gen_dir}")
        return np.array([]), np.array([])

    sum_per_t: List[float] = []
    cnt_per_t: List[int] = []
    matched = 0

    for gpath in tqdm(gen_files, desc=f"[{split}] {setting}", leave=False):
        base = os.path.splitext(os.path.basename(gpath))[0]
        gtpath = os.path.join(gt_dir, f"{base}.mp4")
        if not os.path.exists(gtpath):
            continue
        vals = per_frame_rmse(gpath, gtpath, normalize_01=normalize_01, max_frames=max_frames)
        if not vals:
            continue
        matched += 1
        if len(sum_per_t) < len(vals):
            grow = len(vals) - len(sum_per_t)
            sum_per_t.extend([0.0] * grow)
            cnt_per_t.extend([0]   * grow)
        for t, v in enumerate(vals):
            sum_per_t[t] += v
            cnt_per_t[t] += 1

    if matched == 0 or sum(cnt_per_t) == 0:
        print(f"[{split}/{setting}] No matched pairs with frames; skipping.")
        return np.array([]), np.array([])

    per_frame_mean = np.array([(sum_per_t[t] / cnt_per_t[t]) if cnt_per_t[t] > 0 else np.nan
                               for t in range(len(sum_per_t))], dtype=np.float64)
    cum_num = np.cumsum(sum_per_t, dtype=np.float64)
    cum_den = np.cumsum(cnt_per_t, dtype=np.float64)
    cum_mean = np.divide(cum_num, cum_den, out=np.full_like(cum_num, np.nan), where=cum_den > 0)
    print(f"[{split}/{setting}] matched files: {matched}, max frames used: {len(cum_mean)}")
    return cum_mean, per_frame_mean

def accumulate_challenging_curves_window(gen_root: str, gt_root: str, split: str, setting: str,
                                         threshold: float, window: int, normalize_01: bool,
                                         max_files: int, max_frames: int,
                                         reduction: str = "best") -> np.ndarray:
    """
    Aggregate RMSEs on 'challenging' events re-indexed by event order.
    For each event in each video, reduce RMSE within ±window around GT event index by
    'best' (min) or 'mean'. Returns cumulative mean vs number of challenging events.
    """
    gen_dir = os.path.join(gen_root, split, setting)
    gt_dir  = os.path.join(gt_root,  split)
    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.mp4")))
    if max_files > 0:
        gen_files = gen_files[:max_files]
    if not gen_files:
        print(f"[{split}/{setting}] No generated MP4s found in {gen_dir}")
        return np.array([])

    sum_per_e: List[float] = []
    cnt_per_e: List[int] = []
    matched = 0

    for gpath in tqdm(gen_files, desc=f"[{split}] {setting} (chal±{window}, {reduction})", leave=False):
        base = os.path.splitext(os.path.basename(gpath))[0]
        gtpath = os.path.join(gt_dir, f"{base}.mp4")
        if not os.path.exists(gtpath):
            continue
        vals = per_event_rmse_on_challenging_window(
            gpath, gtpath, threshold=threshold, window=window,
            normalize_01=normalize_01, max_frames=max_frames, reduction=reduction
        )
        if not vals:
            continue
        matched += 1
        if len(sum_per_e) < len(vals):
            grow = len(vals) - len(sum_per_e)
            sum_per_e.extend([0.0] * grow)
            cnt_per_e.extend([0]   * grow)
        for e_idx, v in enumerate(vals):
            sum_per_e[e_idx] += v
            cnt_per_e[e_idx] += 1

    if matched == 0 or sum(cnt_per_e) == 0:
        print(f"[{split}/{setting}] No matched pairs with challenging events; skipping.")
        return np.array([])

    cum_num = np.cumsum(sum_per_e, dtype=np.float64)
    cum_den = np.cumsum(cnt_per_e, dtype=np.float64)
    cum_mean = np.divide(cum_num, cum_den, out=np.full_like(cum_num, np.nan), where=cum_den > 0)
    print(f"[{split}/{setting}] challenging matched files: {matched}, max events used: {len(cum_mean)} (±{window}, {reduction})")
    return cum_mean

# -------------------
# Baselines
# -------------------
def compute_baseline_curve(gt_root: str, split: str,
                           normalize_01: bool, max_pairs: int, max_frames: int) -> np.ndarray:
    """Baseline over frames: compare GT runs 1_*, 2_* … vs 0_*."""
    gt_dir = os.path.join(gt_root, split)
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.mp4")))
    if not gt_files:
        print(f"[{split}/baseline] No GT MP4s in {gt_dir}")
        return np.array([])
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
    cnt_per_t: List[int] = []
    pairs_used = 0
    for rest, runs in tqdm(suffix_map.items(), desc=f"[{split}] baseline", leave=False):
        if 0 not in runs:
            continue
        base0 = runs[0]
        for run_idx, rp in sorted(runs.items()):
            if run_idx == 0:
                continue
            if max_pairs and pairs_used >= max_pairs:
                break
            vals = per_frame_rmse(rp, base0, normalize_01=normalize_01, max_frames=max_frames)
            if not vals:
                continue
            pairs_used += 1
            if len(sum_per_t) < len(vals):
                grow = len(vals) - len(sum_per_t)
                sum_per_t.extend([0.0] * grow)
                cnt_per_t.extend([0]   * grow)
            for t, v in enumerate(vals):
                sum_per_t[t] += v
                cnt_per_t[t] += 1

    if pairs_used == 0 or sum(cnt_per_t) == 0:
        print(f"[{split}/baseline] No valid 0_* vs i_* pairs found.")
        return np.array([])
    cum_num = np.cumsum(sum_per_t, dtype=np.float64)
    cum_den = np.cumsum(cnt_per_t, dtype=np.float64)
    baseline_cum = np.divide(cum_num, cum_den, out=np.full_like(cum_num, np.nan), where=cum_den > 0)
    print(f"[{split}/baseline] pairs used: {pairs_used}, max frames used: {len(baseline_cum)}")
    return baseline_cum

def compute_baseline_challenging_curve_window(gt_root: str, split: str,
                                              threshold: float, window: int,
                                              normalize_01: bool, max_pairs: int, max_frames: int,
                                              reduction: str = "best") -> np.ndarray:
    """
    Baseline on challenging events with windowing: compare runs i>0 vs 0_*.
    For challenging events in 0_* at index t, reduce RMSE within ±window by 'best' or 'mean'.
    """
    gt_dir = os.path.join(gt_root, split)
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.mp4")))
    if not gt_files:
        print(f"[{split}/baseline-chal±{window}] No GT MP4s in {gt_dir}")
        return np.array([])
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

    sum_per_e: List[float] = []
    cnt_per_e: List[int] = []
    pairs_used = 0
    for rest, runs in tqdm(suffix_map.items(), desc=f"[{split}] baseline (chal±{window}, {reduction})", leave=False):
        if 0 not in runs:
            continue
        base0 = runs[0]
        for run_idx, rp in sorted(runs.items()):
            if run_idx == 0:
                continue
            if max_pairs and pairs_used >= max_pairs:
                break
            vals = per_event_rmse_on_challenging_window(
                rp, base0, threshold=threshold, window=window,
                normalize_01=normalize_01, max_frames=max_frames, reduction=reduction
            )
            if not vals:
                continue
            pairs_used += 1
            if len(sum_per_e) < len(vals):
                grow = len(vals) - len(sum_per_e)
                sum_per_e.extend([0.0] * grow)
                cnt_per_e.extend([0]   * grow)
            for e_idx, v in enumerate(vals):
                sum_per_e[e_idx] += v
                cnt_per_e[e_idx] += 1

    if pairs_used == 0 or sum(cnt_per_e) == 0:
        print(f"[{split}/baseline-chal±{window}] No valid challenging pairs found ({reduction}).")
        return np.array([])

    cum_num = np.cumsum(sum_per_e, dtype=np.float64)
    cum_den = np.cumsum(cnt_per_e, dtype=np.float64)
    baseline_cum = np.divide(cum_num, cum_den, out=np.full_like(cum_num, np.nan), where=cum_den > 0)
    print(f"[{split}/baseline-chal±{window}] pairs used: {pairs_used}, max events used: {len(baseline_cum)} ({reduction})")
    return baseline_cum

# -------------------
# Plotting
# -------------------
def plot_split(split: str,
               curves: Dict[str, np.ndarray],
               out_png: str,
               title_suffix: str = "",
               y_label: str = "Cumulative RMSE (0–1 scale)",
               baseline: Optional[np.ndarray] = None):
    plt.figure(figsize=(9, 6))
    for setting, y in sorted(curves.items()):
        if y.size == 0:
            continue
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, label=setting)
    if baseline is not None and baseline.size > 0:
        x_b = np.arange(1, len(baseline) + 1)
        plt.plot(x_b, baseline, linestyle="--", linewidth=2.0, label="baseline (GT runs)")
    plt.title(f"{split.capitalize()} RMSE vs Frames {title_suffix}".strip())
    plt.xlabel("Frames so far")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved: {out_png}")

def plot_split_challenging_window(split: str,
                                  curves: Dict[str, np.ndarray],
                                  out_png: str,
                                  window: int,
                                  reduction_label: str,
                                  title_suffix: str = "",
                                  y_label: str = "Cumulative RMSE (0–1 scale)",
                                  baseline: Optional[np.ndarray] = None):
    plt.figure(figsize=(9, 6))
    for setting, y in sorted(curves.items()):
        if y.size == 0:
            continue
        x = np.arange(1, len(y) + 1)
        plt.plot(x, y, label=setting)
    if baseline is not None and baseline.size > 0:
        x_b = np.arange(1, len(baseline) + 1)
        plt.plot(x_b, baseline, linestyle="--", linewidth=2.0, label="baseline (GT runs)")
    plt.title(f"{split.capitalize()} RMSE on Challenging Transitions ±{window} [{reduction_label}] {title_suffix}".strip())
    plt.xlabel("Challenging events so far")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved: {out_png}")

# -------------------
# Main
# -------------------
def main():
    ap = argparse.ArgumentParser(description="Plot per-setting RMSE (overall + challenging±W) with GT baseline.")
    ap.add_argument("--gen-root", type=str, default="model_generated",
                    help="Root of generated videos: model_generated/<split>/<setting>/*.mp4")
    ap.add_argument("--gt-root", type=str, default="evaluation_frames",
                    help="Root of ground-truth videos: evaluation_frames/<split>/*.mp4")
    ap.add_argument("--splits", type=str, default="train,test",
                    help="Comma-separated: train,test or just train")
    ap.add_argument("--settings", type=str, default="",
                    help="Comma-separated list to include (default: auto-discover subfolders of gen-root/<split>)")
    ap.add_argument("--out-dir", type=str, default="mse_plots",
                    help="Output directory for PNGs.")
    ap.add_argument("--normalize01", action="store_true",
                    help="Compute RMSE on [0,1] (recommended).")
    ap.add_argument("--max-files", type=int, default=0,
                    help="Cap on generated-vs-GT files per setting (0 = no cap).")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Cap frames per pair (0 = no cap).")
    ap.add_argument("--no-baseline", action="store_true",
                    help="Disable plotting the GT baseline curve.")
    ap.add_argument("--max-baseline-pairs", type=int, default=0,
                    help="Cap on number of GT baseline pairs (0 = no cap).")
    ap.add_argument("--challenging-threshold", type=float, default=0.1,
                    help="Threshold on GT transition magnitude (on 0–1 scale).")
    ap.add_argument("--challenging-max-window", type=int, default=5,
                    help="Max ±window (frames) for challenging-event alignment.")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    include_settings = [s.strip() for s in args.settings.split(",") if s.strip()] if args.settings else None

    for split in splits:
        settings = list_settings(args.gen_root, split, include_settings)
        if not settings:
            print(f"[{split}] No settings found under {os.path.join(args.gen_root, split)}")
            continue

        # ---------- Overall (frame-indexed) ----------
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
        ylabel = "Cumulative RMSE (0–1 scale)" if args.normalize01 else "Cumulative RMSE (0–255 scale)"
        plot_split(split, split_curves, out_png, title_suffix=title_suffix, y_label=ylabel, baseline=baseline)

        # ---------- Challenging (event-indexed, windowed) ----------
        for window in range(0, args.challenging_max_window + 1):
            # BEST (min) within window → keep old filename
            split_chal_curves_best: Dict[str, np.ndarray] = {}
            for setting in settings:
                if 'ddim_4' in setting or 'ddim_8' in setting:
                    continue
                chal_cum = accumulate_challenging_curves_window(
                    args.gen_root, args.gt_root, split, setting,
                    threshold=args.challenging_threshold,
                    window=window,
                    normalize_01=args.normalize01,
                    max_files=args.max_files,
                    max_frames=args.max_frames,
                    reduction="best"
                )
                split_chal_curves_best[setting] = chal_cum

            baseline_chal_best = None
            if not args.no_baseline:
                baseline_chal_best = compute_baseline_challenging_curve_window(
                    args.gt_root, split,
                    threshold=args.challenging_threshold,
                    window=window,
                    normalize_01=args.normalize01,
                    max_pairs=args.max_baseline_pairs,
                    max_frames=args.max_frames,
                    reduction="best"
                )

            out_png_chal_best = os.path.join(args.out_dir, f"mse_{split}_challenging_pm{window}.png")
            title_suffix_chal = f"(thr={args.challenging_threshold:.2f}; GT Δ on 0–1)"
            plot_split_challenging_window(
                split, split_chal_curves_best, out_png_chal_best, window,
                reduction_label="best-in-window",
                title_suffix=title_suffix_chal,
                y_label=ylabel,
                baseline=baseline_chal_best
            )

            # MEAN within window → new filename with _mean suffix
            split_chal_curves_mean: Dict[str, np.ndarray] = {}
            for setting in settings:
                if 'ddim_4' in setting or 'ddim_8' in setting:
                    continue
                chal_cum = accumulate_challenging_curves_window(
                    args.gen_root, args.gt_root, split, setting,
                    threshold=args.challenging_threshold,
                    window=window,
                    normalize_01=args.normalize01,
                    max_files=args.max_files,
                    max_frames=args.max_frames,
                    reduction="mean"
                )
                split_chal_curves_mean[setting] = chal_cum

            baseline_chal_mean = None
            if not args.no_baseline:
                baseline_chal_mean = compute_baseline_challenging_curve_window(
                    args.gt_root, split,
                    threshold=args.challenging_threshold,
                    window=window,
                    normalize_01=args.normalize01,
                    max_pairs=args.max_baseline_pairs,
                    max_frames=args.max_frames,
                    reduction="mean"
                )

            out_png_chal_mean = os.path.join(args.out_dir, f"mse_{split}_challenging_pm{window}_mean.png")
            plot_split_challenging_window(
                split, split_chal_curves_mean, out_png_chal_mean, window,
                reduction_label="mean-in-window",
                title_suffix=title_suffix_chal,
                y_label=ylabel,
                baseline=baseline_chal_mean
            )

if __name__ == "__main__":
    main()

