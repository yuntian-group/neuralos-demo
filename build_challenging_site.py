#!/usr/bin/env python3
import os
import re
import argparse
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional

# Consistent preview size (you can change if you want)
SCREEN_WIDTH = 512
SCREEN_HEIGHT = 384

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    """Ensure RGB uint8 and resize to a consistent size for display."""
    if frame.ndim == 2:
        frame = np.stack([frame]*3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]
    elif frame.shape[2] != 3:
        frame = frame[:, :, :3]
    if (frame.shape[1], frame.shape[0]) != (SCREEN_WIDTH, SCREEN_HEIGHT):
        frame = np.array(Image.fromarray(frame).resize((SCREEN_WIDTH, SCREEN_HEIGHT), Image.BILINEAR))
    return frame

def read_frame(video_path: str, idx: int) -> Optional[np.ndarray]:
    """Read a single frame by index with small memory footprint."""
    try:
        reader = imageio.get_reader(video_path)
    except Exception as e:
        print(f"[open fail] {video_path}: {e}")
        return None
    frame = None
    try:
        try:
            frame = reader.get_data(idx)
        except Exception:
            for i, fr in enumerate(reader):
                if i == idx:
                    frame = fr
                    break
    except Exception as e:
        print(f"[read fail] {video_path}@{idx}: {e}")
        frame = None
    finally:
        try: reader.close()
        except Exception: pass
    return frame

def mean_abs_delta01(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel diff on [0,1]."""
    a01 = a.astype(np.float32) / 255.0
    b01 = b.astype(np.float32) / 255.0
    return float(np.mean(np.abs(a01 - b01)))

def rmse01(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE on [0,1]."""
    a01 = a.astype(np.float32) / 255.0
    b01 = b.astype(np.float32) / 255.0
    return float(np.sqrt(np.mean((a01 - b01) ** 2)))

def list_settings(gen_root: str, split: str, include: Optional[List[str]]) -> List[str]:
    base = os.path.join(gen_root, split)
    if include:
        return include  # preserve user-specified order
    if not os.path.isdir(base):
        return []
    return sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

def gather_gt_variants(gt_dir: str) -> Dict[str, Dict[int, str]]:
    """
    Map suffix -> {run_idx: path}
    where files look like '0_foo.mp4', '1_foo.mp4', ...
    suffix is 'foo.mp4' (everything after the first underscore).
    """
    out: Dict[str, Dict[int, str]] = {}
    for name in os.listdir(gt_dir):
        if not name.endswith(".mp4"): 
            continue
        if "_" not in name:
            continue
        idx_str, rest = name.split("_", 1)
        if not idx_str.isdigit():
            continue
        run = int(idx_str)
        out.setdefault(rest, {})[run] = os.path.join(gt_dir, name)
    return out

def find_challenging_events(gt_dir: str, threshold: float, max_examples: int,
                            sort_by: str = "magnitude") -> List[Dict]:
    """
    Scan 0_*.mp4 in gt_dir, find frames t>=1 with mean |Δ| > threshold.
    Returns list of dicts: { 'base': '0_xxx', 'suffix': 'xxx', 'path': <str>, 't': int, 'mag': float }
    """
    events: List[Dict] = []
    for name in sorted(os.listdir(gt_dir)):
        if not (name.endswith(".mp4") and name.startswith("0_")):
            continue
        path = os.path.join(gt_dir, name)
        suffix = name.split("_", 1)[1]  # e.g., 'abc.mp4'
        try:
            reader = imageio.get_reader(path)
        except Exception as e:
            print(f"[skip open] {path}: {e}")
            continue
        prev = None
        t = 0
        try:
            for frame in reader:
                frame = to_rgb_uint8(frame)
                if prev is not None:
                    mag = mean_abs_delta01(frame, prev)
                    if mag > threshold:
                        events.append({
                            "base": name[:-4], # without .mp4
                            "suffix": suffix,
                            "path": path,
                            "t": t,
                            "mag": mag
                        })
                prev = frame
                t += 1
        except Exception as e:
            print(f"[partial read] {path}: {e}")
        finally:
            try: reader.close()
            except Exception: pass

    if sort_by == "magnitude":
        events.sort(key=lambda d: d["mag"], reverse=True)
    else:
        events.sort(key=lambda d: (d["base"], d["t"]))
    if max_examples > 0:
        events = events[:max_examples]
    return events

def save_png(img: np.ndarray, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    Image.fromarray(img).save(out_path)

def frame_triplet_html(prev_rel: Optional[str], curr_rel: Optional[str], next_rel: Optional[str]) -> str:
    """Return HTML for (t-1 → t → t+1) with 'N/A' placeholders when missing."""
    def cell(rel, tag):
        if rel:
            return f"<div class='frame'><div class='tag'>{tag}</div><img src='{rel}'></div>"
        else:
            return f"<div class='frame'><div class='tag'>{tag}</div><div class='empty'>N/A</div></div>"
    return (
        "<div class='triplet'>"
        f"{cell(prev_rel, 't-1')}"
        "<div class='arrow'>→</div>"
        f"{cell(curr_rel, 't')}"
        "<div class='arrow'>→</div>"
        f"{cell(next_rel, 't+1')}"
        "</div>"
    )

def build_event_page(out_dir: str, split: str, idx: int, total: int,
                     event: Dict, settings: List[str],
                     gen_root: str, gt_variants: Dict[str, Dict[int, str]]):
    """
    Create HTML page for one event and materialize all needed thumbnails.
    Layout: 3 columns — Source | Frames (t-1 → t → t+1) | RMSE@t vs GT
    """
    page_dir = os.path.join(out_dir, f"event_{idx:03d}")
    img_dir = os.path.join(page_dir, "img")
    ensure_dir(img_dir)

    base_noext = event["base"]   # '0_foo'
    suffix = event["suffix"]     # 'foo.mp4'
    t = event["t"]
    mag = event["mag"]
    gt_path = event["path"]

    # --- Extract GT (t-1, t, t+1) ---
    gt_prev = read_frame(gt_path, t-1)
    gt_curr = read_frame(gt_path, t)
    gt_next = read_frame(gt_path, t+1)
    if gt_prev is None or gt_curr is None:
        print(f"[skip page] cannot read GT frames for {gt_path}@{t}")
        return
    gt_prev = to_rgb_uint8(gt_prev)
    gt_curr = to_rgb_uint8(gt_curr)
    gt_prev_rel = "img/gt_prev.png"
    gt_curr_rel = "img/gt_curr.png"
    save_png(gt_prev, os.path.join(img_dir, "gt_prev.png"))
    save_png(gt_curr, os.path.join(img_dir, "gt_curr.png"))
    if gt_next is not None:
        gt_next = to_rgb_uint8(gt_next)
        gt_next_rel = "img/gt_next.png"
        save_png(gt_next, os.path.join(img_dir, "gt_next.png"))
    else:
        gt_next_rel = None

    # --- GT variants (1_*, 2_* ...) if available ---
    variant_rows: List[Tuple[str, Optional[str], Optional[str], Optional[str], float]] = []
    if suffix in gt_variants:
        for run_idx in sorted(gt_variants[suffix].keys()):
            if run_idx == 0:
                continue
            vp = gt_variants[suffix][run_idx]
            vp_prev = read_frame(vp, t-1)
            vp_curr = read_frame(vp, t)
            vp_next = read_frame(vp, t+1)
            if vp_prev is None or vp_curr is None:
                continue
            vp_prev = to_rgb_uint8(vp_prev)
            vp_curr = to_rgb_uint8(vp_curr)
            prev_png = os.path.join(img_dir, f"gt_run{run_idx}_prev.png")
            curr_png = os.path.join(img_dir, f"gt_run{run_idx}_curr.png")
            save_png(vp_prev, prev_png)
            save_png(vp_curr, curr_png)
            if vp_next is not None:
                vp_next = to_rgb_uint8(vp_next)
                next_png = os.path.join(img_dir, f"gt_run{run_idx}_next.png")
                save_png(vp_next, next_png)
                next_rel = os.path.relpath(next_png, page_dir)
            else:
                next_rel = None
            rm = rmse01(vp_curr, gt_curr)
            variant_rows.append((
                f"GT run {run_idx}",
                os.path.relpath(prev_png, page_dir),
                os.path.relpath(curr_png, page_dir),
                next_rel,
                rm
            ))

    # --- Settings rows (KEEP user order) ---
    setting_rows: List[Tuple[str, Optional[str], Optional[str], Optional[str], float]] = []
    for setting in settings:
        gen_path = os.path.join(gen_root, split, setting, f"{base_noext}.mp4")
        if not os.path.exists(gen_path):
            continue
        sp = read_frame(gen_path, t-1)
        sc = read_frame(gen_path, t)
        sn = read_frame(gen_path, t+1)
        if sp is None or sc is None:
            continue
        sp = to_rgb_uint8(sp)
        sc = to_rgb_uint8(sc)
        prev_png = os.path.join(img_dir, f"{setting}_prev.png")
        curr_png = os.path.join(img_dir, f"{setting}_curr.png")
        save_png(sp, prev_png)
        save_png(sc, curr_png)
        if sn is not None:
            sn = to_rgb_uint8(sn)
            next_png = os.path.join(img_dir, f"{setting}_next.png")
            save_png(sn, next_png)
            next_rel = os.path.relpath(next_png, page_dir)
        else:
            next_rel = None
        rm = rmse01(sc, gt_curr)
        setting_rows.append((
            setting,
            os.path.relpath(prev_png, page_dir),
            os.path.relpath(curr_png, page_dir),
            next_rel,
            rm
        ))

    # --- Navigation links (point to sibling dirs) ---
    prev_link = f"../event_{idx-1:03d}/event_{idx-1:03d}.html" if idx > 1 else None
    next_link = f"../event_{idx+1:03d}/event_{idx+1:03d}.html" if idx < total else None
    index_link = "../index.html"

    # --- Write HTML ---
    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'>")
    html.append(f"<title>Challenging Event {idx}/{total} — {base_noext}@{t}</title>")
    html.append("<style>")
    html.append("""
    body{font-family:Arial,Helvetica,sans-serif;margin:24px;line-height:1.4}
    .hdr{display:flex;align-items:baseline;gap:16px;margin-bottom:10px}
    .meta{color:#555}
    .nav a{margin-right:12px}
    table{border-collapse:collapse;margin-top:16px;width:100%}
    th,td{border:1px solid #ddd;padding:10px;vertical-align:middle}
    th{background:#f8f8f8;text-align:left}
    .rowlabel{white-space:nowrap;width:230px}
    .triplet{display:flex;gap:10px;align-items:center}
    .frame{display:flex;flex-direction:column;gap:6px;align-items:center}
    .frame .tag{font-size:12px;color:#666}
    .frame img{height:200px;border:1px solid #ccc}
    .empty{height:200px;width:300px;border:1px dashed #ccc;display:flex;align-items:center;justify-content:center;color:#777;background:#fafafa}
    .arrow{opacity:.5}
    .rmse{color:#333;font-variant-numeric:tabular-nums}
    .section{background:#fafafa;font-weight:bold}
    code{background:#f3f3f3;padding:2px 4px}
    """)
    html.append("</style></head><body>")

    html.append("<div class='hdr'>")
    html.append(f"<h2>Challenging Event {idx} / {total}</h2>")
    html.append(f"<div class='meta'>Video: <code>{base_noext}.mp4</code> &nbsp;|&nbsp; Frame: <b>{t}</b> &nbsp;|&nbsp; Δ(GT)={mag:.4f}</div>")
    html.append("</div>")

    html.append("<div class='nav'>")
    html.append(f"<a href='{index_link}'>↩︎ Back to index</a>")
    if prev_link: html.append(f"<a href='{prev_link}'>← Prev</a>")
    if next_link: html.append(f"<a href='{next_link}'>Next →</a>")
    html.append("</div>")

    html.append("<table>")
    # 3-column header
    html.append("<tr><th class='rowlabel'>Source</th><th>Frames (t-1 → t → t+1)</th><th>RMSE@t vs GT</th></tr>")

    # Ground truth row
    html.append("<tr>")
    html.append("<td class='rowlabel'>GT (0_*)</td>")
    html.append("<td>")
    html.append(frame_triplet_html(gt_prev_rel, gt_curr_rel, gt_next_rel))
    html.append("</td>")
    html.append("<td class='rmse'>0.0000</td>")
    html.append("</tr>")

    # GT variant rows
    if variant_rows:
        html.append("<tr><td class='section' colspan='3'>GT Variants</td></tr>")
        for label, prev_rel, curr_rel, next_rel, rm in variant_rows:
            html.append("<tr>")
            html.append(f"<td class='rowlabel'>{label}</td>")
            html.append("<td>")
            html.append(frame_triplet_html(prev_rel, curr_rel, next_rel))
            html.append("</td>")
            html.append(f"<td class='rmse'>{rm:.4f}</td>")
            html.append("</tr>")

    # Model settings rows (do NOT resort; keep CLI order)
    html.append("<tr><td class='section' colspan='3'>Model Settings</td></tr>")
    if not setting_rows:
        html.append("<tr><td colspan='3'><em>No matching generated videos for this example.</em></td></tr>")
    else:
        for label, prev_rel, curr_rel, next_rel, rm in setting_rows:
            html.append("<tr>")
            html.append(f"<td class='rowlabel'><b>{label}</b></td>")
            html.append("<td>")
            html.append(frame_triplet_html(prev_rel, curr_rel, next_rel))
            html.append("</td>")
            html.append(f"<td class='rmse'>{rm:.4f}</td>")
            html.append("</tr>")

    html.append("</table>")

    # Footer nav
    html.append("<div class='nav' style='margin-top:16px'>")
    html.append(f"<a href='{index_link}'>↩︎ Back to index</a>")
    if prev_link: html.append(f"<a href='{prev_link}'>← Prev</a>")
    if next_link: html.append(f"<a href='{next_link}'>Next →</a>")
    html.append("</div>")

    html.append("</body></html>")

    with open(os.path.join(page_dir, f"event_{idx:03d}.html"), "w") as f:
        f.write("\n".join(html))


def build_index(out_dir: str, events: List[Dict]):
    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'>")
    html.append("<title>Challenging Events (train)</title>")
    html.append("<style>")
    html.append("""
    body{font-family:Arial,Helvetica,sans-serif;margin:24px;line-height:1.4}
    table{border-collapse:collapse;width:100%}
    th,td{border:1px solid #ddd;padding:8px}
    th{background:#f8f8f8;text-align:left}
    a{text-decoration:none}
    code{background:#f3f3f3;padding:2px 4px}
    """)
    html.append("</style></head><body>")
    html.append("<h2>Challenging Events (train)</h2>")
    html.append("<p>Showing up to 100 events with mean |Δ(GT)| above threshold (window = 0).</p>")
    html.append("<table>")
    html.append("<tr><th>#</th><th>Link</th><th>Video</th><th>Frame</th><th>Δ(GT)</th></tr>")
    for i, ev in enumerate(events, start=1):
        page_link = f"event_{i:03d}/event_{i:03d}.html"
        html.append("<tr>")
        html.append(f"<td>{i}</td>")
        html.append(f"<td><a href='{page_link}'>Open</a></td>")
        html.append(f"<td><code>{ev['base']}.mp4</code></td>")
        html.append(f"<td>{ev['t']}</td>")
        html.append(f"<td>{ev['mag']:.4f}</td>")
        html.append("</tr>")
    html.append("</table>")
    html.append("</body></html>")
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write("\n".join(html))

def main():
    ap = argparse.ArgumentParser(description="Build static website of challenging transitions (window=0) for train split.")
    ap.add_argument("--gt-root", type=str, default="evaluation_frames",
                    help="Ground truth root (expects evaluation_frames/train/*.mp4)")
    ap.add_argument("--gen-root", type=str, default="model_generated",
                    help="Generated root (expects model_generated/train/<setting>/*.mp4)")
    ap.add_argument("--out-dir", type=str, default="site_challenging_train",
                    help="Output directory for the static site")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="Mean |Δpixel| (on [0,1]) threshold for challenging transition")
    ap.add_argument("--max-examples", type=int, default=100,
                    help="Maximum number of events/pages to generate")
    ap.add_argument("--settings", type=str, default="",
                    help="Comma-separated list; if empty, auto-discover all settings in model_generated/train")
    ap.add_argument("--sort-by", type=str, default="magnitude", choices=["magnitude","index"],
                    help="magnitude: largest GT deltas first; index: by file then frame")
    args = ap.parse_args()

    split = "train"
    gt_dir = os.path.join(args.gt_root, split)
    ensure_dir(args.out_dir)

    # pick settings (preserve user order if provided)
    include = [s.strip() for s in args.settings.split(",") if s.strip()] if args.settings else None
    settings = list_settings(args.gen_root, split, include)
    if not settings:
        print(f"[warn] No settings found under {os.path.join(args.gen_root, split)}; pages will have only GT rows.")

    # find events
    events = find_challenging_events(gt_dir, args.threshold, args.max_examples, sort_by=args.sort_by)
    if not events:
        print("[warn] No challenging events found; nothing to do.")
        return

    # gather gt variants map once
    gt_variants = gather_gt_variants(gt_dir)

    # build per-event pages
    total = len(events)
    for i, ev in enumerate(events, start=1):
        build_event_page(args.out_dir, split, i, total, ev, settings, args.gen_root, gt_variants)

    # index
    build_index(args.out_dir, events)
    print(f"Site generated in: {args.out_dir}\nOpen {os.path.join(args.out_dir, 'index.html')} in a browser.")

if __name__ == "__main__":
    main()

