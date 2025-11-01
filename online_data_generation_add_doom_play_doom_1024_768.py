#!/usr/bin/env python3
import os
import json
import glob
import time
import random
import sqlite3
import logging
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from omegaconf import OmegaConf
from computer.util import load_model_from_config
from PIL import Image, ImageEnhance
import io
import torch
from einops import rearrange
import webdataset as wds
import pandas as pd
import ast
import pickle
from moviepy.editor import VideoFileClip
import signal

# Import the existing functions
from data.data_collection.synthetic_script_compute_canada import process_trajectory, initialize_clean_state

# VizDoom for Doom regions
import vizdoom as vzd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trajectory_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
DB_FILE = "trajectory_processor.db"
FRAMES_DIR = "interaction_logs"
OUTPUT_DIR = 'train_dataset_encoded_online'
os.makedirs(OUTPUT_DIR, exist_ok=True)
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
MEMORY_LIMIT = "2g"
CHECK_INTERVAL = 60  # Check for new data every 60 seconds

# Doom icon placement and detection (keeping original size for realistic desktop behavior)
DOOM_ICON_WIDTH = 50
DOOM_ICON_HEIGHT = 70
DOOM_ICON_OFFSET_X = 90  # 2X of original 45 to maintain proportional spacing from right edge
DOOM_ICON_POS_X = SCREEN_WIDTH - DOOM_ICON_WIDTH - DOOM_ICON_OFFSET_X  # 884
DOOM_ICON_POS_Y = 46  # 2X of original 23 to maintain proportional spacing from top edge
DOOM_ICON_RECT = (DOOM_ICON_POS_X, DOOM_ICON_POS_Y, DOOM_ICON_POS_X + DOOM_ICON_WIDTH, DOOM_ICON_POS_Y + DOOM_ICON_HEIGHT)  # (884,46,934,116)

# load autoencoder
config = OmegaConf.load('../computer/autoencoder/config_kl4_lr4.5e6_load_acc1_512_384_mar10_keyboard_init_16_contmar15_acc1.yaml')
autoencoder = load_model_from_config(config, '../computer/autoencoder/saved_kl4_bsz8_acc8_lr4.5e6_load_acc1_512_384_mar10_keyboard_init_16_cont_mar15_acc1_cont_1e6_cont_2e7_cont/model-2076000.ckpt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder = autoencoder.to(device)

# Global flag for graceful shutdown
running = True

KEYMAPPING = {
    'arrowup': 'up',
    'arrowdown': 'down',
    'arrowleft': 'left',
    'arrowright': 'right',
    'meta': 'command',
    'contextmenu': 'apps',
    'control': 'ctrl',
}

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
INVALID_KEYS = ['f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20',
                'f21', 'f22', 'f23', 'f24', 'select', 'separator', 'execute']
VALID_KEYS = [key for key in KEYS if key not in INVALID_KEYS]
itos = VALID_KEYS
stoi = {key: i for i, key in enumerate(itos)}

# Desktop and doom icon assets
_DESKTOP_REF_BGR = cv2.imread('desktop.png', cv2.IMREAD_COLOR)
if _DESKTOP_REF_BGR is None:
    assert False, 'desktop.png not found or unreadable; desktop detection will be disabled.'


_DOOM_ICON_BASE = Image.open('doom_icon.png').convert('RGBA')
_DOOM_ICON = _DOOM_ICON_BASE.resize((DOOM_ICON_WIDTH, DOOM_ICON_HEIGHT), Image.Resampling.LANCZOS)
# Build highlighted variant (brighten top 105 px, clipped by height)
width_h, height_h = _DOOM_ICON.size
top_region = _DOOM_ICON.crop((0, 0, width_h, min(105, height_h)))
enhancer = ImageEnhance.Brightness(top_region)
bright_top = enhancer.enhance(1.5)
_DOOM_ICON_HL = _DOOM_ICON.copy()
_DOOM_ICON_HL.paste(bright_top, (0, 0))

# Doom control mapping used by VizDoom runner
# Keys are normalized inline: lower() then KEYMAPPING application
KEY_TO_BUTTON = {
    # movement (classic: arrows move/turn; WASD emulate arrows)
    # W/A/S/D mapped to classic arrow semantics
    "w": vzd.Button.MOVE_FORWARD,     # like ArrowUp
    "s": vzd.Button.MOVE_BACKWARD,    # like ArrowDown
    "a": vzd.Button.TURN_LEFT,        # like ArrowLeft (turn)
    "d": vzd.Button.TURN_RIGHT,       # like ArrowRight (turn)

    # classic DOOM arrows: up/down move, left/right turn
    "up": vzd.Button.MOVE_FORWARD,
    "down": vzd.Button.MOVE_BACKWARD,
    "left": vzd.Button.TURN_LEFT,
    "right": vzd.Button.TURN_RIGHT,

    # strafe modifier + run modifier
    "alt":   vzd.Button.STRAFE,
    "shift": vzd.Button.SPEED,

    # actions (hybrid: classic + modern)
    "space": vzd.Button.USE,      # open doors/switches
    " ": vzd.Button.USE,      # open doors/switches
    "e":     vzd.Button.USE,      # also allow E to use
    "ctrl":  vzd.Button.ATTACK,   # classic fire

    # optional extras
    "q":     vzd.Button.SELECT_PREV_WEAPON,
    "r":     vzd.Button.SELECT_NEXT_WEAPON,
    "z":     vzd.Button.ZOOM,

    # weapon quick-select (top number row)
    "1": vzd.Button.SELECT_WEAPON1,
    "2": vzd.Button.SELECT_WEAPON2,
    "3": vzd.Button.SELECT_WEAPON3,
    "4": vzd.Button.SELECT_WEAPON4,
    "5": vzd.Button.SELECT_WEAPON5,
    "6": vzd.Button.SELECT_WEAPON6,
    "7": vzd.Button.SELECT_WEAPON7,

    # allow Enter as attack too
    "enter": vzd.Button.ATTACK,
}
MOUSE_LEFT = "mouse1"

KEY_TO_BUTTON[MOUSE_LEFT]  = vzd.Button.ATTACK

# Helpers for overlay
def detect_cursor_in_frame(frame_bgr: np.ndarray, white_thresh: int = 35, min_area: int = 50):
    try:
        H, W = frame_bgr.shape[:2]
        roi_x = W - 256
        roi_y = 0
        roi_w = 256
        roi_h = 256
        region = frame_bgr[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, white_bin = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(white_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_area:
            return None, None, None
        x, y, w, h = cv2.boundingRect(cnt)
        mask = np.zeros((h, w), dtype=np.uint8)
        cnt_local = cnt - np.array([[x, y]])
        cv2.drawContours(mask, [cnt_local], -1, 255, thickness=cv2.FILLED)
        cursor_patch = region[y:y + h, x:x + w].copy()
        return cursor_patch, mask, (roi_x + x, roi_y + y)
    except Exception:
        return None, None, None

def is_desktop_frame(frame_bgr: np.ndarray, desktop_bgr: np.ndarray, mae_threshold: float = 12.0) -> bool:
    if desktop_bgr is None:
        return False
    if frame_bgr.shape[:2] != desktop_bgr.shape[:2]:
        return False
    diff = cv2.absdiff(frame_bgr, desktop_bgr)
    mae = float(np.mean(diff))
    return mae < mae_threshold

def overlay_doom_icon_on_frame(frame_bgr: np.ndarray) -> np.ndarray:
    cursor_patch, cursor_mask, cursor_pos = detect_cursor_in_frame(frame_bgr)
    highlight = False
    if cursor_patch is not None and cursor_pos is not None:
        px, py = cursor_pos
        h, w = cursor_patch.shape[:2]
        cx1, cy1, cx2, cy2 = (px, py, px + w, py + h)
        ix1, iy1, ix2, iy2 = DOOM_ICON_RECT
        inter = not (cx2 <= ix1 or cx1 >= ix2 or cy2 <= iy1 or cy1 >= iy2)
        highlight = inter
    frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    desktop_img = Image.fromarray(frame_rgba)
    icon_to_use = _DOOM_ICON_HL if highlight else _DOOM_ICON
    desktop_img.paste(icon_to_use, (DOOM_ICON_POS_X, DOOM_ICON_POS_Y), icon_to_use)
    out_bgr = cv2.cvtColor(np.array(desktop_img), cv2.COLOR_RGBA2BGR)
    if cursor_patch is not None and cursor_pos is not None:
        px, py = cursor_pos
        h, w = cursor_patch.shape[:2]
        roi = out_bgr[py:py + h, px:px + w]
        mask_bool = (cursor_mask > 0)[..., None]
        roi[:] = np.where(mask_bool, cursor_patch, roi)
        out_bgr[py:py + h, px:px + w] = roi
    return out_bgr

# Doom helpers
def chw_to_hwc_rgb(arr):
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        return np.transpose(arr, (1, 2, 0))
    return arr

class AbsToDelta:
    def __init__(self):
        self.prev = None
    def reset(self):
        self.prev = None
    def __call__(self, x, y, sens_x=1.0, sens_y=1.0, clamp=80):
        if self.prev is None:
            self.prev = (x, y)
            return 0, 0
        dx = int(round((x - self.prev[0]) * sens_x))
        dy = int(round((y - self.prev[1]) * sens_y))
        self.prev = (x, y)
        if clamp is not None:
            dx = max(-clamp, min(clamp, dx))
            dy = max(-clamp, min(clamp, dy))
        return dx, dy

def build_action_vector(buttons_order, held_keys_set, left_click, dx, dy):
    action = [0] * len(buttons_order)
    # Disable pitch: ignore dy entirely
    dy = 0
    # Support classic strafe with ALT: if STRAFE held, treat left/right as MOVE_LEFT/RIGHT in addition to turn buttons
    strafe_held = ('alt' in held_keys_set and KEY_TO_BUTTON.get('alt') == vzd.Button.STRAFE)
    for i, btn in enumerate(buttons_order):
        for k in held_keys_set:
            if KEY_TO_BUTTON.get(k) == btn:
                action[i] = 1
                break
        if btn == vzd.Button.TURN_LEFT_RIGHT_DELTA:
            action[i] = int(dx)
        elif btn == vzd.Button.LOOK_UP_DOWN_DELTA:
            action[i] = int(dy)
        elif btn == vzd.Button.MOVE_LEFT and strafe_held:
            # emulate analog strafe from mouse by mapping dx<0 to move left
            if dx < 0:
                actino[i] = 1
        elif btn == vzd.Button.MOVE_RIGHT and strafe_held:
            if dx > 0:
                action[i] = 1
        elif btn == vzd.Button.ATTACK:
            if left_click:
                action[i] = 1
            
    return action

def run_vizdoom_segment(action_seq: List[Tuple[Tuple[int, int], bool, bool, List[Tuple[str, str]]]], fps_skip: int = 3) -> List[np.ndarray]:
    game = vzd.DoomGame()
    game.set_doom_game_path("doom1.wad")  # ensure this file exists or adjust path
    game.set_doom_map("E1M1")
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_1024X768)
    game.set_window_visible(False)
    # Fixed, hardcoded order of buttons for stability (no duplicates)
    buttons_order = [
        # movement
        vzd.Button.MOVE_FORWARD,
        vzd.Button.MOVE_BACKWARD,
        vzd.Button.MOVE_LEFT,
        vzd.Button.MOVE_RIGHT,
        # turning
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
        # modifiers
        vzd.Button.STRAFE,
        vzd.Button.SPEED,
        # actions
        vzd.Button.JUMP,
        vzd.Button.CROUCH,
        vzd.Button.USE,
        vzd.Button.RELOAD,
        vzd.Button.SELECT_PREV_WEAPON,
        vzd.Button.SELECT_NEXT_WEAPON,
        vzd.Button.ZOOM,
        # weapon quick-selects
        vzd.Button.SELECT_WEAPON1,
        vzd.Button.SELECT_WEAPON2,
        vzd.Button.SELECT_WEAPON3,
        vzd.Button.SELECT_WEAPON4,
        vzd.Button.SELECT_WEAPON5,
        vzd.Button.SELECT_WEAPON6,
        vzd.Button.SELECT_WEAPON7,
        # attack
        vzd.Button.ATTACK,
        # continuous deltas appended once at the end
        vzd.Button.TURN_LEFT_RIGHT_DELTA,
        vzd.Button.LOOK_UP_DOWN_DELTA,
    ]
    # Validate: every mapped button must be in buttons_order
    required_buttons = {btn for btn in KEY_TO_BUTTON.values() if btn is not None}
    missing = [btn for btn in required_buttons if btn not in buttons_order]
    if len(missing) > 0:
        assert False, f"Some mapped buttons are missing from fixed buttons order: {missing}"
    game.set_available_buttons(buttons_order)
    game.set_episode_timeout(10_000_000)
    # Make visuals closer to classic Doom experience
    game.set_render_hud(True)
    game.set_render_crosshair(True)
    game.set_render_weapon(True)
    game.set_render_messages(True)
    game.set_render_decals(True)
    game.set_render_particles(True)
    game.set_render_corpses(True)
    game.init()
    game.new_episode()
    held_keys = set()
    abs2delta = AbsToDelta()
    frames: List[np.ndarray] = []
    for t, act in enumerate(action_seq):
        (x, y), left_click, _right_click, key_events = act
        # Apply all key events for this tick in order
        if key_events:
            for evt_type, key in key_events:
                if evt_type == "key_down":
                    held_keys.add(key)
                elif evt_type == "key_up":
                    held_keys.discard(key)
        dx, dy = abs2delta(x, y, sens_x=1.0, sens_y=1.0, clamp=80)
        action_vec = build_action_vector(buttons_order, held_keys, bool(left_click), dx, dy)
        _reward = game.make_action(action_vec, fps_skip)
        done = game.is_episode_finished()
        num_restarts = 0
        while done:
            # Restart episode and re-issue the same action so this tick still yields a frame
            game.new_episode()
            _reward = game.make_action(action_vec, fps_skip)
            done = game.is_episode_finished()
            num_restarts += 1
            assert num_restarts < 10, 'should not happen that we restart more than 10 times'
        state = game.get_state()
        if state is None or state.screen_buffer is None:
            assert False, 'should not happen that state is None or state.screen_buffer is None'
        rgb = chw_to_hwc_rgb(state.screen_buffer)
        if rgb is None or rgb.size == 0:
            assert False, 'should not happen that rgb is None or rgb.size == 0'
        resized = cv2.resize(rgb, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_AREA)
        frames.append(resized)
    game.close()
    return frames

# Graceful shutdown
def signal_handler(sig, frame):
    global running
    logger.info("Shutdown signal received. Finishing current processing and exiting...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# DB init
def initialize_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processed_sessions (
        id INTEGER PRIMARY KEY,
        log_file TEXT UNIQUE,
        client_id TEXT,
        processed_time TIMESTAMP
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processed_segments (
        id INTEGER PRIMARY KEY,
        log_file TEXT,
        client_id TEXT,
        segment_index INTEGER,
        start_time REAL,
        end_time REAL,
        processed_time TIMESTAMP,
        trajectory_id INTEGER,
        UNIQUE(log_file, segment_index)
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS config (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    ''')
    cursor.execute("SELECT value FROM config WHERE key = 'next_id'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO config (key, value) VALUES ('next_id', '0')")
    conn.commit()
    conn.close()

# Trajectory helpers
def load_trajectory(log_file):
    trajectory = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    trajectory.append(entry)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {log_file}")
                    continue
        return trajectory
    
    except Exception as e:
        logger.error(f"Error loading trajectory from {log_file}: {e}")
        return []

@torch.no_grad()
def format_trajectory_for_processing(trajectory):
    formatted_events = []
    down_keys = set([])
    for entry in trajectory:
        if entry.get("is_reset") or entry.get("is_eos"):
            continue
        inputs = entry.get("inputs", {})
        key_events = []
        for key in inputs.get("keys_down", []):
            key = key.lower()
            if key in KEYMAPPING:
                key = KEYMAPPING[key]
            if key not in stoi:
                continue
            if key not in down_keys:
                down_keys.add(key)
                key_events.append(("keydown", key))
        for key in inputs.get("keys_up", []):
            key = key.lower()
            if key in KEYMAPPING:
                key = KEYMAPPING[key]
            if key not in stoi:
                continue
            if key in down_keys:
                down_keys.remove(key)
                key_events.append(("keyup", key))
        x = inputs.get("x")
        y = inputs.get("y")
        x = min(max(0, x), SCREEN_WIDTH - 1) if x is not None else 0
        y = min(max(0, y), SCREEN_HEIGHT - 1) if y is not None else 0
        event = {
            "pos": (x, y),
            "left_click": inputs.get("is_left_click", False),
            "right_click": inputs.get("is_right_click", False),
            "key_events": key_events,
        }
        formatted_events.append(event)
    return formatted_events

# Doom region detection
def _inside_icon(x: int, y: int) -> bool:
    return (x is not None and y is not None and DOOM_ICON_RECT[0] <= x <= DOOM_ICON_RECT[2] - 1 and DOOM_ICON_RECT[1] <= y <= DOOM_ICON_RECT[3] - 1)

def find_doom_regions(sub_traj: List[Dict[str, Any]], max_gap_frames: int = 3) -> List[Tuple[int, int]]:
    regions: List[Tuple[int, int]] = []
    last_click_idx = None
    i = 0
    while i < len(sub_traj):
        e = sub_traj[i]
        if e.get("is_reset") or e.get("is_eos"):
            i += 1
            continue
        inputs = e.get("inputs", {})
        x = inputs.get("x")
        y = inputs.get("y")
        left = bool(inputs.get("is_left_click", False))
        if left and _inside_icon(x, y):
            if last_click_idx is not None and (i - last_click_idx) <= max_gap_frames:
                # Double click detected — find first ensuing ESC keydown
                start_idx = last_click_idx
                end_idx = None
                j = i
                while j < len(sub_traj):
                    ej = sub_traj[j]
                    if ej.get("is_reset") or ej.get("is_eos"):
                        j += 1
                        continue
                    keys_down = [k.lower() for k in ej.get("inputs", {}).get("keys_down", [])]
                    if ("esc" in keys_down) or ("escape" in keys_down):
                        end_idx = j
                        break
                    j += 1
                if end_idx is None:
                    end_idx = len(sub_traj) - 1
                regions.append((start_idx, end_idx))
                # Reset state and jump beyond end of doom region
                last_click_idx = None
                i = end_idx + 1
                continue
            else:
                last_click_idx = i
        i += 1
    # Merge overlapping/adjacent regions if any
    if not regions:
        return regions
    regions.sort()
    merged = [regions[0]]
    for s, e in regions[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            assert False, 'should not happen that two doom regions are adjacent'
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

def build_action_seq_for_doom(traj_slice: List[Dict[str, Any]]) -> List[Tuple[Tuple[int, int], bool, bool, List[Tuple[str, str]]]]:
    seq: List[Tuple[Tuple[int, int], bool, bool, List[Tuple[str, str]]]] = []
    down_keys = set([])
    for entry in traj_slice:
        if entry.get("is_reset") or entry.get("is_eos"):
            continue
        inputs = entry.get("inputs", {})
        x = inputs.get("x")
        y = inputs.get("y")
        x = min(max(0, x), SCREEN_WIDTH - 1) if x is not None else 0
        y = min(max(0, y), SCREEN_HEIGHT - 1) if y is not None else 0
        left_click = bool(inputs.get("is_left_click", False))
        right_click = bool(inputs.get("is_right_click", False))
        # collect all key events for this tick (downs then ups) using same logic as regular mode
        key_events: List[Tuple[str, str]] = []
        for key in inputs.get("keys_down", []):
            key = key.lower()
            if key in KEYMAPPING:
                key = KEYMAPPING[key]
            if key not in stoi:
                continue
            if key not in down_keys:
                down_keys.add(key)
                key_events.append(("key_down", key))
        for key in inputs.get("keys_up", []):
            key = key.lower()
            if key in KEYMAPPING:
                key = KEYMAPPING[key]
            if key not in stoi:
                continue
            if key in down_keys:
                down_keys.remove(key)
                key_events.append(("key_up", key))
        seq.append(((x, y), left_click, right_click, key_events))
    return seq

def _count_non_control_up_to(traj: List[Dict[str, Any]], idx_inclusive: int) -> int:
    cnt = 0
    for j in range(0, idx_inclusive + 1):
        e = traj[j]
        if e.get("is_reset") or e.get("is_eos"):
            continue
        cnt += 1
    return cnt

def _find_first_double_click_pair(sub_traj: List[Dict[str, Any]], start_idx: int, max_gap_frames: int = 3) -> Tuple[int, int] or None:
    last_click_idx = None
    i = start_idx
    n = len(sub_traj)
    while i < n:
        e = sub_traj[i]
        if e.get("is_reset") or e.get("is_eos"):
            i += 1
            continue
        inputs = e.get("inputs", {})
        x = inputs.get("x")
        y = inputs.get("y")
        left = bool(inputs.get("is_left_click", False))
        if left and _inside_icon(x, y):
            if last_click_idx is not None and (i - last_click_idx) <= max_gap_frames:
                return (last_click_idx, i)
            else:
                last_click_idx = i
        i += 1
    return None

def _render_slice_and_check_desktop(sub_traj_slice: List[Dict[str, Any]], check_start_idx: int, check_end_idx: int, clean_state) -> bool:
    """Render the given sub-traj slice (up to second click) with docker and verify frames between indexes are desktop.
    check_start_idx/check_end_idx refer to indices within sub_traj_slice (which is prefix up to second click).
    """
    # Format events
    formatted = format_trajectory_for_processing(sub_traj_slice)
    if len(formatted) == 0:
        return False
    # Temp record id
    temp_id = int(time.time() * 1000) + random.randint(0, 1000)
    try:
        args = (temp_id, formatted)
        process_trajectory(args, SCREEN_WIDTH, SCREEN_HEIGHT, clean_state, MEMORY_LIMIT)
        video_file = f'raw_data/raw_data/videos/record_{temp_id}.mp4'
        # Map entry indices to frame indices
        f1 = _count_non_control_up_to(sub_traj_slice, check_start_idx) - 1
        f2 = _count_non_control_up_to(sub_traj_slice, check_end_idx) - 1
        if f1 < 0 or f2 < f1:
            assert False, 'should not happen that f1 < 0 or f2 < f1'
        with VideoFileClip(video_file) as video:
            fps = video.fps
            total_frames = int(round(fps * video.duration))
            # Basic sanity: generated frames should match number of formatted events
            assert total_frames == len(formatted), 'video frames should match formatted events'
            for fi in range(f1, f2 + 1):
                frame_rgb = video.get_frame(fi / fps)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if not is_desktop_frame(frame_bgr, _DESKTOP_REF_BGR):
                    return False
        return True
    finally:
        # Clean temp outputs to avoid interference
        try:
            os.remove(f'raw_data/raw_data/videos/record_{temp_id}.mp4')
        except Exception:
            pass
        try:
            os.remove(f'raw_data/raw_data/actions/record_{temp_id}.csv')
        except Exception:
            pass

def find_doom_regions_on_desktop(sub_traj: List[Dict[str, Any]], clean_state, max_gap_frames: int = 3) -> List[Tuple[int, int]]:
    """Iteratively validate doom double-clicks by re-rendering with docker and checking desktop frames between clicks.
    Returns validated (start_idx, end_idx) pairs in original sub_traj index space.
    """
    validated: List[Tuple[int, int]] = []
    base_offset = 0
    remaining = list(sub_traj)
    scan_start = 0
    while True:
        pair = _find_first_double_click_pair(remaining, scan_start, max_gap_frames=max_gap_frames)
        if pair is None:
            break
        i1, i2 = pair
        # Render prefix up to second click and verify desktop frames
        prefix_slice = remaining[:i2 + 1]
        if _render_slice_and_check_desktop(prefix_slice, i1, i2, clean_state):
            # Valid doom double-click; find ensuing ESC in remaining
            end_idx = None
            j = i2
            while j < len(remaining):
                ej = remaining[j]
                if ej.get("is_reset") or ej.get("is_eos"):
                    j += 1
                    continue
                keys_down = [k.lower() for k in ej.get("inputs", {}).get("keys_down", [])]
                if ("esc" in keys_down) or ("escape" in keys_down):
                    end_idx = j
                    break
                j += 1
            if end_idx is None:
                end_idx = len(remaining) - 1
            # Record region in original indices
            validated.append((base_offset + i1, base_offset + end_idx))
            # Remove region (everything up to and including end_idx) and continue
            base_offset += end_idx + 1
            remaining = remaining[end_idx + 1:]
            scan_start = 0
        else:
            # Not valid; continue searching after second click
            scan_start = i2 + 1
    return validated

@torch.no_grad()
def process_session_file(log_file, clean_state):
    """Process a session file, splitting into multiple trajectories at reset points."""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("BEGIN TRANSACTION")
        cursor = conn.cursor()
        os.makedirs("generated_videos", exist_ok=True)
        trajectory = load_trajectory(log_file)
        if not trajectory:
            logger.error(f"Empty trajectory for {log_file}, skipping")
            return []
        client_id = trajectory[0].get("client_id", "unknown")
        reset_indices = []
        has_eos = False
        for i, entry in enumerate(trajectory):
            if entry.get("is_reset", False):
                reset_indices.append(i)
            if entry.get("is_eos", False):
                has_eos = True
        
        # If no resets and no EOS, this is incomplete - skip
        if not reset_indices and not has_eos:
            logger.warning(f"Session {log_file} has no resets and no EOS, may be incomplete")
            return []
        
        # Split trajectory at reset points
        sub_trajectories = []
        start_idx = 0
        
        # Add all segments between resets
        for reset_idx in reset_indices:
            if reset_idx > start_idx:  # Only add non-empty segments
                sub_trajectories.append(trajectory[start_idx:reset_idx])
            start_idx = reset_idx + 1  # Start new segment after the reset
        
        # Add the final segment if it's not empty
        if start_idx < len(trajectory):
            sub_trajectories.append(trajectory[start_idx:])
        
        # Process each sub-trajectory
        processed_ids = []
        for i, sub_traj in enumerate(sub_trajectories):
            # Skip segments with no interaction data (just control messages)
            if not any(not entry.get("is_reset", False) and not entry.get("is_eos", False) for entry in sub_traj):
                continue
            
            # Get the next ID for this sub-trajectory
            cursor.execute("SELECT value FROM config WHERE key = 'next_id'")
            next_id = int(cursor.fetchone()[0])
            # Identify doom regions and validate on desktop via docker re-rendering
            doom_regions = find_doom_regions_on_desktop(sub_traj, clean_state, max_gap_frames=3)
            # Build spans (type, start, end) inclusive
            spans: List[Tuple[str, int, int]] = []
            cur = 0
            for (s, e) in doom_regions:
                if cur < s:
                    spans.append(("normal", cur, s - 1))
                spans.append(("doom", s, e))
                cur = e + 1
            if cur <= len(sub_traj) - 1:
                spans.append(("normal", cur, len(sub_traj) - 1))
            # Build concatenated normal events
            normal_entries: List[Dict[str, Any]] = []
            normal_spans_meta: List[int] = []
            for t, s, e in [(t, s, e) for (t, s, e) in spans if t == "normal"]:
                seg = sub_traj[s:e + 1]
                # store length for partitioning later
                seg_len = sum(1 for z in seg if not z.get("is_reset") and not z.get("is_eos"))
                normal_spans_meta.append(seg_len)
                normal_entries.extend(seg)
            formatted_events = format_trajectory_for_processing(normal_entries)
            record_num = next_id
            # Process normal events via docker
            assert len(formatted_events) > 0, 'should not happen that no normal events are formatted'
            try:
                args = (record_num, formatted_events)
                process_trajectory(args, SCREEN_WIDTH, SCREEN_HEIGHT, clean_state, MEMORY_LIMIT)
            except Exception as e:
                logger.error(f"Failed to process normal events for trajectory {record_num}: {e}")
                continue
            video_file = f'raw_data/raw_data/videos/record_{record_num}.mp4'
            action_file = f'raw_data/raw_data/actions/record_{record_num}.csv'
            # Combined output writer (doom + normal, in chronological order)
            doom_combined_file = f'raw_data/raw_data/videos/record_{record_num}_doom.mp4'
            doom_action_file = f'raw_data/raw_data/actions/record_{record_num}_doom.csv'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            combined_fps = 15
            doom_writer = cv2.VideoWriter(doom_combined_file, fourcc, combined_fps, (SCREEN_WIDTH, SCREEN_HEIGHT))
            sink = wds.TarWriter(os.path.join(OUTPUT_DIR, f'record_{record_num}.tar'))
            # Map and targets
            mapping_dict: Dict[Tuple[int, int], Tuple[int, int, bool, bool, List[str]]] = {}
            target_data: List[Tuple[int, int]] = []
            doom_rows: List[Dict[str, Any]] = []
            image_num_global = 0
            # Prepare reading normal video frames sequentially
            normal_frame_iter = None
            normal_mouse_df = None
            total_normal_frames = 0
            try:
                video = VideoFileClip(video_file)
                normal_mouse_df = pd.read_csv(action_file)
                total_normal_frames = int(round(video.fps * video.duration))
                #import pdb; pdb.set_trace()
                assert total_normal_frames == len(normal_mouse_df), 'should not happen that total_normal_frames does not match len(normal_mouse_df)'
                assert total_normal_frames == len(formatted_events), 'should not happen that total_normal_frames does not match len(formatted_events)'
                # generator over frames
                def _frame_gen():
                    for j in range(total_normal_frames):
                        yield video.get_frame(j / video.fps)
                normal_frame_iter = _frame_gen()
            except Exception as e:
                logger.error(f"Error opening normal video for trajectory {record_num}: {e}")
                normal_frame_iter = None
                normal_mouse_df = None
                continue
            # Iterate spans in order and write frames
            try:
                for span_type, s, e in spans:
                    seg_entries = [x for x in sub_traj[s:e + 1] if not x.get("is_reset") and not x.get("is_eos")]
                    if len(seg_entries) == 0:
                        continue
                    if span_type == "normal":
                        # For each entry in this normal span, pull next frame from normal video
                        for entry in seg_entries:
                            if normal_frame_iter is None:
                                raise RuntimeError("Normal frame iterator is not available")
                            frame_rgb = next(normal_frame_iter)
                            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            # Overlay doom icon if this is desktop
                            if is_desktop_frame(frame_bgr, _DESKTOP_REF_BGR):
                                frame_bgr = overlay_doom_icon_on_frame(frame_bgr)
                            doom_writer.write(frame_bgr)
                            # Encode latent from modified frame
                            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                            image_array = (frame / 127.5 - 1.0).astype(np.float32)
                            images_tensor = torch.tensor(image_array).unsqueeze(0)
                            images_tensor = rearrange(images_tensor, 'b h w c -> b c h w').to(device)
                            posterior = autoencoder.encode(images_tensor)
                            latents = posterior.sample().cpu()
                            latent = latents[0]
                            key = str(image_num_global)
                            latent_bytes = io.BytesIO()
                            np.save(latent_bytes, latent.numpy())
                            latent_bytes.seek(0)
                            sink.write({"__key__": key, "npy": latent_bytes.getvalue()})
                            # Build mapping entry from original entry
                            inputs = entry.get("inputs", {})
                            x = int(min(max(0, inputs.get("x", 0)), SCREEN_WIDTH - 1))
                            y = int(min(max(0, inputs.get("y", 0)), SCREEN_HEIGHT - 1))
                            left_click = bool(inputs.get("is_left_click", False))
                            right_click = bool(inputs.get("is_right_click", False))
                            down_keys = set([])
                            for ks, k in [("keys_down", "keydown"), ("keys_up", "keyup")]:
                                for key_name in inputs.get(ks, []) or []:
                                    kname = key_name.lower()
                                    if kname in KEYMAPPING:
                                        kname = KEYMAPPING[kname]
                                    if kname in stoi:
                                        if k == "keydown":
                                            down_keys.add(kname)
                                        elif k == "keyup" and kname in down_keys:
                                            down_keys.remove(kname)
                            mapping_dict[(record_num, image_num_global)] = (x, y, left_click, right_click, list(down_keys))
                            target_data.append((record_num, image_num_global))
                            # Collect doom CSV row (normal span)
                            ts = image_num_global / float(combined_fps)
                            # Construct key events list for this tick from inputs
                            key_events_list = []
                            kd = inputs.get("keys_down", []) or []
                            ku = inputs.get("keys_up", []) or []
                            for kk in kd:
                                key_events_list.append(("keydown", str(kk)))
                            for kk in ku:
                                key_events_list.append(("keyup", str(kk)))
                            doom_rows.append({
                                'Timestamp': ts,
                                'Timestamp_formated': f"0:{int(round(ts*1000))}",
                                'X': x,
                                'Y': y,
                                'Left Click': bool(left_click),
                                'Right Click': bool(right_click),
                                'Key Events': str(key_events_list),
                            })
                            image_num_global += 1
                    else:
                        # Doom span: simulate via VizDoom
                        action_seq = build_action_seq_for_doom(seg_entries)
                        try:
                            doom_frames_rgb = run_vizdoom_segment(action_seq, fps_skip=3)
                        except Exception as e:
                            logger.error(f"Error running VizDoom segment: {e}")
                            doom_frames_rgb = []
                        #import pdb; pdb.set_trace()
                        assert len(doom_frames_rgb) == len(seg_entries), 'should not happen that doom frames are missing'
                        for idx_local, entry in enumerate(seg_entries):
                            if idx_local < len(doom_frames_rgb):
                                frame_rgb = doom_frames_rgb[idx_local]
                            else:
                                # Fallback: black frame if missing
                                assert False, 'should not happen that doom frames are missing'
                            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            doom_writer.write(frame_bgr)
                            # Encode latent
                            frame = frame_rgb
                            image_array = (frame / 127.5 - 1.0).astype(np.float32)
                            images_tensor = torch.tensor(image_array).unsqueeze(0)
                            images_tensor = rearrange(images_tensor, 'b h w c -> b c h w').to(device)
                            posterior = autoencoder.encode(images_tensor)
                            latents = posterior.sample().cpu()
                            latent = latents[0]
                            key = str(image_num_global)
                            latent_bytes = io.BytesIO()
                            np.save(latent_bytes, latent.numpy())
                            latent_bytes.seek(0)
                            sink.write({"__key__": key, "npy": latent_bytes.getvalue()})
                            # Mapping from original entry
                            inputs = entry.get("inputs", {})
                            x = int(min(max(0, inputs.get("x", 0)), SCREEN_WIDTH - 1))
                            y = int(min(max(0, inputs.get("y", 0)), SCREEN_HEIGHT - 1))
                            left_click = bool(inputs.get("is_left_click", False))
                            right_click = bool(inputs.get("is_right_click", False))
                            down_keys = set([])
                            for ks, k in [("keys_down", "keydown"), ("keys_up", "keyup")]:
                                for key_name in inputs.get(ks, []) or []:
                                    kname = key_name.lower()
                                    if kname in KEYMAPPING:
                                        kname = KEYMAPPING[kname]
                                    if kname in stoi:
                                        if k == "keydown":
                                            down_keys.add(kname)
                                        elif k == "keyup" and kname in down_keys:
                                            down_keys.remove(kname)
                            mapping_dict[(record_num, image_num_global)] = (x, y, left_click, right_click, list(down_keys))
                            target_data.append((record_num, image_num_global))
                            # Collect doom CSV row (doom span)
                            ts = image_num_global / float(combined_fps)
                            # Construct key events list for this tick
                            key_events_list = []
                            kd = inputs.get("keys_down", []) or []
                            ku = inputs.get("keys_up", []) or []
                            for kk in kd:
                                key_events_list.append(("keydown", str(kk)))
                            for kk in ku:
                                key_events_list.append(("keyup", str(kk)))
                            doom_rows.append({
                                'Timestamp': ts,
                                'Timestamp_formated': f"0:{int(round(ts*1000))}",
                                'X': x,
                                'Y': y,
                                'Left Click': bool(left_click),
                                'Right Click': bool(right_click),
                                'Key Events': str(key_events_list),
                            })
                            image_num_global += 1
            finally:
                doom_writer.release()
                if 'video' in locals():
                    try:
                        video.close()
                    except Exception:
                        pass
                try:
                    sink.close()
                except Exception:
                    pass
            # Save combined actions CSV for _doom stream
            os.makedirs(os.path.dirname(doom_action_file), exist_ok=True)
            df = pd.DataFrame(doom_rows, columns=[
                'Timestamp','Timestamp_formated','X','Y','Left Click','Right Click','Key Events'
            ])
            # atomic write
            tmp_csv = doom_action_file + ".temp"
            df.to_csv(tmp_csv, index=False)
            os.replace(tmp_csv, doom_action_file)
            # Save mapping dict (merge if exists)
            if os.path.exists(os.path.join(OUTPUT_DIR, 'image_action_mapping_with_key_states.pkl')):
                with open(os.path.join(OUTPUT_DIR, 'image_action_mapping_with_key_states.pkl'), 'rb') as f:
                    existing_mapping_dict = pickle.load(f)
                for key, value in existing_mapping_dict.items():
                    if key not in mapping_dict:
                        mapping_dict[key] = value
            temp_path = os.path.join(OUTPUT_DIR, 'image_action_mapping_with_key_states.pkl.temp')
            with open(temp_path, 'wb') as f:
                pickle.dump(mapping_dict, f)
            os.rename(temp_path, os.path.join(OUTPUT_DIR, 'image_action_mapping_with_key_states.pkl'))
            # Save target frames CSV (merge + dedup)
            target_df = pd.DataFrame(target_data, columns=['record_num', 'image_num'])
            if os.path.exists(os.path.join(OUTPUT_DIR, 'train_dataset.target_frames.csv')):
                existing_target_data = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_dataset.target_frames.csv'))
                target_df = pd.concat([existing_target_data, target_df])
            target_df = target_df.drop_duplicates()
            temp_path = os.path.join(OUTPUT_DIR, 'train_dataset.target_frames.csv.temp')
            target_df.to_csv(temp_path, index=False)
            os.rename(temp_path, os.path.join(OUTPUT_DIR, 'train_dataset.target_frames.csv'))
            # Mark processed segment
            start_time = sub_traj[0]["timestamp"]
            end_time = sub_traj[-1]["timestamp"]
            cursor.execute(
                """INSERT INTO processed_segments 
                   (log_file, client_id, segment_index, start_time, end_time, 
                    processed_time, trajectory_id) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (log_file, client_id, i, start_time, end_time, datetime.now().isoformat(), record_num)
            )
            cursor.execute("UPDATE config SET value = ? WHERE key = 'next_id'", (str(record_num + 1),))
            conn.commit()
            processed_ids.append(record_num)
        if processed_ids:
            try:
                cursor.execute(
                    "INSERT INTO processed_sessions (log_file, client_id, processed_time) VALUES (?, ?, ?)",
                    (log_file, client_id, datetime.now().isoformat())
                )
                conn.commit()
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        return processed_ids
    except Exception as e:
        logger.error(f"Error processing session {log_file}: {e}")
        if conn:
            conn.rollback()
        return []
    finally:
        if conn:
            conn.close()

# Desktop padding latent
def ensure_padding_latent():
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'padding.npy')):
        logger.info("Creating padding image...")
        padding_data = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.float32)
        padding_tensor = torch.tensor(padding_data).unsqueeze(0)
        padding_tensor = rearrange(padding_tensor, 'b h w c -> b c h w').to(device)
        posterior = autoencoder.encode(padding_tensor)
        latent = posterior.sample()
        latent = torch.zeros_like(latent).squeeze(0)
        np.save(os.path.join(OUTPUT_DIR, 'padding.tmp.npy'), latent.cpu().numpy())
        os.rename(os.path.join(OUTPUT_DIR, 'padding.tmp.npy'), os.path.join(OUTPUT_DIR, 'padding.npy'))

def is_session_complete(log_file):
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("is_eos", False):
                        return True
                except json.JSONDecodeError:
                    continue
        return False
    except Exception as e:
        logger.error(f"Error checking if session {log_file} is complete: {e}")
        return False

def is_session_valid(log_file):
    try:
        entry_count = 0
        has_non_eos = False
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entry_count += 1
                    if not entry.get("is_eos", False) and not entry.get("is_reset", False):
                        has_non_eos = True
                except json.JSONDecodeError:
                    continue
        return entry_count > 0 and has_non_eos
    except Exception as e:
        logger.error(f"Error checking if session {log_file} is valid: {e}")
        return False

def main():
    global running
    ensure_padding_latent()
    initialize_database()
    
    # Initialize clean Docker state
    logger.info("Initializing clean container state...")
    clean_state = initialize_clean_state()
    logger.info(f"Clean state initialized: {clean_state}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Starting continuous monitoring for new sessions (check interval: {CHECK_INTERVAL} seconds)")
    try:
        # Main monitoring loop
        while running:
            try:
                # Find all log files
                log_files = glob.glob(os.path.join(FRAMES_DIR, "session_*.jsonl"))
                logger.info(f"Found {len(log_files)} log files")
                
                # Filter for complete sessions
                complete_sessions = [f for f in log_files if is_session_complete(f)]
                logger.info(f"Found {len(complete_sessions)} complete sessions")
                
                # Sort sessions by the numeric timestamp in the filename (session_<timestamp>_*.jsonl)
                def _extract_ts(path):
                    """Return int timestamp from session_<ts>_<n>.jsonl; fallback to 0 if parse fails."""
                    try:
                        basename = os.path.basename(path)  # session_1750138392_3.jsonl
                        ts_part = basename.split('_')[1]   # '1750138392'
                        return int(ts_part)
                    except Exception:
                        return 0
                complete_sessions.sort(key=_extract_ts)
                conn = sqlite3.connect(DB_FILE)
                cursor = conn.cursor()
                cursor.execute("SELECT log_file FROM processed_sessions")
                processed_files = set(row[0] for row in cursor.fetchall())
                conn.close()
                new_sessions = [f for f in complete_sessions if f not in processed_files]
                logger.info(f"Found {len(new_sessions)} new sessions to process")
                valid_sessions = [f for f in new_sessions if is_session_valid(f)]
                logger.info(f"Found {len(valid_sessions)} valid new sessions to process")
                total_trajectories = 0
                for log_file in valid_sessions:
                    if not running:
                        logger.info("Shutdown in progress, stopping processing")
                        break
                    logger.info(f"Processing session file: {log_file}")
                    processed_ids = process_session_file(log_file, clean_state)
                    total_trajectories += len(processed_ids)
                if total_trajectories > 0:
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute("SELECT value FROM config WHERE key = 'next_id'")
                    next_id = int(cursor.fetchone()[0])
                    conn.close()
                    logger.info(f"Processing cycle complete. Generated {total_trajectories} new trajectories.")
                    logger.info(f"Next ID will be {next_id}")
                else:
                    logger.info("No new trajectories processed in this cycle")
                remaining_sleep = CHECK_INTERVAL
                while remaining_sleep > 0 and running:
                    sleep_chunk = min(5, remaining_sleep)
                    time.sleep(sleep_chunk)
                    remaining_sleep -= sleep_chunk
            except Exception as e:
                logger.error(f"Error in processing cycle: {e}")
                time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    finally:
        logger.info("Shutting down trajectory processor")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
