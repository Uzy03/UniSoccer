import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decord import VideoReader
import decord
decord.bridge.set_bridge('torch')
from PIL import Image

MATCH_DIR = 'SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley'

for half in [1, 2]:
    video_path = os.path.join(MATCH_DIR, f'{half}_720p.mkv')
    vr = VideoReader(video_path, num_threads=1)
    fps = vr.get_avg_fps()
    total_sec = len(vr) / fps
    print(f'Half {half}: FPS={fps:.2f}, Duration={total_sec:.1f}s ({total_sec/60:.1f}min)')

# ハーフ1のゴール（13:10 = 790秒）のフレームをJPEGに保存
vr1 = VideoReader(os.path.join(MATCH_DIR, '1_720p.mkv'), num_threads=1)
fps1 = vr1.get_avg_fps()
os.makedirs('results', exist_ok=True)

for label, event_sec in [('goal_13m10s', 790), ('ball_out_02m13s', 133), ('corner_03m02s', 182)]:
    frame_idx = int(event_sec * fps1)
    frame = vr1.get_batch([frame_idx]).numpy()[0]  # (H, W, C) uint8
    img = Image.fromarray(frame)
    out_path = f'results/debug_{label}.jpg'
    img.save(out_path)
    print(f'Saved {out_path} (frame #{frame_idx})')
