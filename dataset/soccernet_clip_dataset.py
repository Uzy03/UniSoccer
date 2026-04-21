import sys
import os
import json
import torch
from einops import rearrange
from torch.utils.data import Dataset
from decord import VideoReader
from dataset.video_utils_siglip import get_frame_indices, set_transform


class SoccerNetClipDataset(Dataset):
    def __init__(self, json_path, num_frames=30, sample='middle'):
        with open(json_path) as f:
            self.data = json.load(f)
        
        self.transform = set_transform()
        self.num_frames = num_frames
        self.sample = sample
        self.keywords = [
            'corner', 'goal', 'injury', 'own goal', 'penalty', 'penalty missed',
            'red card', 'second yellow card', 'substitution', 'start of game(half)',
            'end of game(half)', 'yellow card', 'throw in', 'free kick',
            'saved by goal-keeper', 'shot off target', 'clearance', 'lead to corner',
            'off-side', 'var', 'foul (no card)', 'statistics and summary',
            'ball possession', 'ball out of play'
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        video_reader = VideoReader(entry['video'], num_threads=1)
        fps = video_reader.get_avg_fps()
        
        start_frame = int(entry['start_ms'] / 1000 * fps)
        end_frame = min(int(entry['end_ms'] / 1000 * fps), len(video_reader) - 1)
        window_len = max(1, end_frame - start_frame)
        
        local_indices = get_frame_indices(self.num_frames, window_len, sample=self.sample)
        abs_indices = [start_frame + i for i in local_indices]
        
        frames = video_reader.get_batch(abs_indices).permute(0, 3, 1, 2)
        frames = torch.cat(
            [self.transform(images=f, return_tensors='pt')['pixel_values'] for f in frames],
            dim=0
        )
        frames = rearrange(frames, 't c h w -> c t h w')
        
        caption_idx = torch.tensor(
            self.keywords.index(entry['caption']) if entry['caption'] in self.keywords else -1,
            dtype=torch.long
        )
        
        return frames, caption_idx, entry['video'], entry['caption'], entry['gameTime']
