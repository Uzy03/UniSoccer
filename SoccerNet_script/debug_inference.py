import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from dataset.soccernet_clip_dataset import SoccerNetClipDataset
from model.MatchVision_classifier import MatchVision_Classifier

JSON_PATH = 'SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/clip_dataset.json'
CKPT_PATH = 'checkpoints/pretrained_classification.pth'

# モデルロード
ckpt = torch.load(CKPT_PATH, map_location='cpu')
state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
model = MatchVision_Classifier()
model.load_state_dict(state_dict, strict=False)
model.eval()

# temporal_alpha_attn が非ゼロか確認
alphas = [blk.temporal_alpha_attn.item() for blk in model.siglip_model.timesformer.resblocks]
print(f'temporal_alpha_attn (first 3 blocks): {alphas[:3]}')
print(f'temporal_alpha_attn tanh (first 3):   {[torch.tensor(a).tanh().item() for a in alphas[:3]]}')

# pixel_values の統計を確認
dataset = SoccerNetClipDataset(JSON_PATH)
frames, caption_idx, video_path, caption, game_time = dataset[0]
frames_batch = frames.unsqueeze(0)
print(f'\nframes shape: {frames_batch.shape}')
print(f'pixel_values min={frames_batch.min():.3f} max={frames_batch.max():.3f} mean={frames_batch.mean():.3f}')

# ゴールイベントを探してtop5予測を表示
goal_indices = [i for i, d in enumerate(dataset.data) if d['caption'] == 'goal']
check_indices = goal_indices[:2] + [0, 1, 2]  # goal2件 + 先頭3件
print()
with torch.no_grad():
    for idx in check_indices:
        entry = dataset.data[idx]
        frames, _, _, caption, game_time = dataset[idx]
        logits = model.get_logits(frames.unsqueeze(0))
        probs = F.softmax(logits, dim=1)[0]
        top5 = probs.topk(5)
        top5_labels = [(model.keywords[i], f'{v:.3f}') for i, v in zip(top5.indices.tolist(), top5.values.tolist())]
        print(f'[{game_time}] gt={caption}')
        print(f'  top5: {top5_labels}')
