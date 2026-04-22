import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.MatchVision_classifier import MatchVision_Classifier

model = MatchVision_Classifier()
ckpt = torch.load('checkpoints/pretrained_classification.pth', map_location='cpu')
state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))

model_keys = set(model.state_dict().keys())
ckpt_keys = set(state_dict.keys())
missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys

print(f'モデルのキー数:          {len(model_keys)}')
print(f'チェックポイントのキー数: {len(ckpt_keys)}')
print(f'ロードされないキー:      {len(missing)}')
print(f'余分なキー:              {len(unexpected)}')
if missing:
    print('missing例:', list(missing)[:5])
if unexpected:
    print('unexpected例:', list(unexpected)[:5])
