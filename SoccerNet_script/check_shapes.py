import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.MatchVision_classifier import MatchVision_Classifier

model = MatchVision_Classifier()
ckpt = torch.load('checkpoints/pretrained_classification.pth', map_location='cpu')
state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

model_sd = model.state_dict()
mismatched = [
    (k, model_sd[k].shape, state_dict[k].shape)
    for k in model_sd
    if k in state_dict and model_sd[k].shape != state_dict[k].shape
]

print(f'shape不一致: {len(mismatched)}件')
for k, ms, cs in mismatched:
    print(f'  {k}: model={ms}, ckpt={cs}')

if not mismatched:
    print('全キーのshapeが一致しています')
