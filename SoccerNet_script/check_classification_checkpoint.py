import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.MatchVision_classifier import MatchVision_Classifier


def main():
    ckpt_path = 'checkpoints/pretrained_classification.pth'
    
    print(f"Loading checkpoint from: {ckpt_path}")
    print("-" * 80)
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    
    # Extract num_classes
    num_classes_ckpt = state_dict['classifier.weight'].shape[0]
    
    # Print checkpoint keys
    print(f"✓ Checkpoint keys: {len(state_dict)} items")
    for i, key in enumerate(sorted(state_dict.keys()), 1):
        print(f"  {i}. {key} {state_dict[key].shape}")
    
    print("-" * 80)
    print(f"✓ Num classes from checkpoint: {num_classes_ckpt}")
    
    # Create model
    model = MatchVision_Classifier()
    print(f"✓ Model keywords count: {len(model.keywords)}")
    
    # Adjust keywords if needed
    if len(model.keywords) != num_classes_ckpt:
        print(f"⚠️ Keywords mismatch: {len(model.keywords)} != {num_classes_ckpt}")
        keywords = [k for k in model.keywords if k != 'ball possession']
        model = MatchVision_Classifier(keywords=keywords[:num_classes_ckpt])
        print(f"✓ Adjusted model keywords count: {len(model.keywords)}")
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print("-" * 80)
    print(f"Missing keys: {len(missing)} items")
    if missing:
        for i, key in enumerate(sorted(missing), 1):
            print(f"  {i}. {key}")
    
    print(f"\nUnexpected keys: {len(unexpected)} items")
    if unexpected:
        for i, key in enumerate(sorted(unexpected), 1):
            print(f"  {i}. {key}")
    
    print("-" * 80)
    if len(missing) == 0 and len(unexpected) == 0:
        print("✅ 全重みが正常にロードされました")
    else:
        if len(missing) > 0:
            print(f"⚠️ {len(missing)}個のキーがロードされていません")
        if len(unexpected) > 0:
            print(f"⚠️ {len(unexpected)}個の予期しないキーが検出されました")


if __name__ == '__main__':
    main()
