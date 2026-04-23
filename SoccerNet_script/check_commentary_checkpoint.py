import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

ckpt = torch.load('checkpoints/downstream_commentary_all_open.pth', map_location='cpu')
sd = ckpt.get('state_dict', ckpt)
sd = {k.replace('module.', '', 1): v for k, v in sd.items()}

# Count and display checkpoint information
total_keys = len(sd)
components = sorted(set(k.split('.')[0] for k in sd.keys()))
lora_keys = [k for k in sd.keys() if 'lora' in k]
llama_keys = [k for k in sd.keys() if 'llama_model' in k]
visual_encoder_keys = [k for k in sd.keys() if 'visual_encoder' in k]

print(f"チェックポイントの総キー数: {total_keys}")
print(f"コンポーネント一覧: {components}")
print(f"LoRA キー数: {len(lora_keys)}")
print(f"LLaMA キー数: {len(llama_keys)}")
print(f"visual_encoder キー数: {len(visual_encoder_keys)}")

if lora_keys:
    print(f"\nLoRA キーの先頭3件:")
    for key in lora_keys[:3]:
        print(f"  {key}")
    print("⚠️  open_llm_decoder=True が必要")
else:
    print("\n✓ open_llm_decoder=False で OK")
