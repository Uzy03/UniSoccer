import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import argparse
import json
import torch
from einops import rearrange
from decord import VideoReader
from tqdm import tqdm
from dataset.video_utils_siglip import get_frame_indices, set_transform
from model.matchvoice_model_all_blocks import matchvoice_model_all_blocks


def load_clip_frames(entry, num_frames=30):
    """
    Load frames from a video clip.
    
    Args:
        entry: dict with 'video', 'start_ms', 'end_ms' keys
        num_frames: number of frames to extract
    
    Returns:
        frames: tensor of shape [C, T, H, W]
    """
    vr = VideoReader(entry['video'], num_threads=1)
    fps = vr.get_avg_fps()
    start_frame = int(entry['start_ms'] / 1000 * fps)
    end_frame = min(int(entry['end_ms'] / 1000 * fps), len(vr) - 1)
    
    window_len = max(1, end_frame - start_frame)
    local_indices = get_frame_indices(num_frames, window_len, sample='middle')
    abs_indices = [start_frame + i for i in local_indices]
    
    frames = vr.get_batch(abs_indices).permute(0, 3, 1, 2)  # [T, C, H, W]
    
    transform = set_transform()
    frames = torch.cat(
        [transform(images=f, return_tensors='pt')['pixel_values'] for f in frames],
        dim=0
    )
    frames = rearrange(frames, 't c h w -> c t h w')  # [C, T, H, W]
    
    return frames


def main():
    parser = argparse.ArgumentParser(
        description='Generate commentary for correct clips using matchvoice model with instruction'
    )
    parser.add_argument('--results_csv', type=str, required=True,
                        help='Path to classification inference results CSV')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to clip_dataset.json')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/downstream_commentary_all_open.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--llm_ckpt', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Local path or HuggingFace ID for LLaMA-3-8B-Instruct')
    parser.add_argument('--out_csv', type=str, default='results/commentary_results.csv',
                        help='Path to output CSV')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    parser.add_argument('--config', type=str, default='configs/instruction_explain.json',
                        help='Path to instruction config JSON')
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    instruction = config.get('instruction', '')
    max_new_tokens = config.get('max_new_tokens', 128)
    if 'out_csv' in config:
        args.out_csv = config['out_csv']
    
    # Step 1: Get gameTime list of correct clips
    correct_game_times = []
    with open(args.results_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('correct') == '1':
                correct_game_times.append(row['gameTime'])
    
    if len(correct_game_times) == 0:
        print("No correct clips found.")
        return
    
    # Step 2: Get entries from clip_dataset.json
    with open(args.json_path, 'r') as f:
        data = json.load(f)
    
    clip_map = {entry['gameTime']: entry for entry in data}
    entries = []
    for gt in correct_game_times:
        if gt in clip_map:
            entries.append(clip_map[gt])
    
    # Step 3: Load model
    model = matchvoice_model_all_blocks(
        load_checkpoint=False,
        num_features=768,
        need_temporal='yes',
        llm_ckpt=args.llm_ckpt,
        tokenizer_ckpt=args.llm_ckpt,
        open_llm_decoder=True,
    )
    model.to(args.device)
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = dict([(k.replace('module.', '', 1), v) for k, v in state_dict.items()])
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict
    model.eval()
    
    model.instruction = instruction
    model.use_logits_filter = False
    model._max_new_tokens = max_new_tokens
    
    # Step 4: Inference on each correct clip
    results = []
    for entry in tqdm(entries, desc='Generating commentary'):
        try:
            frames = load_clip_frames(entry)
            frames_batch = frames.unsqueeze(0).to(args.device)  # [1, C, T, H, W]
            
            samples = {
                'frames': frames_batch,
                'labels': torch.zeros(1, 1, dtype=torch.long).to(args.device),
                'attention_mask': torch.ones(1, 1, dtype=torch.long).to(args.device),
                'input_ids': torch.zeros(1, 1, dtype=torch.long).to(args.device),
                'caption_text': [entry['caption']],
                'video_path': [entry['video']]
            }
            
            with torch.no_grad():
                temp_res_text, _, _ = model(samples, validating=True)
            
            generated = temp_res_text[0] if temp_res_text else ''
            
            print(f"gameTime: {entry['gameTime']}, generated: {generated}")
            
            results.append({
                'gameTime': entry['gameTime'],
                'caption_gt': entry['caption'],
                'instruction': instruction,
                'generated_commentary': generated
            })
        except Exception as e:
            print(f"Error processing {entry['gameTime']}: {e}")
            continue
    
    # Step 5: Output to CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['gameTime', 'caption_gt', 'instruction', 'generated_commentary'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {args.out_csv}")


if __name__ == '__main__':
    main()
