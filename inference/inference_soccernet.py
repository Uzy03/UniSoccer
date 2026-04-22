import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.soccernet_clip_dataset import SoccerNetClipDataset
from model.MatchVision_classifier import MatchVision_Classifier


def main():
    parser = argparse.ArgumentParser(description='Inference on SoccerNet clips')
    parser.add_argument('--json_path', type=str, required=True, help='Path to JSON file with clip data')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints/pretrained_classification.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--out_csv', type=str, default='inference/soccernet_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers (0 = main process only)')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='先頭N件だけ使う（0=全件）')
    args = parser.parse_args()
    
    dataset = SoccerNetClipDataset(args.json_path)
    if args.max_samples > 0:
        dataset.data = dataset.data[:args.max_samples]
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    model = MatchVision_Classifier()
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device).eval()
    
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    
    csv_file = open(args.out_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['gameTime', 'caption_gt', 'caption_pred', 'correct'])
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            frames, caption_idx, video_paths, captions, game_times = batch
            logits = model.get_logits(frames.to(args.device))
            preds = logits.argmax(dim=1)
            
            for i in range(len(captions)):
                pred_label = model.keywords[preds[i].item()]
                is_correct = int(pred_label == captions[i])
                csv_writer.writerow([
                    game_times[i],
                    captions[i],
                    pred_label,
                    is_correct
                ])
                
                total_correct += is_correct
                total_samples += 1
    
    csv_file.close()
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f'Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})')


if __name__ == '__main__':
    main()
