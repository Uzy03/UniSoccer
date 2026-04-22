#!/usr/bin/env python3
"""
Generate clip metadata JSON from SoccerNet Labels-v2.json
"""

import json
import argparse
import os
import sys
from pathlib import Path

LABEL_MAP = {
    'Goal': 'goal',
    'Ball out of play': 'ball out of play',
    'Throw-in': 'throw in',
    'Corner': 'corner',
    'Shots off target': 'shot off target',
    'Offside': 'off-side',
    'Clearance': 'clearance',
    'Foul': 'foul (no card)',
    'Yellow card': 'yellow card',
    'Red card': 'red card',
    'Substitution': 'substitution',
    'Indirect free-kick': 'free kick',
    'Direct free-kick': 'free kick',
    'Penalty': 'penalty',
    'Yellow->red card': 'second yellow card',
}


def main():
    parser = argparse.ArgumentParser(
        description='Generate clip metadata JSON from SoccerNet Labels-v2.json'
    )
    parser.add_argument(
        '--match_dir',
        required=True,
        help='Path to match directory'
    )
    parser.add_argument(
        '--window_sec',
        type=int,
        default=30,
        help='Clip window in seconds (default: 30)'
    )
    parser.add_argument(
        '--out',
        help='Output JSON path (default: match_dir/clip_dataset.json)'
    )

    args = parser.parse_args()

    match_dir = args.match_dir
    window_sec = args.window_sec
    output_path = args.out or os.path.join(match_dir, 'clip_dataset.json')

    labels_file = os.path.join(match_dir, 'Labels-v2.json')
    if not os.path.exists(labels_file):
        print(f'Error: {labels_file} not found', file=sys.stderr)
        sys.exit(1)

    with open(labels_file, 'r') as f:
        labels_data = json.load(f)

    clips = []
    skipped = 0

    for annotation in labels_data.get('annotations', []):
        label = annotation.get('label')

        if label not in LABEL_MAP:
            skipped += 1
            continue

        game_time = annotation.get('gameTime', '')
        half = int(game_time[0]) if game_time else 0

        position = int(annotation.get('position', 0))
        window_ms = window_sec * 1000
        half_window = window_ms // 2
        start_ms = max(0, position - half_window)
        end_ms = position + half_window

        video_path = os.path.join(match_dir, f'{half}_720p.mkv')

        clip_entry = {
            'video': video_path,
            'start_ms': start_ms,
            'end_ms': end_ms,
            'caption': LABEL_MAP[label],
            'label_original': label,
            'half': half,
            'gameTime': game_time,
        }

        clips.append(clip_entry)

    with open(output_path, 'w') as f:
        json.dump(clips, f, indent=2)

    saved_count = len(clips)
    print(f'Saved: {saved_count}, Skipped: {skipped}, Output: {output_path}')


if __name__ == '__main__':
    main()
