#!/usr/bin/env python3
import sys
import os
import csv
import json
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description='Extract video clips from results')
    parser.add_argument('--results_csv', type=str, default='results/commentary_results.csv',
                        help='Path to commentary_results.csv')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to clip_dataset.json')
    parser.add_argument('--out_dir', type=str, default='results/presentation',
                        help='Output directory for extracted clips')
    args = parser.parse_args()

    # Step 1: Read gameTime list from CSV
    game_times = []
    try:
        with open(args.results_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'gameTime' in row:
                    game_times.append(row['gameTime'])
    except FileNotFoundError:
        print(f"Error: Results CSV not found at {args.results_csv}")
        return

    # Step 2: Create clip_map from JSON
    clip_map = {}
    try:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    if 'gameTime' in entry:
                        clip_map[entry['gameTime']] = entry
            elif isinstance(data, dict):
                clip_map = data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {args.json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.json_path}")
        return

    # Step 3: Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Step 4: Extract clips using ffmpeg
    for game_time in game_times:
        if game_time not in clip_map:
            print(f"Skipping {game_time}: not found in clip_map")
            continue

        entry = clip_map[game_time]

        # Check required fields
        if 'start_ms' not in entry or 'end_ms' not in entry or 'video' not in entry:
            print(f"Skipping {game_time}: missing required fields in entry")
            continue

        start_sec = entry['start_ms'] / 1000.0
        duration_sec = (entry['end_ms'] - entry['start_ms']) / 1000.0

        # Create safe filename
        safe_name = game_time.replace(' - ', '_').replace(':', '_').replace(' ', '_')
        out_path = os.path.join(args.out_dir, safe_name + '.mp4')

        if os.path.exists(out_path):
            print(f'Skipping {game_time}: already exists')
            continue

        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-ss', str(start_sec),
            '-t', str(duration_sec),
            '-i', entry['video'],
            '-c:v', 'libx264', '-crf', '18',
            '-c:a', 'aac',
            out_path
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Extracted {game_time} -> {out_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting {game_time}: {e}")
            continue


if __name__ == '__main__':
    main()
