import argparse
import os

import SoccerNet.Downloader as SNDown
from SoccerNet.utils import getListGames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./SoccerNet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--game_index", type=int, default=0)
    args = parser.parse_args()

    downloader = SNDown.SoccerNetDownloader(LocalDirectory=args.local_dir)

    games = getListGames(split=args.split)
    one_game = games[args.game_index]
    print(one_game)

    game_dir = os.path.join(args.local_dir, one_game)
    expected_files = ["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"]
    if all(os.path.exists(os.path.join(game_dir, file_name)) for file_name in expected_files):
        print("skip")
        return

    orig = SNDown.getListGames
    SNDown.getListGames = lambda split, task="spotting": [one_game] if split == args.split else []
    try:
        downloader.downloadGames(
            files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"],
            split=[args.split],
            task="spotting",
            verbose=True,
        )
    finally:
        SNDown.getListGames = orig


if __name__ == "__main__":
    main()
