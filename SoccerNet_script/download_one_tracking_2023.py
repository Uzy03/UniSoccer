import argparse
import os

import SoccerNet.Downloader as SNDown
from SoccerNet.utils import getListGames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./SoccerNet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--game_index", type=int, default=0)
    parser.add_argument('--password', default=os.environ.get('SOCCERNET_PASSWORD'))
    args = parser.parse_args()

    downloader = SNDown.SoccerNetDownloader(LocalDirectory=args.local_dir)
    if args.password:
        downloader.password = args.password

    games = getListGames(split=args.split)
    one_game = games[args.game_index]
    print(one_game)

    game_dir = os.path.join(args.local_dir, one_game)
    if os.path.exists(os.path.join(game_dir, "tracking-2023")) or os.path.exists(os.path.join(game_dir, "tracking_2023")):
        print("skip")
        return

    orig = SNDown.getListGames
    SNDown.getListGames = lambda split, task="tracking-2023": [one_game] if split == args.split else []
    try:
        downloader.downloadDataTask(
            task="tracking-2023",
            split=[args.split],
        )
    finally:
        SNDown.getListGames = orig


if __name__ == "__main__":
    main()
