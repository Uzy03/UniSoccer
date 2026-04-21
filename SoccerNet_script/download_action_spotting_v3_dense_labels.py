import argparse
import os

import SoccerNet.Downloader as SNDown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./SoccerNet")
    parser.add_argument('--password', default=os.environ.get('SOCCERNET_PASSWORD'))
    args = parser.parse_args()

    downloader = SNDown.SoccerNetDownloader(LocalDirectory=args.local_dir)
    if args.password:
        downloader.password = args.password

    output_dir = os.path.join(args.local_dir, "action-spotting-v3")
    if os.path.exists(output_dir):
        print("skip")
        return

    downloader.downloadDataTask(
        task="action-spotting-v3",
        split=["train", "valid", "test"],
    )


if __name__ == "__main__":
    main()
