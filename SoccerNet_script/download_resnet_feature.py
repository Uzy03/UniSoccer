import SoccerNet
import SoccerNet.Downloader as SNDown
from SoccerNet.utils import getListGames
import os

d = SNDown.SoccerNetDownloader(LocalDirectory="./SoccerNet")

# train の最初の1試合だけ、など
one_game = getListGames(split="train")[0]
print("target:", one_game)

game_dir = os.path.join("./SoccerNet", one_game)
expected_files = ["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"]
if all(os.path.exists(os.path.join(game_dir, file_name)) for file_name in expected_files):
    print("skip")
    raise SystemExit(0)

# Downloader.py 側が参照する getListGames を差し替える
orig = SNDown.getListGames
SNDown.getListGames = lambda split, task="spotting": [one_game] if split in ["train"] else []

try:
    d.downloadGames(
        files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"],
        split=["train"],
        task="spotting",
        verbose=True,
    )
finally:
    SNDown.getListGames = orig
