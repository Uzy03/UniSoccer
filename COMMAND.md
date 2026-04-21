# Commands

## Docker

| コマンド | 説明 |
|---|---|
| `make build` | Dockerイメージをビルド（キャッシュなし） |
| `make run` | コンテナに入る（ワークスペース全体マウント） |
| `make clean` | ダングリングイメージ・ビルドキャッシュを削除 |

## データ前処理

| コマンド | 説明 |
|---|---|
| `make preprocess` | Labels-v2.json → clip_dataset.json を生成 |

変数上書き例:
```bash
make preprocess MATCH_DIR="SoccerNet/england_epl/2014-2015/別の試合"
```

## 推論

| コマンド | 説明 |
|---|---|
| `make inference` | Docker上でGPU推論を実行 → `inference/soccernet_results.csv` |

変数上書き例:
```bash
make inference BATCH_SIZE=8 OUT_CSV=inference/my_results.csv
```

## デフォルト変数

| 変数 | デフォルト値 |
|---|---|
| `IMAGE` | `unisoccer` |
| `MATCH_DIR` | `SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley` |
| `CKPT_PATH` | `checkpoints/pretrained_classification.pth` |
| `BATCH_SIZE` | `4` |
| `OUT_CSV` | `inference/soccernet_results.csv` |
