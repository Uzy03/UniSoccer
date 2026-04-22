# Commands

## Docker

| コマンド | 説明 |
|---|---|
| `make build` | Dockerイメージをビルド（キャッシュなし） |
| `make run` | コンテナに入る（ワークスペース全体マウント） |
| `make clean` | ダングリングイメージ・ビルドキャッシュを削除 |

## データ転送（ローカル → spica）

| コマンド | 説明 |
|---|---|
| `make upload` | `SoccerNet/` をspicaに転送（デフォルト） |
| `make upload SRC=SoccerNet/england_epl/` | 特定ディレクトリを転送 |
| `make upload SRC="SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley"` | スペースを含むパスは引用符で囲む |
| `make upload SRC=checkpoints/pretrained_classification.pth` | 特定ファイルを転送 |

転送先: `ujihara@solar.arch.cs.kumamoto-u.ac.jp:/user/arch/ujihara/UniSoccer/`（ポート2222）

> **実装メモ**: `scp` はリモートパスのスペースを正しく扱えないため、`tar | ssh` 方式を使用。
> tar でアーカイブを作成しつつ SSH 経由でパイプし、リモートで展開する。ディレクトリ構造は自動で再現される。

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
| `make inference` | Docker上でGPU推論を実行（ローカルMac用） |
| `make inference_local` | Dockerコンテナ内から直接推論（spica上のコンテナ内で使用） |

変数上書き例:
```bash
# Docker外（ローカルMac）から実行
make inference BATCH_SIZE=8 OUT_CSV=results/my_results.csv

# Docker内（spicaのコンテナ内）から実行
make inference_local BATCH_SIZE=8 OUT_CSV=results/my_results.csv
```

## デフォルト変数

| 変数 | デフォルト値 |
|---|---|
| `IMAGE` | `unisoccer` |
| `MATCH_DIR` | `SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley` |
| `CKPT_PATH` | `checkpoints/pretrained_classification.pth` |
| `BATCH_SIZE` | `4` |
| `OUT_CSV` | `inference/soccernet_results.csv` |
