# hihobot
自分のチャットボットを作る

## 作り方
### 必要なライブラリの準備
```bash
pip install -r requirements.txt
```

### 必要なファイルの準備
学習済みの[doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html)モデルが必要です。
私は、[pixiv小説で学習されたモデル](https://github.com/pixiv/pixivnovel2vec/releases)をお借りしました。
下の３ファイルが必要です。
```
doc2vec.model
doc2vec.model.syn0.npy
doc2vec.model.syn1.npy
```

### 学習データセットの準備
#### １．文章ごとに改行で区切られたテキストファイルを作成
```text
サンプルテキスト１
サンプルテキスト２
```
[マストドン](https://github.com/tootsuite/mastodon)の場合は、outbox機能を使ってテキストファイルを作るコードがあります。
（[extract_text_from_mastodon.py](extract_text_from_mastodon.py)）

#### ２．データセット作成用のコマンドを実行
[make_dataset.py](make_dataset.py)

#### 3.設定ファイルを用意
<details><summary>サンプル設定ファイル.json</summary>
  
```json
{
  "dataset": {
    "char_path": "/path/to/dataset_char.json",
    "text_path": "/path/to/dataset_text.json",
    "doc2vec_model_path": "/path/to/doc2vec.model",
    "seed": 0,
    "num_test": 100
  },
  "network": {
    "n_layers": 2,
    "in_size": 2148,
    "hidden_size": 128,
    "out_size": 2049,
    "dropout": 0.2
  },
  "loss": {
  },
  "train": {
    "batchsize": 50,
    "gpu": 0,
    "log_iteration": 100,
    "prune_iteration": 10000,
    "snapshot_iteration": 10000,
    "stop_iteration": 100000,
    "optimizer": {
      "name": "adam"
    },
    "optimizer_gradient_clipping": 5.0,
    "linear_shift": {
      "attr": "alpha",
      "value_range": [
        0.01,
        0.001
      ],
      "time_range": [
        2000,
        50000
      ]
    }
  },
  "project": {
    "name": "",
    "tags": []
  }
}
```

</details>

#### 4.学習
[train.py](train.py)で学習できます。

### 文章の生成
[generate.py](generate.py)で文章を生成できます。

## その他
[hihobot_tts](https://github.com/Hiroshiba/hihobot-tts/tree/master/hihobot_tts)でAPIとして使えます。

## ライセンス
MIT License
