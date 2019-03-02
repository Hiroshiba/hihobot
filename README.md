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

## 使い方
[generate.py](generate.py)でチェックできます。
[hihobot_tts](https://github.com/Hiroshiba/hihobot-tts/tree/master/hihobot_tts)でAPIとして使えます。

## ライセンス
MIT License
