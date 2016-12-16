nico-illust tag prediction
==================================

必要なモデルファイル、タグ一覧は [nico-opendata.jp](http://nico-opendata.jp) からダウンロードしてください。
解凍して出来たディレクトリ下の `v1/` 下にある `model.npz` 及び `tags.txt` をカレントディレクトリにコピーして使います。

USAGE
--------------

依存ライブラリのインストール

```sh
pip install -r requirements.txt
```

CPUで実行

```sh
python predict_tag.py \
  --gpu=-1 \
  --tags=tags.txt
  --model=model.npz
  http://lohas.nicoseiga.jp/thumb/4313120i
# tag: 川内改二 / score: 0.9832866787910461
# tag: 艦これ / score: 0.9811543226242065
# tag: 夜戦忍者 / score: 0.934027910232544
#    :
# と出力
```

GPUで実行

```sh
python predict_tag.py \
  --gpu=0 \
  --tags=tags.txt
  --model=model.npz
  http://lohas.nicoseiga.jp/thumb/4313120i
```

## License

MIT License (see `LICENSE` file).
