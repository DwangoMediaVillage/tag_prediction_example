nico-illust tag prediction
==================================

必要なモデルファイル、平均画像ファイル、タグ一覧は(nico-opendata.jp)<http://nico-opendata.jp>からダウンロードしてください

USAGE
--------------

```
pip install -r requirements.txt
python predict_tag.py \
  openmodel.dump\
  mean_resized.dump \
  character_series_sorted.txt \
  http://lohas.nicoseiga.jp/thumb/605863i
```
