# 学習不要のCMax比較

ユーザー指定により、Gallego 2018とShiba / Aoki / Gallego 2022の両方を追加します。入力データは既存のMVSEC HDF5で、学習やモデル重みのダウンロードは不要です。

| 条件 | 実装 | 推定 |
|---|---|---|
| `patch2018` | [Gallego 2018 §2.1](https://arxiv.org/pdf/1804.01306)に基づく本リポジトリの再実装 | 32×32の独立パッチ内で一定速度、極性を合算したIWEの分散最大化 |
| `secrets2022` | [公式コード](https://github.com/tub-rip/event_based_optical_flow/tree/f3b899a001052d97ed68f0eec0478c4f6440c59c)を外部checkoutから直接使用 | 多時刻の勾配コントラスト・TV・粗密ピラミッド・Burgers時間発展 |

2018論文に今回のMVSEC報告値はありません。2022版も本実験は原論文の完全再現ではありません。256×256の既存パック、因果的入力、初期化と採点区間の変更があるため、論文値は参考として区別します。

## 共通評価

- `outdoor_day1`, `indoor_flying1/2/3` × dt1/dt4。先頭50kイベントを除外し、既存matched-v1と同じGT区間を採点。
- 2手法 × 2入力窓 × 4収録 × 2dt = **32条件**。ニューラルネットの追加学習ではありません。
- `matched1k`: 評価区間末尾までの直近1,000イベント。非再帰count/voxel5と入力の参照範囲を揃えます。CMaxは生タイムスタンプから速度を推定し、NNは最終stride変位を換算するため、時間の扱いまで同一とは主張しません。
- `long`: 屋内30k・屋外40kイベント。同じ長い窓を両CMaxに与えます。2022論文のイベント数に基づき、短い窓の情報不足を別要因として見る補助比較です。
- 両者とも未来イベントを使わず、窓ごとに初期化。前窓の推定値を持ち越しません。GTは最適化関数へ渡しません。
- 出力速度(u,v) [px/s] × GT区間長 [s]をGT変位と比較。公式コードのrow/column規約をu/vへ変換し、符号と時間単位をテストしています。
- GT有効かつ当該GT区間にイベントがある画素で、AEE・`3px AND 5% GT`外れ値率を計算。参考用に`3pxのみ`も保存。系列内は区間ごとのmacro平均です。
- 全画素へ推定値を返す。2018版の10イベント未満のパッチはゼロ速度として採点から除外せず、推定可能パッチ割合を別記。
- 計算時間、入力時間幅、ソルバー収束状態、seed、設定ハッシュ、ソース版を保存。精度だけで速度や収束成功を推定しません。
- パラメータは原実装の設定・合成データの単位検証に基づいて固定。MVSEC GTを見て探索・モデル選択は行いません。

## 実装上の選択と原論文との差分

2018版は重複しない32×32タイル、最近傍による画素への割当、双線形splat+sigma=1のGaussianぼかし、固定の粗い変位探索後Adam 80反復です。最大変位±32px、探索候補と学習率は `configs/cmax.json` に記録。これらは本再実装の選択で、著者のMVSEC実装とは呼びません。IWEのキャンバスを広げ、移動イベントがパッチ境界で失われる影響を抑えています。

2022版は2022年中の固定コミット `f3b899a001052d97ed68f0eec0478c4f6440c59c` の `configs/mvsec_indoor_burgers.yaml` を基準に、解像度・cropを256×256へ変更しています。公開設定は `scale=5`（コードのrangeは1〜4で実際は4段階）、Newton-CG最大25反復、初期探索40試行、TV係数0.01、time bins=10です。[論文本文 §4.1](https://arxiv.org/html/2207.10022v2#S4.SS1)の5段階・20反復・TV=0.0025・5binsとは異なるので、公開コード設定に基づく比較として扱います。

上流 `main.py` はイベント不足時にGT区間の前後へ入力を広げ、前窓の推定を初期値に使います。本評価は未来入力と履歴の持ち越しを無効化します。公式ソルバー本体は変更せず、アダプターでTPEのseed固定、収束記録、profiling/stdoutの抑制を行います。2022年実装はGPL-3.0であり、このリポジトリへコピーせず外部checkoutのライセンスを保持します。

## 実行

PyTorch CUDA環境に加え、torchvision（torchと対応する版）、SciPy、Optuna 2.10.1、scikit-image 0.24.0、plotly 5.24.1、OpenCV、matplotlibを使います。既存の学習用環境を変更しない専用venvで実行します。

```bash
git clone https://github.com/tub-rip/event_based_optical_flow.git external/secrets-event-flow
git -C external/secrets-event-flow checkout --detach f3b899a001052d97ed68f0eec0478c4f6440c59c
export PYTHONPATH="$PWD/src"
export EFBENCH_SECRETS_CHECKOUT="$PWD/external/secrets-event-flow"
python -m unittest discover -s tests -p test_cmax.py -v
python scripts/run_cmax.py --method patch2018 --profile matched1k \
  --sequence indoor_flying1 --dt 1 --manifest /path/to/mvsec.json \
  --output runs/cmax-smoke --limit 2
```

Izarでは専用checkoutで `EFBENCH_PYTHON`, `EFBENCH_MVSEC_MANIFEST` を設定して `sbatch scripts/slurm_izar_cmax.sh`。既定は最大4 GPU同時、1条件1 GPU・4 CPU・24 GB、壁時間71時間です。`EFBENCH_CMAX_SMOKE=1` は最初の2区間だけを測り、`smoke_completed` と記録します。本実験と同一視しません。

各条件に `identity.json`, `environment.json`, `window-XXXXX.json`, `windows.jsonl` を保存し、全区間を処理できたときだけ `completed.json` を作ります。区間ファイルを原子的に保存するため、`--resume` で完了済み区間を安全に飛ばせます。設定・ソース版が違えば再開を拒否します。単一seedのためばらつきを評価した実験ではありません。

結果集計は `python scripts/collect_cmax.py runs/cmax-JOB1 runs/cmax-JOB2 --output runs/cmax-report`。欠損条件を成功扱いせず終了コード1を返します。`--allow-incomplete` を明示すると途中表を生成できます。2022年のBurgers論文報告値は `configs/literature/cmax2022_table1.csv` に保存し、実測と並べますが同条件再現の合否判定には使いません。
