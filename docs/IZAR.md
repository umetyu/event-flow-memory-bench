# Izarでの実行

対象はStep 2の入力×記憶の5条件と、対応するFireNet系の先行確認です。SNNを含む全モデル一括学習は今回の実行範囲に含めません。

## 既存データの利用

データは複製せず、既存の著者配布HDF5を参照します。実行用コード・ログはユーザーのscratch配下の専用ディレクトリへ置き、既存実験を変更しません。重要な結果・チェックポイントはscratch以外へ退避してください。[SCITASのscratch保存規則](https://scitas-doc.epfl.ch/storage/scratch_fs/)には30日削除規則があります。

```bash
export EFBENCH_ROOT=/scratch/izar/USER/event-flow-memory-bench
export EFBENCH_UZH=/path/to/author_training
export EFBENCH_MVSEC=/path/to/MVSEC
export EFBENCH_PYTHON=/path/to/cuda-venv/bin/python
cd "$EFBENCH_ROOT"
module load gcc/11.3.0 python/3.10.4 cuda/11.8.0
export PYTHONPATH="$PWD/src:$PWD/external/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
"$EFBENCH_PYTHON" scripts/prepare_author_data.py \
  --uzh "$EFBENCH_UZH" --mvsec "$EFBENCH_MVSEC" --output runs/setup
```

著者ファイルの`<recording>_davis[_with_gt]_<crop>.h5`形式を検査し、crop番号と`with_gt`の有無を除いた元収録IDでgroupを固定します。未知の名前は停止するため、一般データには`efbench inventory --groups`を使ってください。列長、各ファイル先頭・末尾1kイベント、全GT時刻と端点GT形状を検査します。これは全イベント走査やアーカイブchecksum検査の代用ではありません。

2026-11-18の検査では、UZHは9収録・140本・計70Mイベント、全ファイル500kイベントでした。`uzh-all.json`は論文再現用の全140本です。`uzh-matched.json`はseed 0で学習116本、検証24本（`indoor_forward_8_davis`）です。全モデル・学習seedでこのsplitを共用し、MVSECをモデル選択に使いません。

MVSECの`outdoor_day1`、`indoor_flying1/2/3`は展開済みで、4収録とも256×256の`flow_dt1`と`flow_dt4`を確認しました。データのダウンロード・再展開は不要でした。

## 学習時間の実測

```bash
sbatch scripts/slurm_izar_benchmark.sh
```

PyTorch 2.5.1+cu121、Tesla V100-PCIE-32GB 1枚、CPU 4 threads、FP32、batch 8、128×128、1k入力、10k loss、元のAdam・勾配clip・augmentationで実測しました。20更新warmup後、100更新を計測しています。HDF5読み込み、入力生成、GPU転送、10回のforward、loss、backward、optimizerを含みます。モデル・loss・ローダーは固定した上流コードを使い、計測後の重みは捨てます。

| モデル | 平均秒/更新 | GPU peak allocation | 100 epoch・1 seedの推定 |
|---|---:|---:|---:|
| FireNet | 0.2889 | 3.84 GiB | 7.02時間 |
| RecEVFlowNet | 0.5553 | 4.02 GiB | 13.50時間 |

推定更新数は`70M × 100 / (batch 8 × loss 10k) = 87,500`です。系列境界の端数、初期化、保存、検証・評価、キュー待ちは含めていません。100 epochを実行済みという意味ではなく、短い計測からの外挿です。matched経路は検証セットとloss集約が異なるため、上の値をそのまま保証しません。他のアーキテクチャやvoxel入力にも流用しないでください。

## 配布重みでの評価

[著者README](https://github.com/tudelft/event_flow#download-models)のOneDriveはcurlでは403でしたが、通常のブラウザでは公開フォルダを確認できました。既存の`mlruns.zip`には同じ著者OneDriveから取得した履歴があり、それを再利用しました。ZIPのSHA-256は`9f65e47509031de116b248d1712fc7eb114145d368da8a1b0caf35151ad9ca1d`です。FireNetとEVFlowNetのrunだけを展開し、後者の`params/model`が`RecEVFlowNet`であることを確認しています。データ・重みはこのGitリポジトリに含めません。

```bash
export EFBENCH_MLRUNS='/path/to/extracted/mlruns 2/42'
sbatch scripts/slurm_izar_reproduce.sh
```

2モデル×dt1/dt4の4タスク、同時最大2GPUです。検証済みの著者配布pickleを読み、元の採点規約で4収録を評価します。結果JSONと論文表との差分JSONを別々に保存し、許容差外ならジョブは失敗状態になります。ランダム重みへのフォールバックはありません。

## Step 2の設定生成

```bash
"$EFBENCH_PYTHON" -m event_flow_bench.cli grid --stage pilot \
  --manifest runs/setup/uzh-matched.json --seeds 0 --output runs/setup/pilot
"$EFBENCH_PYTHON" -m event_flow_bench.cli grid --stage main \
  --manifest runs/setup/uzh-matched.json --seeds 0 --output runs/setup/main
```

各5条件を生成します。設定生成は学習の投入を行いません。最初のseedで実行経路と収束を確認した後、同じsplitを保った複数seed比較へ進みます。履歴窓を振る実験とAffineObjectsは別段階です。
