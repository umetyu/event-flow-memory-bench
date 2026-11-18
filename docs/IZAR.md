# Izarでの実行

論文モデル比較は既報値・配布済み重みを使い、[56]は論文報告値とUZH再実装を別条件として保持します。[56]再実装はjob 3256469で100 epochとMVSEC評価を完了しましたが、検証RSAT約1で精度改善は未確認です。[専用手順](ZHU2019.md)と[最新の対象](BASELINES.md)。追加指示により、非再帰count / 5-bin voxelも[専用手順](INPUT_BASELINES.md)で学習します。既存5条件×3 seedsの一括gridとSNNは今回の実行範囲に含めません。

Table 1の最上段`EV-FlowNet* [54]`を1行目として数えると、比較対象は2・4・5・6行目、FireNet系は11・12・13行目です。2行目の[56]は「論文報告値（MVSEC outdoor_day2学習）」と「UZHで再学習」の2条件に分け、採点規約も明記します。既存Hagenaars実装の非再帰count / 5-bin voxelは別の要因切り分け用対照です。配布重み評価は選択した6モデルで完了し、学習速度の短時間計測はRecEVFlowNetとFireNetだけで行いました。

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

RNN・Leakyの残り4モデルは、同じ著者ZIPから対応するrunを展開したうえで次のように評価します。8タスクで全32条件（4モデル×dt1/dt4×4収録）を採点します。学習は行いません。

```bash
export EFBENCH_MODELS='RNNEVFlowNet LeakyEVFlowNet RNNFireNet LeakyFireNet'
sbatch --array=0-7%2 scripts/slurm_izar_reproduce.sh
```

モデルリストを変える場合は配列範囲も`0-(2×モデル数−1)`に合わせます。論文比較の許容差は既存のAEE 0.02 px・外れ値率0.2 percentage pointのままです。

2026-11-18の評価は全16条件（2モデル×2種類のdt×4収録）について完了しました。[全条件の比較CSV](../results/2026-11-18-izar/paper-comparison.csv)と[チェックポイントhash付き結果](../results/2026-11-18-izar/reproduction-summary.json)を公開しています。AEEは全16条件で絶対差0.00859 px以下（事前の許容差0.02 px）でした。外れ値率は14/16条件で許容差0.2 percentage point以内ですが、次の2条件は未一致です。

| 条件 | 論文の外れ値率 | 今回の外れ値率 | 差（percentage points） |
|---|---:|---:|---:|
| FireNet / dt4 / outdoor_day1 | 20.93% | 19.61904% | -1.31096 |
| RecEVFlowNet / dt4 / indoor_flying3 | 29.60% | 30.05828% | +0.45828 |

論文表を読み直し、参照CSVの転記誤りではないことを確認しました。両dt4比較は失敗判定を保持し、許容差や参照値を変更していません。**全指標の完全再現とは報告しません。** 元の配布物のPyTorchは1.4.0、今回は2.5.1であり、差の原因は現時点では未確定です。`scripts/diagnose_outliers.py`は同じ予測・同じmaskに対して閾値の定義を診断し、主評価の値を置き換えません。

[追加診断](../results/2026-11-18-izar/outlier-diagnostics.json)では、各収録を単独で評価しても主評価値は一致しました。GT相対5%、予測相対5%、絶対3 pxのみの3規約は、この2条件では同じ外れ画素を選びました（集計の浮動小数点差は4×10^-8 percentage points未満）。従ってこの差を閾値の分母や収録間の状態持ち越しだけで説明することはできません。旧実行環境との照合などは未実施です。

## Step 2の設定生成

```bash
"$EFBENCH_PYTHON" -m event_flow_bench.cli grid --stage pilot \
  --manifest runs/setup/uzh-matched.json --seeds 0 --output runs/setup/pilot
"$EFBENCH_PYTHON" -m event_flow_bench.cli grid --stage main \
  --manifest runs/setup/uzh-matched.json --seeds 0 --output runs/setup/main
```

各5条件を生成します。設定生成は学習の投入を行いません。このgrid全体は投入せず、今回指定された非再帰count / 5-bin voxelだけを専用Slurmスクリプトで実行します。[56]も別の専用スクリプトです。履歴窓を振る実験とAffineObjectsは別段階です。

## 配布重みの収録確認

著者ZIPにはTable 1のHagenaars 14モデル全てとFireFlowNet系3モデルの計17個のmodel.pthが含まれます。RNNEVFlowNetのparamsはRNNRecEVFlowNet、LeakyEVFlowNetはLeakyRecEVFlowNetであることを確認しました。[56]の重みは含まれません。重みの存在確認と、その重みでの評価完了は区別します。

## RNN・Leakyの4モデルの評価結果

2026-11-18、job array **3256690**で4モデル×dt1/dt4×4収録の32条件を完了しました。ZIPのchecksumを再照合し、必要な4 runのみ展開してstrictな重み読み込みを使用しました。学習・データダウンロードは行っていません。[実行条件と重みhash](../results/2026-11-18-izar/four-model-evaluation.json)。

LeakyRecEVFlowNet・RNNFireNet・LeakyFireNetは全24条件でAEEと外れ値率が事前許容差内でした。RNNRecEVFlowNetはdt1/dt4とも論文比較が不一致で、最大差はAEE 1.245750 px・外れ値率14.681035 percentage pointsです。推論は全区間で有限値を返しましたが、比較判定によって該当2タスクのSlurm終了状態はFAILEDです。これを推論の途中停止や再現成功とは扱いません。原因は未確定です。

先行2モデルを含めた配布6モデルでは48条件中AEE 41条件、外れ値率38条件が許容差内です。[全7モデル（[56]再実装を含む）の比較表](../results/2026-11-18-izar/comparison.md)に全数値と未一致条件を掲載しています。
