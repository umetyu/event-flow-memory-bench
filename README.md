# event-flow-memory-bench

イベントカメラのoptical flowで、**入力の履歴**と**ネットワークの記憶**を分けて比較する実験基盤です。Hebbian augmentationを追加する前の基準を作ります。

主ベースラインはConvGRU版 **`RecEVFlowNet`**。対照に非再帰`EVFlowNet`のcount / 5-bin voxel、RNN / Leakyのcountを使います。まず同じ組のFireNet系で動作を確認します。論文で「EV-FlowNet」と呼ばれる再帰モデルは、コードでは`RecEVFlowNet`です。

比較対象はTable 1の**2・4・5・6行目**、すなわちEV-FlowNet* [56]（Zhu 2019）、RecEVFlowNet、RNNRecEVFlowNet、LeakyRecEVFlowNetです。論文の手法については既報値・配布重みを使い、**[56]は「論文報告値」と「UZHで再学習 → MVSEC」の2条件**を用意します。前者はMVSEC outdoor_day2で学習した報告値、後者は本基盤で実施した再実装の実測値として区別します。追加指示により、非再帰count / 5-bin voxelの2条件もseed 0・100 epochで学習します。[対象・実装状況](docs/BASELINES.md)と[2条件の実行仕様](docs/INPUT_BASELINES.md)。

現在生成・実行できるmain gridは従来の5条件です。[56]の[4収録 × dt1/dt4の報告値](configs/literature/zhu2019_table1.csv)と[UZH再学習用の実装・専用設定](docs/ZHU2019.md)はgridから分離しています。今回新たに実行するのは[非再帰count / 5-bin voxelの2条件](docs/INPUT_BASELINES.md)で、5条件×3 seedsの一括gridは投入しません。先行確認用FireNet系を維持し、14モデル一括・SNN実験は実行しません。Hebbian更新則自体は未実装です。

Izar上の著者配布UZH/MVSECを検査し、FireNet / RecEVFlowNetの学習速度を実測済みです。100 epoch・1 seedの推定は約7.0 / 13.5時間。[GPU実行手順と測定条件](docs/IZAR.md)を参照してください。`demo`は合成データによる実装検証です。

配布6モデルのMVSEC評価（4収録×dt1/dt4、計48条件）を完了しました。追加したLeaky-EV-FlowNet・RNN-FireNet・Leaky-FireNetの24条件は全指標が事前許容差内でした。RNN-EV-FlowNetはAEE最大差1.246 pxで未一致、先行2モデルの外れ値率にも2条件の差が残ります。[全7モデルの論文値との比較表](results/2026-11-18-izar/comparison.md)。Step 2の既存5条件の一括学習は未投入です。

2026-11-18に[56]のUZH再学習をIzar job **3256469**で完了しました。V100 1枚・seed 0・100 epochsの学習は約39.5分、続けてMVSEC dt1/dt4を評価しました。[完了記録](results/2026-11-18-izar/zhu2019-completion.json)と[収録別結果](results/2026-11-18-izar/zhu2019-summary.json)。検証RSATは約1のままで、精度改善や論文再現の成功は確認できていません。これは著者配布実装ではなく、[差分を明記した論文ベースの再実装](docs/ZHU2019.md)です。

同日に非再帰EVFlowNetのcount / voxel5を **3256757_0 / 3256757_1** で学習開始しました。両方seed 0・100 epochs、同じUZH分割・1k入力・10k損失窓で、終了後にMVSEC dt1/dt4を自動評価します。[開始時点の実行記録](results/2026-11-18-izar/input-baselines-launch.json)。事前に[実データ45更新ずつ・検証・GPUメモリの確認](results/2026-11-18-izar/input-baselines-smoke.json)を通過しました。これは学習完了や精度向上を示す記録ではありません。

## 実装済み

Gallego 2018のパッチCMax再実装と、Shiba et al. 2022の公式CMaxソルバーを追加しました。学習不要の2手法を、因果的な1k窓と長い窓でMVSEC評価します。Izar `3256789 / 3256790` で全32条件の評価を開始し、終了後は `3256791` で比較表を集計します。[実行条件・原論文との差分](docs/CMAX.md)と[開始時点の記録](results/2026-11-18-izar/cmax-launch.json)。全評価はまだ完了していません。

- NeurIPS 2021のモデル18種を固定コミットから収録。MITライセンス・出典を保持。
- 原実装のローダーと指標を使うMVSEC `dt=1/4`再現評価、論文表との比較。
- count / signed voxel、因果的な履歴窓、更新間隔・loss窓・評価窓の独立指定。
- 記録単位のtrain/val分割、系列境界のreset、TBPTT、検証RSATでのモデル選択、学習再開。
- UZH-FPV → MVSEC、ECD/HQF、AffineObjectsに共通のHDF5入力。
- AEE・外れ値率、FWL・RSAT、画面内に残るwarped eventの割合、LayerSegのIoU/DR。
- 3 seedsの段階別実験、履歴1k/5k/15k/50kの設定生成、集計CSV・曲線、Slurmスクリプト。
- 固定コミットのEV-LayerSegNetを読み込むアダプター。上流コードを改変・再配布せず接続。
- CPUテストとGitHub Actions。

## セットアップと小規模検証

Python 3.10以上。GPU用PyTorchは実行環境に合うものを先に導入します。

```bash
git clone https://github.com/umetyu/event-flow-memory-bench.git
cd event-flow-memory-bench
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[reproduction,plots]'
python -m unittest discover -s tests -v
efbench --threads 1 demo --output runs/demo
```

`runs/demo`に合成HDF5、学習ログ、`best.pt`、評価JSONができます。再実行には別の出力先を使います。既存runを黙って上書きしません。

## 1. 配布重みでMVSECを再現

[データ配置](docs/DATA.md)に従い配布パックを外部ストレージやGPU環境へ置きます。主データは圧縮19.4GB、展開約94GBです。

```bash
# MLflow runのparams/とartifacts/model/data/model.pthが必要
efbench reproduce --data /data/event_flow/MVSEC --run-dir /data/mlruns/0/RUN_ID \
  --dt 1 --device cuda --trust-legacy-pickle --output runs/reproduce-dt1.json
efbench reproduce --data /data/event_flow/MVSEC --run-dir /data/mlruns/0/RUN_ID \
  --dt 4 --device cuda --trust-legacy-pickle --output runs/reproduce-dt4.json
efbench compare-paper runs/reproduce-dt1.json --output runs/paper-check-dt1.json
efbench compare-paper runs/reproduce-dt4.json --output runs/paper-check-dt4.json
```

`--trust-legacy-pickle`は配布元を確認した古いPythonオブジェクト形式にだけ必要です。state_dictも`--checkpoint`と`--train-config`で読めます。重みがない・不一致の場合は停止し、ランダム重みで続行しません。参照表はRecEVFlowNet・FireNetと各RNN/Leaky版の6モデルを収録しています。

このモードは元のframe単位入力・指標を保持します。以下の統一比較とは別実験です。[相違点](docs/PROTOCOL.md)を参照してください。

## 2. 学習・検証データを固定

配布学習パックの140クロップについて、**同じ元記録から切り出したファイルを同じgroupにする**対応表を用意します。クロップ名から独立性を推測して分割しません。

```bash
efbench inventory /data/event_flow/training --dataset uzh_fpv --split train \
  --groups data/uzh-original-recordings.json --output data/uzh-inventory.json
efbench split data/uzh-inventory.json --seed 0 --val-fraction 0.1 --output data/uzh.json
efbench inventory /data/event_flow/MVSEC --dataset mvsec --split test --output data/mvsec.json
efbench inspect /data/event_flow/training/example.h5 --resolution 128 128
```

対応表の書式は[DATA.md](docs/DATA.md)を参照。MVSECは学習・モデル選択に使いません。

## 3. FireNet系 → EV-FlowNet系

```bash
efbench grid --stage pilot --manifest data/uzh.json --output configs/generated/pilot
python scripts/run_grid.py configs/generated/pilot/jobs.json --dry-run
python scripts/run_grid.py configs/generated/pilot/jobs.json

efbench grid --stage main --manifest data/uzh.json --output configs/generated/main
python scripts/run_grid.py configs/generated/main/jobs.json
```

上のコマンドは5モデル条件 × 3 seeds = 15 runsを投入するため、今回の2条件の実行には使いません。[count / voxel5専用の手順](docs/INPUT_BASELINES.md)で各seed 0だけを実行します。[56]もこのgridに含めず、[専用の実行手順](docs/ZHU2019.md)を使います。生成YAMLを確認し、batch sizeなどの変更は条件間で揃えて記録します。RNNRecEVFlowNetの元設定にはlr=5e-5の注記がありますが、統一比較は全モデル2e-4から始めます。探索はUZH検証側でのみ行います。

```bash
# 単一モデルを実行・再開
efbench train --config configs/uzh_firenet.yml --output runs/uzh-firenet
efbench train --config configs/uzh_firenet.yml --output runs/uzh-firenet \
  --resume runs/uzh-firenet/latest.pt

# 学習設定から評価設定を生成
efbench eval-config --train-config configs/uzh_firenet.yml --manifest data/mvsec.json \
  --dataset mvsec --mode gtflow_dt1 --resolution 256 256 --output runs/mvsec-dt1.yml
efbench evaluate --config runs/mvsec-dt1.yml --checkpoint runs/uzh-firenet/best.pt \
  --output runs/mvsec-dt1.json
```

同様に`--mode gtflow_dt4`も評価します。ECD/HQFは`--mode events`と各データの解像度を使います。

## 4. 履歴窓の比較

```bash
efbench grid --stage main --manifest data/uzh.json --histories 1000 5000 15000 50000 \
  --output configs/generated/history
python scripts/run_grid.py configs/generated/history/jobs.json
efbench collect runs/evaluations/*.json --output runs/results.csv
python scripts/plot_history.py runs/results.csv --metric aee --output runs/figures
```

60 runsです。**stride=1k、loss=10k、proxy評価窓=50k、共通warmup=50kを固定**し、入力履歴だけを変えます。窓ごとに再学習します。集計では評価anchorと採点条件も識別し、異なる条件を混ぜません。標準偏差はseed間の標本標準偏差です。

## 5. AffineObjectsとEV-LayerSegNet

独立した副実験として、AffineObjectsのtrain内でvalを作り、testを固定します。フローGTの存在は前提にしません。

```bash
python scripts/bootstrap_upstream.py event_segmentation
efbench grid --stage main --dataset affineobjects --manifest data/affineobjects.json \
  --output configs/generated/affine-main
efbench grid --stage layerseg --dataset affineobjects --manifest data/affineobjects.json \
  --output configs/generated/affine-layerseg
python scripts/run_grid.py configs/generated/affine-layerseg/jobs.json
```

両モデルに同じstride/history/loss/eval条件を設定します。原論文の200k学習窓の再現は別ラベルで扱います。IoU/DRはマスク時刻とPNGの対応を明示し、`--mode masks`で計算します。前景channelは検証データで固定し、テストGTを見て入れ替えません。SNNは`efbench grid --stage snn ...`で追加できます。

## 設計・状況

- [選択したベースラインとZhu 2019 [56]の位置づけ](docs/BASELINES.md)
- [条件・単位・既知の限界](docs/PROTOCOL.md)
- [データ書式・入手先](docs/DATA.md)
- [実施順序・完了条件](docs/ROADMAP.md)
- [上流コード・変更点・引用](THIRD_PARTY.md)
- [動作検証](docs/VALIDATION.md)
- [Izarでのデータ検査・速度計測・再現評価](docs/IZAR.md)

本体はMIT。vendor内のTU Delftライセンスを保持してください。データとEV-LayerSegNetは各配布元の条件に従います。
