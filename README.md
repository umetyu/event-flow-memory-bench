# event-flow-memory-bench

イベントカメラのoptical flowで、**入力の履歴**と**ネットワークの記憶**を分けて比較する実験基盤です。Hebbian augmentationを追加する前の基準を作ります。

主ベースラインはConvGRU版 **`RecEVFlowNet`**。対照に非再帰`EVFlowNet`のcount / 5-bin voxel、RNN / Leakyのcountを使います。まず同じ組のFireNet系で動作を確認します。論文で「EV-FlowNet」と呼ばれる再帰モデルは、コードでは`RecEVFlowNet`です。

**実データの再現値や学習済み重みはまだありません。** `demo`は合成データによる実装検証です。データとGPUを配置してから下記の順に実験します。Hebbian更新則自体は未実装です。

## 実装済み

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

`--trust-legacy-pickle`は配布元を確認した古いPythonオブジェクト形式にだけ必要です。state_dictも`--checkpoint`と`--train-config`で読めます。重みがない・不一致の場合は停止し、ランダム重みで続行しません。参照表はRecEVFlowNetとFireNetを収録しています。

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

各stageは5モデル条件 × 3 seeds = 15 runs。生成YAMLを確認し、batch sizeなどの変更は条件間で揃えて記録します。RNNRecEVFlowNetの元設定にはlr=5e-5の注記がありますが、統一比較は全モデル2e-4から始めます。探索はUZH検証側でのみ行います。

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

- [条件・単位・既知の限界](docs/PROTOCOL.md)
- [データ書式・入手先](docs/DATA.md)
- [実施順序・完了条件](docs/ROADMAP.md)
- [上流コード・変更点・引用](THIRD_PARTY.md)
- [動作検証](docs/VALIDATION.md)

本体はMIT。vendor内のTU Delftライセンスを保持してください。データとEV-LayerSegNetは各配布元の条件に従います。
