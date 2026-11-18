# 非再帰EV-FlowNet: countと5-bin voxelの比較

2026-11-18の追加指示により、Step 2の上2条件を実行します。両方ともコード上の`EVFlowNet`で、再帰状態はありません。Table 1のConvGRU版`RecEVFlowNet`、Zhu 2019 [56]の再実装、別プロジェクトのvoxel9実験とは別条件です。

| 条件 | カウント版 | voxel版 |
|---|---|---|
| 設定 | [uzh_evflownet_count.yml](../configs/uzh_evflownet_count.yml) | [uzh_evflownet_voxel5.yml](../configs/uzh_evflownet_voxel5.yml) |
| 入力 | 正負2チャネルのカウント | signed voxel、5時間ビン |
| seed / epochs / batch | 0 / 100 / 8 | 同左 |
| データ | UZH 116クロップ学習、別収録の24クロップ検証 | 同左 |
| 入力・更新間隔 | 1,000イベント | 同左 |
| 損失窓 | 10,000イベント | 同左 |
| 検証窓 / warmup除外 | 50,000 / 50,000イベント | 同左 |
| optimizer | Adam、lr=2e-4、clip=100 | 同左 |
| 損失 | Hagenaars EventWarping、batch平均、smoothness=0.001 | 同左 |
| モデル選択 | UZH検証RSATの最小checkpoint | 同左 |
| test | MVSEC 4収録 × dt1/dt4 | 同左 |

履歴窓の長さは同じで、5ビン化によって長くはなりません。この2条件は窓内の時間情報の符号化を比べます。入力チャネル数に応じて最初の畳み込みのパラメータ数も変わるため、完全な同容量比較ではありません。設定、seed、サンプル順、flipの生成規約は共通です。

採点は[matched-v1](PROTOCOL.md)です。配布重みのTable 1再現用とは入力partition・maskなどが異なるため、数値を同一条件のランキングに混ぜません。窓長sweepと3 seedsへの拡張は別段階です。今回追加する本学習はこの2条件のseed 0だけです。

## Izarでの実行

既存データを参照し、ダウンロードやデータ複製はしません。

```bash
export EFBENCH_PYTHON=/path/to/cuda-python
cd /path/to/event-flow-memory-bench

# 各モデル: 実データ8 training crops / 1 held-out validation cropを1 epoch。
# index 0では既存の全unittestも実行。出力はinput-smoke-*に分離。
EFBENCH_INPUT_SMOKE=1 sbatch --time=00:30:00 scripts/slurm_izar_input_baselines.sh

# 2 GPUで各100 epoch。本学習は毎回新しいrunから開始。
EFBENCH_INPUT_SMOKE=0 EFBENCH_INPUT_RESUME=0 sbatch scripts/slurm_izar_input_baselines.sh
```

array index 0はcount、1はvoxel5です。`runs/input-baselines-JOBID/count`と`voxel5`へ保存します。実行前に、2設定がencodingとnum_bins以外で一致することを検査します。

`epochs.jsonl`にloss・検証RSAT・更新数・時間を、`latest.pt`と`best.pt`にepoch境界の状態を保存します。学習終了後、best.ptからMVSEC dt1/dt4を評価し、両方が終わってから`completed.json`を書きます。smokeのcheckpointは本学習に引き継ぎません。`smoke.json`を本学習の完了と扱わないでください。

時間切れなどで再開する場合は、以前の配列runの親ディレクトリを指定し、同じ条件の`latest.pt`から再開します。データ・設定の差があれば停止します。

```bash
EFBENCH_INPUT_SMOKE=0 EFBENCH_INPUT_RESUME=1 \
EFBENCH_INPUT_RUN_ROOT=runs/input-baselines-PREVIOUS_JOBID \
sbatch --array=0-1%2 scripts/slurm_izar_input_baselines.sh
```

既に完了した条件がある場合は、`--array`で未完了のindexだけを指定します。
