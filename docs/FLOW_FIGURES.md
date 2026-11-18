# MVSECの推定フローとGTの比較図

`scripts/plot_flow_comparison.py`は、実際のチェックポイントをGPUで推論し、MVSECのGTと同じ区間・同じ色スケールで並べます。画像生成モデルによるイラストではありません。

## モデルと今回の実行

- ConvGRU EV-FlowNet：著者配布の`RecEVFlowNet`。入力は正負2チャネルのイベントカウント。
- EV-FlowNet voxel5：本リポジトリでUZHを使って学習中の非再帰`EVFlowNet`。入力は時間5ビンのsigned voxel。Zhu 2019 [56]の9-bin条件とは別です。
- 2026-11-18の図ではvoxel5の**14/100 epoch時点の検証ベスト**を固定して使用。100 epoch完了後の結果ではありません。途中で更新される`best.pt`をSHA-256確認付きでコピーし、8例すべてに同じコピーを使いました。
- 4収録（outdoor_day1、indoor_flying1–3）× dt=1/4の8例。結果の小さな実行記録は`results/2026-11-18-izar/flow-figures.json`に保存します。

## 図を生成する手順

1. 配布済みHDF5の`flow_dt1`または`flow_dt4`から、各収録の**50万イベント後に始まる最初のGT区間**を選びます。モデルの精度による選別はしません。GTはデータセットに保存された変位をそのまま読み込みます。
2. 選んだGT区間の終端直前までのイベントを読み込みます。末尾の入力がちょうど1,000イベントになるよう、そこから逆算した境界で系列全体を分割します。系列冒頭のウォームアップ窓だけ1,000未満になり得ます。重複・欠落・未来イベントはありません。
3. 同じイベント列からcountと5-bin voxelを作ります。ConvGRUの状態は系列冒頭から引き継ぎ、voxel5モデルは各窓を独立に推論します。収録間とdt条件間で状態をリセットします。
4. 最後の入力窓の推定を、GT区間のピクセル変位に換算します。

   `GT区間の推定変位 = ネットワーク出力 × 128 × GTの時間幅 / 最後の1,000イベントの時間幅`

   128はこの学習設定の`flow_scale`です。最終窓内の平均速度がGT区間を通じて一定だと仮定します。GT区間の時間幅はHDF5のタイムスタンプから取得し、単純にdt=4を4倍する処理はしません。
5. GT有効画素と、GT区間内にイベントが1個以上ある画素の共通部分を3枚のフローパネルに使います。配布ファイルでは原則(0,0)のGTは無効ですが、明示的なvalidマスクがあるファイルではそちらを優先します。イベント画像はGT区間内の全イベントです。
6. 方向を色相、変位の大きさを彩度に変換します。右が赤、左がシアン。白はゼロ変位、黒は表示対象外です。全モデル・全収録で共通の上限を使用し、dt=1は5 px、dt=4は20 pxで彩度を飽和させます。画像ごとの自動正規化はしません。上限超過の画素率も記録します。

AEEは表示したその1区間の平均誤差です。MVSEC全体の平均や論文Table 1の数値ではありません。ここでのGT終端に揃えた1k入力は、Table 1再現用のフレーム単位入力とも異なります。また、著者ConvGRUと今回のvoxel5では学習条件が異なり、voxel5は学習途中なので、この図だけで構造の優劣は結論づけられません。

### GT境界での時間幅

可視化の検査時に、汎用`matched-v1`評価経路がGT区間末尾で短いstrideを作りながら、モデルには直近1kイベントを渡す場合があると確認しました。その短いstrideの時間幅で1k入力の出力を割ると、変位を過大換算するおそれがあります。この図では上の専用分割によって**実際に入力した1kイベントと時間換算の窓を一致**させています。汎用評価経路は今回の可視化コードから呼びません。この図生成の変更は、実行中の学習や既存のTable 1再現評価コードを変更するものではありません。汎用経路を使う最終定量評価では、この境界処理を別途統一する必要があります。

## 実行コマンド

`.[reproduction,plots]`を導入し、既存のMVSEC manifest、検証済みの著者MLflow run、voxel5 checkpointを指定します。出力先は空のディレクトリを指定してください。

```bash
PYTHONPATH=src python scripts/plot_flow_comparison.py \
  --manifest runs/setup/mvsec.json \
  --author-run 'external/author-models/mlruns 2/42/EVFlowNet' \
  --voxel-checkpoint runs/input-baselines-3256757/voxel5/best.pt \
  --output runs/flow-figures-new \
  --anchor-events 500000 --device cuda --trust-author-pickle
```

`--trust-author-pickle`は出所を確認した著者の古いpickle形式の重みを読み込むための指定です。重みがない・モデル構造が違う場合はエラーにし、ランダム重みで続行しません。上記の`best.pt`は学習とともに更新されるため、同一結果の再現には実行記録のSHA-256に一致する保存コピーを使ってください。

Izarでは`scripts/slurm_izar_flow_figures.sh`を使用できます。GPU 1枚、4 CPU、24 GBを要求します。入力環境変数はスクリプト内に明記しています。

再推論せず、保存済みのフロー配列から再描画する場合：

```bash
PYTHONPATH=src python scripts/plot_flow_comparison.py \
  --output runs/flow-figures-new --render-only
```

## 出力と検証

- `flow-comparison-dt1.png/.svg`、`flow-comparison-dt4.png/.svg`：4収録の比較一覧。
- `flow-detail-indoor_flying2-dt1.png/.svg`、dt4：同じデータから作った1収録の拡大図。
- 各収録・dtの`.npz`：GT、両モデルの推定フロー、マスク、イベント、最後の入力イベント。
- `metadata.json`：重み・配列・スクリプトのSHA-256、入力設定、タイムスタンプ、各区間のAEE、色スケール、飽和率。
- `checkpoints/voxel5.pt`：その実行で固定した重み。GPU環境に保存し、データや重みをGitリポジトリへ追加しません。

テストでは入力の欠落・重複がないこと、最後の入力が完全な1kであること、GT変位への時間換算、未来イベントを書き換えても出力が変わらないこと、共通マスク、同じベクトルがどのパネルでも同じ色になることを確認します。実データの保存後にも配列からAEEを再計算します。

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_flow_visualization.py -v
```
