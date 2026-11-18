# 指定4手法とGTを並べた進捗付き比較図

ユーザー指定により、左からGT、ConvGRU EV-FlowNet（count）、EV-FlowNet [56]型（9-bin voxel）、非再帰EV-FlowNet（count）、ShibaらのCMaxを並べます。最後の2手法については、追加指示に従い**実際のフローを掲載せず、進捗だけを表示**します。

| パネル | 内容 |
|---|---|
| GT | MVSEC HDF5に保存された正解のピクセル変位 |
| ConvGRU EV-FlowNet | 著者配布`RecEVFlowNet`、正負2チャネルcount入力。既に検証した推定配列をSHA-256照合後に再利用 |
| EV-FlowNet [56]型 | UZHで100 epoch学習した本リポジトリのZhu 2019再実装の検証ベスト。signed 9-bin voxel |
| EV-FlowNet + count | 「学習中」。推定フロー配列は保存・描画しない |
| CMax (Shiba et al.) | 「最適化・評価中」。学習を行わない手法なので「トレーニング中」とは記載しない。推定フロー配列は保存・描画しない |

これは作成時点の進捗を示す静的な図です。状態は自動更新されません。既存の学習・CMax全系列評価ジョブは継続します。

## 同じ区間・同じ色で比較

- MVSECの4収録×dt=1/4。区間は[前回の比較図](FLOW_FIGURES.md)と同じ、各収録の50万イベント後に始まる最初のGT区間です。精度を見て選び直していません。
- ConvGRUはGT終端に揃えた完全な1,000イベント窓を入力し、系列冒頭からの再帰状態を保持した出力です。
- [56]型は既存の評価仕様に合わせ、屋外30,000イベント・屋内15,000イベントをGT終端直前まで読み込みます。未来イベントは含めません。
- [56]型の出力はpixels/binなので、`出力 × (9−1) × GT時間幅 / 入力時間幅`でピクセル変位へ変換します。ConvGRUは学習設定の`flow_scale=128`による既存の換算済み配列を再利用します。
- 色は方向、彩度は変位の大きさです。全手法・全収録で同じスケール（dt=1は0–5 px、dt=4は0–20 px）を使います。上限以上の移動量は色が飽和します。白はゼロ変位、黒は表示対象外です。
- **主図はGTが有効な全画素**にフローを表示し、イベントが発生していない画素も含めます。イベントのある画素だけに絞る従来の表示は`-eventmasked`の補助図として別に保存します。GTや推定値を平滑化・補間して見た目を変える処理は追加しません。

入力窓、学習データの分割、損失などは両モデルで異なります。この図は同一条件の構造比較ではありません。[56]型の検証RSATは約1に留まり、論文精度の再現成功は確認できていません。ゼロに近い推定が白く見える場合も、実際の出力をそのまま表示します。

## 再生成

```bash
PYTHONPATH=src python scripts/plot_requested_flow_comparison.py \
  --reference runs/flow-figures-3256861/final-artifacts \
  --checkpoint runs/zhu2019-uzh-seed0-3256469/best.pt \
  --manifest runs/setup/mvsec.json \
  --output runs/requested-flow-figures-new --device cuda

# 保存された数値配列から描画。--fontには日本語対応のフォントを指定する。
PYTHONPATH=src python scripts/plot_requested_flow_comparison.py \
  --output runs/requested-flow-figures-new --render-only \
  --font /path/to/Japanese-font.ttc
```

フォントを指定しない場合は英語表記で描画します。Izar用の推論スクリプトは`scripts/slurm_izar_requested_flow_figures.sh`です。出力先が既存の場合、推論では上書きせず停止します。`--render-only`は同じ配列から画像だけを再生成します。

## 保存物と確認

- `comparison-dt1.png/.svg`、`comparison-dt4.png/.svg`：GT＋指定4手法の4収録一覧。
- `detail-indoor_flying2-dt1.png/.svg`、dt4：同じ配列から作成した1収録の拡大図。
- `-eventmasked`付きのファイル：GT有効かつイベントのある画素だけを描く補助図。
- 8つの`.npz`：GT、ConvGRU、[56]型9-binのフロー、2種類のマスク、イベント、9-bin入力窓、換算前の出力。
- `metadata.json`：学習済み重みのSHA-256、選択epoch、学習完了epoch、元のConvGRU配列の出典、実際の入力時間幅、区間別誤差、各ファイルのハッシュ。

保存時にはGTと元HDF5の一致、入力の因果性、全モデルの共通表示範囲、配列から独立に再計算したAEEを検証します。GT有効画素全体の誤差と、イベント画素に限定した誤差を区別してメタデータに保存します。画像には期間全体の評価値のように見える数値を追加せず、区間・入力条件を表示します。画像の凡例と日本語ラベルも目視確認します。
