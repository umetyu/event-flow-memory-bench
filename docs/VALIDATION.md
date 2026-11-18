# 動作検証

2026-11-18、macOS / Python 3.10.22 / PyTorch 2.14.1、CPUで検証しました。

| 検証 | 結果 |
|---|---|
| unittestの標準スイート | 25テスト中24通過、任意のLayerSegテスト1件は別実行 |
| 固定上流checkoutのLayerSegアダプター | forward/backward、flow/mask shapeの追加テスト通過 |
| 18種類のevent_flowモデル | 出力shape、有限勾配、reset後の出力一致を確認 |
| 非再帰EVFlowNet / FireFlowNetの5-bin入力 | 出力shapeを確認 |
| 因果的履歴・現在窓の採点mask | 将来イベント変更に対する不変性を確認 |
| 大きな絶対timestamp、重複timestamp、境界外座標 | 精度維持・left search・入力検査を確認 |
| train/val group | 漏洩検出、seed付き分割の再現性を確認 |
| AEE / 外れ値 | valid mask、GTとpredictionの閾値規約、NaN処理を確認 |
| dt1 / dt4 | 既知速度の合成GTで変位換算を確認 |
| FWL / RSAT / loss | zero-flow比=1、有限勾配を確認 |
| 学習・保存・再読込・評価 | 合成HDF5で一連の処理を確認 |
| epoch境界からの再開 | 中断なし2 epochsと重みが完全一致 |
| 原実装再現経路 | 合成HDF5とstate_dictでdt1/dt4を確認。実配布2モデルでも評価完了、外れ値率2条件は未一致（下記） |
| 実験表・集計 | 60設定生成、同seed重複拒否を確認 |

```bash
python -m unittest discover -s tests -v
EFB_LAYERSEG_SOURCE=/path/to/event_segmentation \
  python -m unittest discover -s tests -p test_benchmark.py -k layerseg -v
efbench --threads 1 demo --output runs/demo
```

LayerSeg上流はaffine_gridのalign_cornersを省略しており、現行PyTorchで既定Falseの警告を出します。元コードを変更せず保持しています。アダプター検査は小さい合成入力でのforward/backwardであり、480×640のGPU学習・論文精度を保証するものではありません。

上流には大きな実験ファイルも追跡されているため、アダプターは全working treeを走査せず、読み込むPythonファイルのGit blobハッシュを固定コミットと照合します。

2026-11-18にIzarの著者配布UZH 140本とMVSEC 4収録のメタデータ・端点を検査し、元記録単位の116/24 splitを生成しました。FireNet / RecEVFlowNetは実UZHとV100上で各120更新（20 warmup＋100計測）、有限loss・勾配のチェック付きで完走しました。詳細は[IZAR.md](IZAR.md)を参照してください。

配布重み2モデル×dt1/dt4×4収録の評価を完了。AEEは16/16条件で論文との差0.00859 px以下、外れ値率は14/16条件で事前許容差内です。FireNet outdoor_day1 dt4とRecEVFlowNet indoor_flying3 dt4の外れ値率に差が残り、完全再現の判定はしていません。[全条件の結果](../results/2026-11-18-izar/paper-comparison.csv)。

**未実施:** 全条件の本学習、AffineObjectsのGT対応確認と実評価。短い速度計測は学習完了や精度再現を意味しません。

## Zhu 2019再実装の追加検証

Izar job 3256468で全36テストを実行し、35成功・任意のLayerSeg接続1件skipでした。新規11テストは9-bin補間、極性・境界splat、双方向timestamp損失、有限差分との勾配一致、smoothness、単位換算、モデル全段のbackward、設定制約、再開の同一性、smokeの扱い、MVSEC評価の因果性を確認します。

同じjobで実UZHを20更新し、1本の検証データの16窓を評価しました。loss・勾配・RSATが有限で完了し、forward warp後の画面内イベント割合は約0.9928でした。[動作確認の数値](../results/2026-11-18-izar/zhu2019-smoke.json)。この短い確認のRSATは1.0であり、精度向上・収束はまだ示していません。[実装の選択と差分](ZHU2019.md)を参照してください。
