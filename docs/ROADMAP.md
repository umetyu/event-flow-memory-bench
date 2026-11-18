# 実施順序

実装完了と実データでの検証完了は別です。

| 段階 | 実装 | 実データでの完了条件 |
|---|---|---|
| 0: 環境・データ | HDF5検査、split、出典固定 | データ取得、元記録group、GPUを確認 |
| 1: 論文再現 | 重み読込、dt1/dt4、表との差分 | RecEVFlowNet / FireNetの4系列の再現 |
| 2: pilot | 5条件 × 3 seeds | FireNet系の収束・評価 |
| 3: 主対照 | 5条件 × 3 seeds | UZH → MVSECの全条件表 |
| 4: 履歴 | 4履歴 × 5条件 × 3 seeds | 固定anchorで曲線作成 |
| 5: AffineObjects | 共通proxy、LayerSeg、mask評価 | 同一窓のFWL/RSAT、IoU/DR |
| 6: SNN | 8モデルの設定・読込 | 必要なモデルの再現・比較 |
| 7: Hebbian | 接続位置・対照を設計 | 更新則を確定し実装・比較 |

jobs.jsonは実行順序を保持し、run_grid.pyは失敗時に停止します。latest.ptはepoch境界の再開用、best.ptは検証RSATで選択した評価用です。

未実施: 実データ取得、配布重みでの論文数値再現、GPUでの学習、統計的結論、Hebbian更新則。合成テストの値を研究成果として報告しません。
