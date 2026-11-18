# 比較対象

2026-11-18の選択。行番号はNeurIPS 2021 Table 1の最上段`EV-FlowNet* [54]`を1行目として数えます。

| 用途 | 行 | 論文の表記 | この基盤での識別 | 状態 |
|---|---:|---|---|---|
| 論文報告値による参考比較 | 2 | EV-FlowNet* [56] | zhu2019 / published_reference_not_reproduced | 8条件の報告値を収録済み。当プロジェクトで推論した値ではない |
| 追加の再学習比較 | 2と同じ手法 | EV-FlowNet* [56] | zhu2019-uzh（再実装 → UZH学習 → MVSEC評価） | 100 epoch・MVSEC評価完了。検証RSAT約1で精度改善未確認。[実行仕様](ZHU2019.md) |
| 主比較 | 4 | EV-FlowNet | RecEVFlowNet / count | 実装済み。配布重み評価済み |
| 主比較 | 5 | RNN-EV-FlowNet | RNNRecEVFlowNet / count | 配布重み評価完了。論文AEE最大差1.246 pxで未一致 |
| 主比較 | 6 | Leaky-EV-FlowNet | LeakyRecEVFlowNet / count | 配布重み評価完了。8条件とも事前許容差内 |
| 軽量な先行確認 | 11 | FireNet | FireNet / count | 実装済み。配布重み評価済み |
| 軽量な先行確認 | 12 | RNN-FireNet | RNNFireNet / count | 配布重み評価完了。8条件とも事前許容差内 |
| 軽量な先行確認 | 13 | Leaky-FireNet | LeakyFireNet / count | 配布重み評価完了。8条件とも事前許容差内 |

ここまでの論文モデル比較では[56]を再学習し、他6モデルは配布済み重みで評価しました。[全7モデルの比較と未一致条件](../results/2026-11-18-izar/comparison.md)。事前許容差はAEE 0.02 px・外れ値率0.2 percentage pointです。追加指示により、次に非再帰EVFlowNetのcount / 5-bin voxelを各seed 0・100 epochで学習します。5条件 × 3 seedsの一括gridは投入しません。[専用設定と実行手順](INPUT_BASELINES.md)。

別途、後続の要因比較用に同じHagenaars実装の非再帰EVFlowNetのcount / 5-bin voxelを保持します。軽量版の対応モデルはFireFlowNetです。[56]をこの既存voxelモデルの別名として扱いません。

## [56]の2条件

[Zhu et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Unsupervised_Event-Based_Learning_of_Optical_Flow_Depth_and_Egomotion_CVPR_2019_paper.html)が[56]です。[原論文 §4.1–4.2](https://arxiv.org/html/1812.08156#S4.SS1)の学習データはMVSEC outdoor_day2であり、UZHではありません。一方、Hagenaarsの各モデルはUZHからMVSECへの評価です。そのままの論文値の差には、学習データの違いも含まれます。

ユーザーの指定により、[56]は次の両方を比較します。

1. **論文報告値**: MVSEC 4収録・dt1/dt4の既報8条件。学習元はMVSEC outdoor_day2。
2. **UZH再学習版**: 既存のrecording-level split（学習116クロップ、検証24クロップ）で学習し、同じMVSEC 4収録・dt1/dt4で評価。モデル選択はUZH検証側だけで行う。

2のためにoutdoor_day2をダウンロードする必要はありません。再学習版の実装差分・入力窓・crop解像度・学習量・採点規約を記録し、学習データだけを変えた完全な対照実験とは断定しません。

比較表では1に「論文報告値」「学習: MVSEC outdoor_day2」を明示します。こちらでtestを実行した結果や配布重みを再評価した結果とは扱いません。報告値の要約だけから同じmask・外れ値規約へ再採点はできないため、matched-v1と同条件の優劣・seed平均・分散の計算にも含めません。2の結果は再実装の実測値として別ラベルで保存しています。

## 原論文から確認した条件

以下は原論文の条件であり、現在のmatched-v1設定の説明ではありません。

| 項目 | Zhu 2019 [56] |
|---|---|
| 入力 | 時間を9ビンに離散化したevent volume、時間方向に線形補間 |
| 原学習データ | MVSEC outdoor_day2 |
| 原学習入力窓 | 30,000イベント、256×256中央crop |
| 原評価入力窓 | outdoor 30,000イベント、indoor 15,000イベント |
| ネットワーク出力 | pixel/bin。評価時に(B−1)×dt/(t_last−t_first)を掛けてpixel displacementへ変換 |
| 損失 | motion compensation後の平均timestamp画像に基づく双方向損失と空間smoothness |
| 原論文の外れ値率 | EPEが3 pxを超える有効イベント画素の割合（本文§4.2） |

Hagenaarsのactive-pixel数で正規化した損失や、既存の5-bin voxel対照と同一視しません。原法の外れ値率とmatched-v1の3px AND 5%-of-GTも区別し、参考値には原論文の採点規約を保持します。

## 保存と実行範囲

今回の確認では[56]に対応する正式実装・配布重みを同定できていません。著者グループの[EV-FlowNetリポジトリ](https://github.com/daniilidis-group/EV-FlowNet)は、READMEがRSS 2018のグレースケール画像による教師信号を説明しています。これを自動的にCVPR 2019 [56]の実装として取り込むことはしません。

著者は[Issue #12の回答](https://github.com/daniilidis-group/EV-FlowNet/issues/12#issuecomment-644892449)で、当時の勤務先の制約から公開対応が難しく、独自再実装なら支援できると説明しています。現在確認できたmaster branchからCVPR 2019の正式実装・配布重みは同定できていません。

再学習版は、論文に基づく再実装として差分を明示して追加しました。9-bin入力、双方向timestamp損失、smoothness、出力のpixel/bin単位を検証し、既存のHagenaars損失と5-binモデルをそのまま[56]と呼ぶことはしません。[専用実装・学習設定](ZHU2019.md)を追加し、35テスト成功・任意接続1件skip、Izarで20更新と検証1本の確認後、100 epochとMVSEC評価を完了しました。検証RSATは約1で、論文精度の再現成功とは判定していません。

[機械可読の仕様](../configs/literature/zhu2019.json)と[原論文の参考値](../configs/literature/zhu2019_table1.csv)を収録しています。後者は当プロジェクトの実測値ではなく、collect対象のresultsとは分離しています。`main` gridは5条件のままです。[56]は専用コマンドで実行し、論文報告値と再実装の実測値を別に記録します。
