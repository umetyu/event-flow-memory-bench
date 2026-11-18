# 比較対象

2026-11-18の選択。行番号はNeurIPS 2021 Table 1の最上段`EV-FlowNet* [54]`を1行目として数えます。

| 用途 | 行 | 論文の表記 | この基盤での識別 | 状態 |
|---|---:|---|---|---|
| 主比較 | 2 | EV-FlowNet* [56] | zhu2019-uzh（UZHで再学習 → MVSEC） | 採用済み。実装・重みの同定は未完了、未学習 |
| 主比較 | 4 | EV-FlowNet | RecEVFlowNet / count | 実装済み。配布重み評価済み |
| 主比較 | 5 | RNN-EV-FlowNet | RNNRecEVFlowNet / count | 実装済み、未学習 |
| 主比較 | 6 | Leaky-EV-FlowNet | LeakyRecEVFlowNet / count | 実装済み、未学習 |
| 軽量な先行確認 | 11 | FireNet | FireNet / count | 実装済み。配布重み評価済み |
| 軽量な先行確認 | 12 | RNN-FireNet | RNNFireNet / count | 実装済み、未学習 |
| 軽量な先行確認 | 13 | Leaky-FireNet | LeakyFireNet / count | 実装済み、未学習 |

別途、同じHagenaars実装の非再帰EVFlowNetのcount / 5-bin voxelを、入力表現の要因切り分け用に使います。軽量版の対応モデルはFireFlowNetです。[56]をこの既存voxelモデルの別名として扱いません。

## [56]もUZHで学習する理由

[Zhu et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Unsupervised_Event-Based_Learning_of_Optical_Flow_Depth_and_Egomotion_CVPR_2019_paper.html)が[56]です。[原論文 §4.1–4.2](https://arxiv.org/html/1812.08156#S4.SS1)の学習データはMVSEC outdoor_day2であり、UZHではありません。一方、Hagenaarsの各モデルはUZHからMVSECへの評価です。そのままの論文値の差には、学習データの違いも含まれます。

今回の主比較では、[56]も既存のUZH recording-level split（学習116クロップ、検証24クロップ）で再学習し、同じMVSEC 4収録・dt1/dt4で評価します。既報の重みをMVSEC学習済みのまま主比較へ混ぜません。原論文値は学習元と指標を明記した参考値として別に残します。この追加のためにoutdoor_day2をダウンロードする予定はありません。

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

Hagenaarsのactive-pixel数で正規化した損失や、既存の5-bin voxel対照と同一視しません。原法の外れ値率とmatched-v1の3px AND 5%-of-GTも区別します。主比較用には同じGT区間・event mask・外れ値規約で再採点し、原法の指標を併記する場合は別列にします。

## 実装の扱い

今回の確認では[56]に対応する正式実装・配布重みを同定できていません。著者グループの[EV-FlowNetリポジトリ](https://github.com/daniilidis-group/EV-FlowNet)は、READMEがRSS 2018のグレースケール画像による教師信号を説明しています。これを自動的にCVPR 2019 [56]の実装として取り込むことはしません。

実装を同定するか、論文に基づく再実装として差分を明示した後、9-bin入力・元の損失・出力単位を検証し、UZH用の学習設定を追加します。学習データ、crop解像度、学習量、入力窓など、原設定から変更する項目は結果に記録します。これらが未確定のまま既存EVFlowNetを[56]として実行しません。

[機械可読の仕様](../configs/literature/zhu2019.json)と[原論文の参考値](../configs/literature/zhu2019_table1.csv)を収録しています。後者は当プロジェクトの実測値ではなく、collect対象のresultsとは分離しています。`main` gridは現状5条件のままで、[56]を実装済みと数えません。
