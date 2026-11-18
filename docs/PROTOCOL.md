# 比較プロトコル

## 二つの数値系列を区別する

| 項目 | hagenaars2021-upstream | matched-v1 |
|---|---|---|
| 目的 | 配布重みで原実装を再現 | 入力履歴と内部記憶の比較 |
| MVSEC forward単位 | dt1: GT interval全体、dt4: イベントを4分割 | interval内を1kイベントで進め、末尾の端数も処理 |
| 入力履歴 | 当該partition | 現在を含む直近history_events |
| 変位合わせ | 最終出力 × 128 × dt_gt/dt_input | 最終出力 × flow_scale × dt_gt/dt_input |
| 採点event mask | 最終partition（元コード） | GT intervalのイベントのunion |
| 5%外れ値閾値 | 予測フローの大きさ（元コード） | GTの大きさ、設定に明示 |
| warmup | なし | 全条件で50kイベントを除外 |
| hot pixel filter | 元の有効設定 | 無効で統一 |

両者は採点集合が異なるため表を混ぜません。再現モードでも系列境界で指標バッファをresetします。これは元スクリプトとの差分です。

## 比較条件

主実験はUZH-FPV学習・検証、MVSECテスト。元記録groupで検証を分け、近接・重複クロップの漏洩を防ぎます。元記録対応が不明なら厳密なgroup splitは未確定です。ファイル単位のランダム分割を独立検証と主張しません。

[56]（Zhu 2019）はMVSEC outdoor_day2で学習した論文報告値と、UZHで再学習する条件の両方を比較します。報告値4収録 × dt1/dt4の8条件は`configs/literature`に保存し、こちらの推論実測値とは区別します。報告値だけでは同条件に再採点できないため、統一条件の比較やseed集計には混ぜません。再学習版は[論文に基づく専用再実装](ZHU2019.md)です。単位・損失・勾配・因果性をテストし、短いGPU実データ学習を確認しています。入力窓・解像度などの差分も記録し、学習データだけの効果とは断定しません。現状の5-bin voxel対照を[56]とは呼びません。[詳細](BASELINES.md)。

現在の段階では追加再学習を[56]に限定します。以下の5条件・履歴sweepは後続の要因比較用で、今すぐ全条件を学習する指示ではありません。

5条件: 非再帰count、非再帰voxel、Leaky count、RNN count、ConvGRU count。FireNet系の非再帰版はFireFlowNetです。countにも現在partition内の複数イベントが入っています。「過去なし」は過去partitionを追加しない意味です。5-bin voxelにするだけでは履歴は長くなりません。時間順序の符号化と履歴長は別要因です。

履歴sweepは1k/5k/15k/50k。stride=1k、loss=10k、proxy評価=50kを固定し、共通50k warmup後の同じanchorで採点します。再帰状態の記憶時間は入力履歴の制限を受けません。モデル間ではパラメータ数も異なり、厳密な同容量比較ではありません。

既定はAdam、lr=2e-4、clip=100、smoothness=0.001、100 epochs、batch=8。flipは系列全体で固定し、stateは系列境界でreset、10kごとにdetachします。最終の短いbatchも使い、lossはbatch平均です。同期batchは最短系列まで処理し、切り捨てイベント数をログへ保存します。長さが大きく異なる場合は同長に前処理するかbatch=1を全条件へ適用します。

matched学習は原論文の完全再現ではありません。val holdout、warmup除外、batch平均lossが相違点です。原学習は固定上流checkoutのtrain_flow.pyとconfigs/upstreamを参照します。

## 指標と時間軸

出力 × flow_scaleは**現在strideの変位**であり、入力history全体の変位ではありません。評価では最終strideの時間幅からGT区間の変位に換算するため、速度が大きく変化する長区間には限界があります。

EventWarping/FWL/RSATの数学部分は上流のままです。各strideの時間を[0,1]へ正規化し、0,1,2,…を加える規約です。イベントレートが変化する場合、全窓の実時間に一様な速度を仮定する指標とは異なります。秒単位の別実験を追加する場合はprotocol名を変えます。

FWLは大きいほど、RSATは小さいほど良いproxyですが、GT精度と同一ではありません。集中・画面外移動の影響を見るためwarped eventの画面内残存割合も保存します。非有限値はnullとして件数を残し、モデル選択・集計は失敗させます。

明示valid maskがあればGTの静止画素も採点できます。上流MVSECパックの通常の規約では(0,0)を無効とします。GT NaNは除外、予測NaNは停止。有効画素がない窓は誤差0としません。

## AffineObjects

events/xs,ys,ts,psは共通でも、GTやPNG時刻対応まで自動的に互換とは限りません。解像度は[480,640]。上流LayerSegは2 affine layersとbase channels=32を維持します。FWL/RSATは同じイベント窓で比較します。

maskモードは各GT時刻までの過去だけを使い、各anchorで共通warmupから状態を再構築します。連続イベント評価とは別モードです。前景channelは検証データで固定。DRはIoU≥0.5かつTP≥FP、空マスク同士はIoU=1です。

現在の上流configs/eval.ymlはwindow=100000、window_eval=50000に未使用とのコメントがあります。論文の条件と同一視しません。上流gt_testはGT時刻以降からイベントを読むため、本基盤の因果的mask評価とも異なります。

## Hebbian段階

配布モデルの再現、固定split、3 seedsの対照表、履歴曲線を先に確定します。その後data.sampleの履歴→入力表現部分へ具体的な更新則を追加します。未来・GTを使わないこと、状態reset/detach、更新コストを記録します。同容量の固定trace/EMA、更新停止、履歴順序のシャッフルなどの対照を事前に定めます。
