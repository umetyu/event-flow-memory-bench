# データの配置

データと重みはgitに含めません。保存先はYAMLとマニフェストで指定します。

| データ | 配布元 | 用途 |
|---|---|---|
| UZH-FPV / MVSEC / ECD / HQFのHDF5 | [event_flow README](https://github.com/tudelft/event_flow/tree/e81f9630a145da8c7e57b2cfd38cf6c0b47ac14f#download-datasets) | 主実験・補助評価 |
| NeurIPS配布モデル | [event_flow README](https://github.com/tudelft/event_flow/tree/e81f9630a145da8c7e57b2cfd38cf6c0b47ac14f#download-models) | 数値再現 |
| AffineObjects / 配布モデル | [event_segmentation README](https://github.com/Fulmen67/event_segmentation/tree/a6e28dce82720f41c0e62781e01d015a8f3b2a26) | 副実験 |

主データは圧縮19.4GB・展開94GB、AffineObjectsはREADMEで圧縮3.1GB・展開3.4GB。実ファイルのGT内容は取得後に検査します。リンクの期限やログインは配布元に依存します。

## HDF5契約

```
events/xs  N, 整数画素x
events/ys  N, 整数画素y
events/ts  N, 秒、昇順、float64推奨
events/ps  N, {0,1} または {-1,+1}
```

matchedはps<=0を負極性とし、時刻はfloat32化前に先頭時刻を引きます。inspectは全イベントをchunkで走査して時刻順・座標・列長を検査します。解像度推測や暗黙のcropはしません。

原再現ローダーはps∈{0,1}、属性t0とdurationも必要です。生のMVSEC/UZHファイルを直接読むものではなく、配布済みの前処理パックを使います。

MVSECはflow_dt1/<name> / flow_dt4/<name>に(2,H,W)のpixel displacement、各dataset属性timestampを想定します。時刻順の隣接timestampをGT区間とし、明示timestamp_startがあれば優先します。dt4をdt1の単純4倍や加算で生成しません。flow_dtN_valid/<name>があれば明示valid maskに利用できます。

## groupとsplit

data/uzh-original-recordings.jsonの例。実ファイルと元記録の対応に置き換えてください。

```json
{
  "recording-a-crop-001.h5": "recording-a",
  "recording-a-crop-002.h5": "recording-a",
  "recording-b-crop-001.h5": "recording-b"
}
```

マニフェスト形式:

```json
{
  "version": 1,
  "dataset": "uzh_fpv",
  "train": [{"path": "/data/training/a.h5", "group": "recording-a", "id": "a"}],
  "val": [{"path": "/data/training/b.h5", "group": "recording-b", "id": "b"}],
  "test": []
}
```

相対パスはマニフェスト自身のディレクトリから解決。idは結果の系列名。groupは元記録やシーンの独立性を表し、split間の重複を禁止します。全seed・全モデルに共通のマニフェストを使います。

AffineObjectsもdataset=affineobjectsで同形式。公式train内でvalを分け、公式testを保持します。同シーンの派生がある場合は同じgroupにします。

## マスクGT

各testエントリーへ時刻とPNGを明示します。timestampはevents/tsと同じ基準です。

```json
{
  "path": "/data/affine/test/scene.h5",
  "group": "scene-test",
  "id": "scene-test",
  "masks": [{"timestamp": 0.4, "path": "/data/affine/masks/scene-0.4.png"}]
}
```

PNGは元の[H,W]、0/255の二値マスク。無断transpose/resizeをしません。クラス符号、前景channel、時刻単位を確認してください。過去イベントが不足するanchorは除外します。

データは外部ストレージやGPUサーバーへ置きます。HDF5をsliceし、全データをRAMへ読み込みません。Slurmテンプレートは提供しますがジョブ投入は行っていません。
