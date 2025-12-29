# tsavs_cutter.py

AVSファイル（AviSynth スクリプト）のTrim指定に基づいてMPEG-TSファイルをトリムするツール。

## 概要

AviSynthのTrim()指定をパースして、指定されたフレーム範囲のみを抽出したMPEG-TSファイルを作成します。複数のTrim範囲を連結し、PTS/DTS/PCRを自動的に調整して途切れのない動画を生成します。

## インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/noshiket/tsavs_cutter
cd tsavs_cutter
```

### 2. 動作確認

```bash
python3 tsavs_cutter.py --help
```

## 必要な環境

- **Python 3.7以上**
- **外部依存なし**（Python標準ライブラリのみ使用）

## 使用方法

```bash
python3 tsavs_cutter.py -i INPUT.ts -a TRIM.avs -o OUTPUT.ts
```

### オプション

- `-i, --input`: 入力TSファイル（必須）
- `-a, --avs`: AVSファイル（Trim指定を含む）（必須）
- `-o, --output`: 出力TSファイル（必須）

## AVSファイルの形式

AVSファイルには、AviSynthのTrim()関数を使用してフレーム範囲を指定します：

```avisynth
Trim(193,3188) ++ Trim(4988,22040) ++ Trim(23839,46345) ++ Trim(48145,48743)
```

### 形式の詳細

- `Trim(start, end)`: フレーム番号で範囲指定（0始まり）
- `++`: 複数範囲の連結（AviSynth互換）
- フレーム番号は元TSファイルのビデオフレームインデックスに基づく

## 動作の仕組み

### 1. ストリーム解析
- PAT/PMTを解析してPID情報を取得
- ビデオフレームのPTSインデックスを構築

### 2. フレーム範囲計算
- AVSファイルから`Trim(start,end)`を抽出
- 各範囲のフレーム番号をPTS値に変換
- セグメント間のPTSオフセットを計算

### 3. パケット処理
- 各セグメントのPTS範囲内のパケットを抽出
- PCR/PTS/DTSを調整して連続的な時間軸を作成
- Continuity Counterを更新

## 出力例

```
Parsing AVS file: my_trim.avs
Found 4 trim ranges:
  1. [193,3188]
  2. [4988,22040]
  3. [23839,46345]
  4. [48145,48743]

Input file: test.ts
Total video frames: (analyzing...)

Analyzing streams...
  Video PID: 0x100, Audio PID: ['0x110'], Caption PID: ['0x130', '0x138']

Building video index...
Found 53657 video frames (PTS range: 53572.907 - 55363.128)

Processing trim segments...

Trim segment 1: frames [193, 3188]
  Source PTS: 53579.246 - 53679.180 (duration: 99.933s, 2996 frames)
  Output PTS: 53579.246 - 53679.180 (offset: +0.000s)
  Packets: Video=615086, Audio=17729, Caption=213603

Trim segment 2: frames [4988, 22040]
  Source PTS: 53739.240 - 54308.208 (duration: 568.968s, 17053 frames)
  Output PTS: 53679.180 - 54248.148 (offset: -60.060s)
  Packets: Video=3230332, Audio=101034, Caption=1264735

...

Total output: 55019.051s, 43143 frames, 12553584 packets
Written: output.ts (2250.7 MB)
```

## 技術詳細

### PTS/DTS調整

各セグメントのPTS/DTSは、前のセグメントの終了時刻に続くように自動調整されます：

```
セグメント1: PTS 53579.246 - 53679.180 (offset: +0.000s)
セグメント2: PTS 53739.240 - 54308.208 → 調整後 53679.180 - 54248.148 (offset: -60.060s)
```

### PCR調整

PCR（Program Clock Reference）も同様にオフセット調整され、27MHzクロックで管理されます。

### Continuity Counter

各PIDのContinuity Counterは0から開始し、パケットごとに0-15で循環します。

## 注意点

### つなぎ目での問題

**セグメント間のつなぎ目で以下の問題が発生する可能性があります：**

- **音声の途切れ**: セグメント境界で音声が一瞬途切れることがあります
- **映像の乱れ**: GOP（Group of Pictures）境界が不適切な場合、映像が乱れることがあります

### 映像問題の解決方法

映像の乱れは **[tsreplace](https://github.com/rigaya/tsreplace)** を使用することで解決できます。

## 制限事項

- フレーム番号は0始まり
- 範囲の重複はチェックされません
- AVSファイルはUTF-8エンコーディング必須
- セグメント境界はIフレームでない可能性があるため、つなぎ目で問題が発生する場合があります

## エラーハンドリング

### フレーム範囲エラー
```
Error: Frame 60000 is out of range (total frames: 53657)
```

### AVSパースエラー
```
Error: No Trim() specifications found in AVS file
```

---
