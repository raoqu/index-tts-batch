#!/bin/bash

OUT_FILE="merged.mp3"
TMP_DIR="./tmp_merge"
mkdir -p "$TMP_DIR"

# 清理旧文件
rm -f "$OUT_FILE" "$TMP_DIR"/*

# 目标目录（可选参数），默认当前目录
DIR="${1:-.}"

# 获取所有 wav 文件，排除 input*.wav
shopt -s nullglob
files=( "$DIR"/*.wav )
selected=()
for f in "${files[@]}"; do
    base="$(basename "$f")"
    if [[ "$base" == input*.wav ]]; then
        continue
    fi
    # 记录为绝对路径，便于 ffmpeg concat
    abs_path="$(cd "$(dirname "$f")" && pwd)/$base"
    selected+=( "$abs_path" )
done
total=${#selected[@]}

if [ $total -eq 0 ]; then
    echo "目录 '$DIR' 下没有可合并的 .wav 文件（已排除 input*.wav）"
    exit 1
fi

echo "目录 '$DIR' 下共找到 $total 个 .wav 文件（已排除 input*.wav）"

# 动态设定批次大小
if [ $total -le 100 ]; then
    BATCH_SIZE=$total
elif [ $total -le 1000 ]; then
    BATCH_SIZE=100
else
    # 约分成 10 批
    BATCH_SIZE=$(( (total + 9) / 10 ))
fi

echo "采用每批 $BATCH_SIZE 个文件进行合并"

# 分批合成中间文件
i=0
batch=1
LIST_FILE="$TMP_DIR/list_$batch.txt"
> "$LIST_FILE"

for f in "${selected[@]}"; do
    echo "file '$f'" >> "$LIST_FILE"
    i=$((i+1))
    if [ $i -ge $BATCH_SIZE ]; then
        ffmpeg -y -f concat -safe 0 -i "$LIST_FILE" -c copy "$TMP_DIR/part_$batch.wav"
        echo "生成中间文件: part_$batch.wav"
        batch=$((batch+1))
        LIST_FILE="$TMP_DIR/list_$batch.txt"
        > "$LIST_FILE"
        i=0
    fi
done

# 最后一批如果有剩余
if [ -s "$LIST_FILE" ]; then
    ffmpeg -y -f concat -safe 0 -i "$LIST_FILE" -c copy "$TMP_DIR/part_$batch.wav"
    echo "生成中间文件: part_$batch.wav"
fi

# 合并所有中间文件
> "$TMP_DIR/final_list.txt"
for p in $(ls "$TMP_DIR"/part_*.wav | sort); do
    echo "file '$PWD/$p'" >> "$TMP_DIR/final_list.txt"
done

ffmpeg -y -f concat -safe 0 -i "$TMP_DIR/final_list.txt" -c:a libmp3lame -q:a 2 "$OUT_FILE"

echo "合并完成：$OUT_FILE"
