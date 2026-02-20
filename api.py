import subprocess
import json
import os
import argparse
from gradio_client import Client, handle_file
import time

def Gen(text, output_path):
	# replace text "\n" to "。"
	text = text.replace("\n", "。").replace("一一", "——")
	tm_start = time.time()
	output = None
	
	show_text = text
	if len(text) > 15:
		show_text = text[:15] + "..."
	print(f"{show_text} -> {output_path}")

	try:
		client = Client("http://localhost:7860/")
		result = client.predict(
				output_path=output_path,
				emo_control_method="与音色参考音频相同",
				prompt=handle_file('./input.wav'),
				text=text,
				emo_ref_path=handle_file('./input.wav'),
				emo_weight=0.8,
				vec1=0,
				vec2=0,
				vec3=0,
				vec4=0,
				vec5=0,
				vec6=0,
				vec7=0,
				vec8=0,
				emo_text="",
				emo_random=False,
				max_text_tokens_per_segment=120,
				param_16=True,
				param_17=0.8,
				param_18=30,
				param_19=0.8,
				param_20=0,
				param_21=3,
				param_22=10,
				param_23=1500,
				api_name="/gen_single"
		)
		if isinstance(result, dict) and 'value' in result:
			output = result['value']
		else:
			output = None
	except Exception as e:
		print(f"生成音频失败: {show_text} -> {output_path}: {e}")
		output = None
	tm_end = time.time()
	print(f"生成音频耗时: {tm_end - tm_start}")
	return output

def parse_id_range(range_str: str):
    """将范围字符串解析为一组整数ID。
    支持格式：
    - 单个数字："3"
    - 逗号分隔："3,5,7"
    - 闭区间："3-10"（包含两端）
    - 混合："1-3,5,7-9"
    返回：set[int]；若解析失败返回空集合。
    """
    ids = set()
    if not range_str:
        return ids
    for part in range_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                start_s, end_s = part.split('-', 1)
                start = int(start_s.strip())
                end = int(end_s.strip())
                if start > end:
                    start, end = end, start
                ids.update(range(start, end + 1))
            except ValueError:
                print(f"警告：无法解析区间 '{part}'，已跳过。")
        else:
            try:
                ids.add(int(part))
            except ValueError:
                print(f"警告：无法解析编号 '{part}'，已跳过。")
    return ids


def process_jsonl_with_curl(input_file, allowed_ids=None):
	"""
	读取 JSONL 文件，逐行调用 curl 下载 wav 文件。
	"""
	if not os.path.exists(input_file):
		print(f"错误：输入文件 '{input_file}' 不存在。")
		return

	print(f"开始处理文件: {input_file}")

	number = 1
	try:
		with open(input_file, 'r', encoding='utf-8') as f:
			for index, line in enumerate(f):
				try:
					# 1. 解析 JSON 行
					data = json.loads(line)
					paragraph_id = data.get("id", 0)
					text_to_synthesize = data.get('text', '')
					# 允许通过 allowed_ids 过滤需要处理的段落
					if allowed_ids and paragraph_id not in allowed_ids:
						continue

					if not text_to_synthesize:
						print(f"警告：第 {index + 1} 行的 'text' 字段为空，跳过。")
						continue

					# 2. 准备输出文件名 (例如: 000000.wav, 000001.wav)
					# f-string 的 :06d 格式化表示：用0填充，总宽度为6的整数
					if paragraph_id == 0:
						output_filename = f"_{number:06d}.wav"
					else:
						output_filename = f"{paragraph_id:06d}.wav"
					number += 1

					# 3. 生成并保存音频
					path = Gen(text_to_synthesize, output_filename)
					if path is None:
						print(f" 生成音频失败: {text_to_synthesize[:30]}...")
						continue
					# move file path -> output_filename
					time.sleep(5)
					os.rename(path, output_filename)
					print(f" {output_filename} -> 生成音频成功！")
				except json.JSONDecodeError:
					print(f"错误：无法解析第 {index + 1} 行的 JSON: {line.strip()}")
	except IOError as e:
		print(f"错误：无法读取文件 '{input_file}': {e}")

	print("\n所有任务处理完毕。")

def main():
	# 解析命令行参数：
	# 可选位置参数 input_file，用于覆盖默认的 INPUT_FILE
	parser = argparse.ArgumentParser(description="Process a JSONL file and call TTS API to download wav files.")
	parser.add_argument('input_file', nargs='?', help='Path to input JSONL file (default: input.jsonl)')
	parser.add_argument('--range', dest='range', help='IDs to process. Examples: "3-10", "3", "3,5,7", or mixed like "1-3,7-9"')
	args = parser.parse_args()

	allowed_ids = None
	if args.input_file is not None:
		allowed_ids = parse_id_range(args.range) if args.range else allowed_ids
		print(f"处理文件: {args.input_file}", f"{allowed_ids}")
		process_jsonl_with_curl(args.input_file, allowed_ids)
	else:
		path = Gen("时来天地皆同力", "./output.wav")
		if path is None:
			print("生成音频失败，请检查参数设置。")
		else:
			print(path)
			subprocess.run(['afplay', path])

if __name__ == '__main__':
	main()
