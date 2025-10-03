import subprocess
import json
import os
import argparse
from gradio_client import Client, handle_file

def Gen(text, output_path):
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
		return result['value']
	else:
		return None

INPUT_FILE = 'input.jsonl'
API_URL = 'YOUR_SPECIFIED_URL_HERE'

def process_jsonl_with_curl(input_file):
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
					text_to_synthesize = data.get('text', '')
					
					if not text_to_synthesize:
						print(f"警告：第 {index + 1} 行的 'text' 字段为空，跳过。")
						continue
					
					# 2. 准备输出文件名 (例如: 000000.wav, 000001.wav)
					# f-string 的 :06d 格式化表示：用0填充，总宽度为6的整数
					output_filename = f"{number:06d}.wav"
					number += 1
					
					# 3. 构造 curl 命令
					# -X POST: 指定请求方法为 POST
					# -d: 发送 POST 数据
					# -o: 指定输出文件
					# -s: 静默模式，不显示进度和错误
					# -S: 与 -s 一起使用时，只显示错误信息
					# --fail: 当 HTTP 请求失败 (如 4xx, 5xx) 时，返回一个非零的退出码
					
					print(f"正在处理第 {index + 1} 行: '{text_to_synthesize[:30]}...', 保存为 {output_filename}")
					
					
					path = Gen(text_to_synthesize, output_filename)
					if path is None:
						print(f" 生成音频失败: {text_to_synthesize}")
						continue
					
					# move file path -> output_filename
					os.rename(path, output_filename)
					print(f" {output_filename} -> 生成音频成功！")
				except json.JSONDecodeError:
					print(f"错误：无法解析第 {index + 1} 行的 JSON: {line.strip()}")
				except FileNotFoundError:
					print("致命错误：'curl' 命令未找到。请确保 curl 已安装并已添加到系统 PATH 中。")
					return # 如果 curl 不存在，直接退出
	except IOError as e:
		print(f"错误：无法读取文件 '{input_file}': {e}")
	
	print("\n所有任务处理完毕。")

def main():
	# 解析命令行参数：
	# 可选位置参数 input_file，用于覆盖默认的 INPUT_FILE
	# 可选参数 --api-url，用于覆盖默认的 API_URL
	parser = argparse.ArgumentParser(description="Process a JSONL file and call TTS API to download wav files.")
	parser.add_argument('input_file', nargs='?', help='Path to input JSONL file (default: input.jsonl)')
	args = parser.parse_args()
	
	if args.input_file is not None:
		process_jsonl_with_curl(args.input_file)
	else:
		path = Gen("时来天地皆同力", "./output.wav")
		if path is None:
			print("生成音频失败，请检查参数设置。")
		else:
			print(path)
			subprocess.run(['afplay', path])

if __name__ == '__main__':
	main()
