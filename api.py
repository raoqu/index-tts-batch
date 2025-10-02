import subprocess
from gradio_client import Client, handle_file

def Gen(text, output_path):
	client = Client("http://localhost:7860/")
	result = client.predict(
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
			output_path=output_path,
			api_name="/gen_single"
	)
	if isinstance(result, dict) and 'value' in result:
		return result['value']
	else:
		return result

path = Gen("A B C", "./output.wav")
print(path)
subprocess.run(['afplay', path])