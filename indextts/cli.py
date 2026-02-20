import os
import sys
import warnings
# Suppress warnings from tensorflow and other libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _default_output_name(ext: str = "mp3"):
    import datetime
    ts = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    ext = (ext or "mp3").lstrip(".")
    return f"{ts}.{ext}"


def _convert_wav_to_mp3(wav_path: str, mp3_path: str):
    try:
        import ffmpeg
    except Exception:
        raise RuntimeError("ffmpeg-python is not available")

    try:
        (
            ffmpeg
            .input(wav_path)
            .output(mp3_path, acodec="libmp3lame", audio_bitrate="192k")
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as e:
        raise RuntimeError(f"ffmpeg convert failed: {e}")


def _resolve_device(device_arg: str, fp16: bool):
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Please install it first.")
        sys.exit(1)

    if device_arg is not None:
        device = device_arg
    else:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            if fp16:
                print("WARNING: fp16 is not supported on CPU; disabling fp16.")
            fp16 = False
            print("WARNING: Running on CPU may be slow.")
    return device, fp16


def _load_tts2(config: str, model_dir: str, fp16: bool, device: str, deepspeed: bool = False, cuda_kernel: bool = False):
    if not os.path.exists(config):
        print(f"Config file {config} does not exist.")
        sys.exit(1)
    if not os.path.exists(model_dir):
        print(f"Model dir {model_dir} does not exist.")
        sys.exit(1)

    from indextts.infer_v2 import IndexTTS2
    return IndexTTS2(cfg_path=config, model_dir=model_dir, use_fp16=fp16, device=device, use_deepspeed=deepspeed, use_cuda_kernel=cuda_kernel)


def _cmd_list_voices(args):
    device, fp16 = _resolve_device(args.device, args.fp16)
    tts = _load_tts2(args.config, args.model_dir, fp16, device, deepspeed=args.deepspeed, cuda_kernel=args.cuda_kernel)
    for v in tts.list_voices():
        print(v)


def _cmd_clone_voice(args):
    if not os.path.exists(args.prompt):
        print(f"Audio prompt file {args.prompt} does not exist.")
        sys.exit(1)
    device, fp16 = _resolve_device(args.device, args.fp16)
    tts = _load_tts2(args.config, args.model_dir, fp16, device, deepspeed=args.deepspeed, cuda_kernel=args.cuda_kernel)
    saved_name = tts.clone_voice(args.prompt, voice_name=args.name, overwrite=args.overwrite, verbose=args.verbose)
    print(saved_name)


def _cmd_clone(args):
    if len(args.text.strip()) == 0:
        print("ERROR: Text is empty.")
        sys.exit(1)

    device, fp16 = _resolve_device(args.device, args.fp16)
    tts = _load_tts2(args.config, args.model_dir, fp16, device, deepspeed=args.deepspeed, cuda_kernel=args.cuda_kernel)

    voice = args.voice
    try:
        payload = tts.load_voice(voice)
    except Exception:
        payload = None

    if payload is None:
        if args.prompt is None:
            print(f"ERROR: Voice '{voice}' not found. Provide --prompt to create it.")
            sys.exit(1)
        if not os.path.exists(args.prompt):
            print(f"Audio prompt file {args.prompt} does not exist.")
            sys.exit(1)
        tts.clone_voice(args.prompt, voice_name=voice, overwrite=args.overwrite_voice, verbose=args.verbose)
        payload = tts.load_voice(voice)

    out = args.output
    if out is None:
        out = _default_output_name("mp3")

    out_ext = os.path.splitext(out)[1].lower()
    if out_ext in ("", ".wav"):
        out_wav = out if out_ext else (out + ".wav")
        tts.infer(spk_audio_prompt=payload, text=args.text.strip(), output_path=out_wav, verbose=args.verbose)
        print(out_wav)
        return

    if out_ext == ".mp3":
        tmp_wav = os.path.splitext(out)[0] + ".wav"
        tts.infer(spk_audio_prompt=payload, text=args.text.strip(), output_path=tmp_wav, verbose=args.verbose)
        try:
            _convert_wav_to_mp3(tmp_wav, out)
            os.remove(tmp_wav)
        except Exception as e:
            print(f"WARNING: Failed to convert to mp3: {e}")
            print(tmp_wav)
            return
        print(out)
        return

    tmp_wav = os.path.splitext(out)[0] + ".wav"
    tts.infer(spk_audio_prompt=payload, text=args.text.strip(), output_path=tmp_wav, verbose=args.verbose)
    print(tmp_wav)


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="IndexTTS Command Line")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", type=str, default="checkpoints/config.yaml", help="Path to the config file")
    common.add_argument("--model_dir", type=str, default="checkpoints", help="Path to the model directory")
    common.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
    common.add_argument("-d", "--device", type=str, default=None, help="Device to run the model on (cpu, cuda, mps, xpu).")
    common.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
    common.add_argument("--cuda-kernel", action="store_true", default=False, help="Use BigVGAN custom CUDA kernel if available")
    common.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")

    p_voices = sub.add_parser("voices", parents=[common], help="List saved voices")
    p_voices.set_defaults(func=_cmd_list_voices)

    p_clone_voice = sub.add_parser("clone-voice", parents=[common], help="Create a voice profile from a prompt audio")
    p_clone_voice.add_argument("prompt", type=str, help="Path to the audio prompt file")
    p_clone_voice.add_argument("name", type=str, nargs="?", default=None, help="Voice name (default: derived from prompt filename)")
    p_clone_voice.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing voice profile")
    p_clone_voice.set_defaults(func=_cmd_clone_voice)

    p_clone = sub.add_parser("clone", parents=[common], help="Synthesize speech using a saved voice profile")
    p_clone.add_argument("voice", type=str, help="Voice name")
    p_clone.add_argument("text", type=str, help="Text to be synthesized")
    p_clone.add_argument("output", type=str, nargs="?", default=None, help="Output audio filename (default: YYMMDD-hhmmss.mp3)")
    p_clone.add_argument("--prompt", type=str, default=None, help="Optional prompt audio to create the voice if it doesn't exist")
    p_clone.add_argument("--overwrite-voice", action="store_true", default=False, help="Overwrite voice profile when creating via --prompt")
    p_clone.set_defaults(func=_cmd_clone)

    return parser
def main():
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()