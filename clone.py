import argparse
import os
import sys

from gradio_client import Client, handle_file


def main():
    parser = argparse.ArgumentParser(description="Clone a voice via IndexTTS WebUI JSON API")
    parser.add_argument("wav", help="Path to reference wav file")
    parser.add_argument("--name", default=None, help="Voice name to save (default: wav filename)")
    parser.add_argument("--host", default="http://localhost:7860/", help="WebUI base URL")
    args = parser.parse_args()

    wav_path = os.path.abspath(args.wav)
    if not os.path.exists(wav_path):
        print(f"Error: file not found: {wav_path}", file=sys.stderr)
        sys.exit(2)

    voice_name = args.name
    if voice_name is None or str(voice_name).strip() == "":
        voice_name = os.path.splitext(os.path.basename(wav_path))[0]

    client = Client(args.host)
    try:
        try:
            result = client.predict(
                prompt_audio_path=handle_file(wav_path),
                voice_name=voice_name,
                api_name="/clone_voice",
            )
        except Exception as e1:
            msg = str(e1)
            if "Cannot find a function" in msg and "/clone_voice" in msg:
                result = client.predict(
                    prompt_audio_path=handle_file(wav_path),
                    voice_name=voice_name,
                    api_name="//clone_voice",
                )
            else:
                raise
    except Exception as e:
        print(f"Error: clone failed: {e}", file=sys.stderr)
        sys.exit(1)

    if isinstance(result, str) and result.strip():
        print(result)
        return

    print(f"Error: unexpected response: {result!r}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
