import subprocess
import sys

def install_packages():
    packages = [
        "openai-whisper",
        "whisperx",
        "pyannote.audio",
        "pydub",
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "sentencepiece",
        "vaderSentiment",
        "matplotlib",
        "soundfile",
        "scipy",
        "AutoTokenizer",
        "ctranslate2",
        "gtts"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U"] + packages)

if __name__ == "__main__":
    install_packages()