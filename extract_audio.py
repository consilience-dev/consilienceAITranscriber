"""extract_audio.py

Utility for extracting audio from video files to a standard WAV format for downstream processing.
"""
import subprocess
import os

def extract_audio(input_path: str, output_path: str = None, sample_rate: int = 16000) -> str:
    """Ensure input is a mono WAV audio file. If input is video or unsupported audio, extract/convert. Return path to WAV file."""
    supported_audio_exts = {'.wav', '.mp3', '.flac', '.m4a'}
    ext = os.path.splitext(input_path)[1].lower()
    if ext in supported_audio_exts:
        return input_path
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}.wav"
    cmd = [
        "ffmpeg", "-y", "-i", input_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1", output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path
