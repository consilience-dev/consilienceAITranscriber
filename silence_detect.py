"""silence_detect.py

Detects silences to segment audio using pydub."""
from pydub import AudioSegment, silence
from typing import List, Dict

def detect_silences(audio_path: str, min_silence_len: int = 500, silence_thresh: int = -40) -> List[Dict]:
    """Return list of silence segments with start and end times in ms."""
    audio = AudioSegment.from_file(audio_path)
    silent_ranges = silence.detect_silence(audio, min_silence_len, silence_thresh)
    # Each range is [start_ms, end_ms]
    return [{"start_ms": start, "end_ms": end} for start, end in silent_ranges]
