"""transcribe.py

Handles transcription of audio/video using OpenAI Whisper."""
import whisper
import torch
from typing import List, Dict

def load_model(model_name: str = "base") -> whisper.Whisper:
    """Load Whisper model with GPU support if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name)
    return model.to(device)

def transcribe_audio(audio_path: str, model_name: str = "base") -> List[Dict]:
    """Transcribe an audio or video file and return a list of segments.
    
    Each segment is a dict with keys: start (float), end (float), text (str).
    """
    model = load_model(model_name)
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })
    return segments
