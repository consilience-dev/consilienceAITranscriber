"""diarize.py

Handles speaker diarization using pyannote.audio."""
from pyannote.audio import Pipeline
from typing import List, Dict

def load_diarization_pipeline(model_name: str = "pyannote/speaker-diarization") -> Pipeline:
    """Load pyannote speaker diarization pipeline."""
    pipeline = Pipeline.from_pretrained(model_name)
    return pipeline

def diarize_audio(audio_path: str, model_name: str = "pyannote/speaker-diarization") -> List[Dict]:
    """Perform speaker diarization and return list of segments.

    Each segment: {start, end, speaker}
    """
    pipeline = load_diarization_pipeline(model_name)
    diarization = pipeline(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return segments
