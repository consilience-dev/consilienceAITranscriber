import pytest

from transcribe import load_model, transcribe_audio

def test_load_model():
    # Loading a small Whisper model should return an object with a transcribe method
    model = load_model("tiny")
    assert hasattr(model, "transcribe"), "Model should have a transcribe method"


def test_transcribe_audio_invalid_path():
    # Transcribing a non-existent file should raise an exception
    with pytest.raises(Exception):
        transcribe_audio("nonexistent_file.mp3", model_name="tiny")
