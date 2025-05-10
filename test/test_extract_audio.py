import os
import pytest
from extract_audio import extract_audio

def test_extract_audio_passthrough(tmp_path):
    # Should return the same path for .wav, .mp3, .flac, .m4a
    for ext in [".wav", ".mp3", ".flac", ".m4a"]:
        fake_audio = tmp_path / f"audio{ext}"
        fake_audio.write_bytes(b"fake audio data")
        assert extract_audio(str(fake_audio)) == str(fake_audio)

def test_extract_audio_conversion(tmp_path, monkeypatch):
    # Should call ffmpeg for .mkv or other video files
    called = {}
    def fake_run(cmd, check, stdout, stderr):
        called['ran'] = True
        return None
    monkeypatch.setattr("subprocess.run", fake_run)
    fake_video = tmp_path / "video.mkv"
    fake_video.write_bytes(b"fake video data")
    out = extract_audio(str(fake_video))
    assert called.get('ran')
    assert out.endswith(".wav")
