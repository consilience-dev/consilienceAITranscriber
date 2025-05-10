# AI-Driven Transcription & Diarization Tool for Twitch VODs

This project provides GPU-accelerated transcription and speaker diarization for audio/video files, with a focus on processing Twitch VODs.

## Requirements
- Python 3.9+
- NVIDIA GPU with CUDA support (optional but recommended)
- FFmpeg (for pydub)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py <input_file.mp4> -m base -o transcript.json
```

## Modules
- **transcribe.py**: Uses OpenAI Whisper for transcription.
- **diarize.py**: Uses pyannote.audio for speaker diarization.
- **silence_detect.py**: Uses pydub to detect silences.
- **main.py**: Orchestrates the pipeline and outputs JSON.

## Output Format
Each segment in the output JSON includes:
- **start**: float (seconds) segmentation start time
- **end**: float (seconds) segmentation end time
- **text**: string transcript text
- **speaker**: speaker label (e.g., 'Speaker 1')

## Testing
Run unit tests with:
```bash
pytest
```
