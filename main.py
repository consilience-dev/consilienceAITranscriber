"""main.py

Orchestrates transcription pipeline and outputs JSON transcript."""
import json
import argparse
from transcribe import transcribe_audio
from diarize import diarize_audio
from extract_audio import extract_audio
# from silence_detect import detect_silences

def main():
    parser = argparse.ArgumentParser(description="AI-driven transcription pipeline")
    parser.add_argument("input", help="Path to input audio/video file")
    parser.add_argument("-m", "--model", default="base", help="Whisper model name")
    parser.add_argument("-o", "--output", default="transcript.json", help="Output JSON file")
    args = parser.parse_args()

    # If input is not a .wav, extract audio
    input_path = args.input
    if not input_path.lower().endswith(".wav"):
        print(f"Extracting audio from {input_path}...")
        input_path = extract_audio(input_path)
        print(f"Audio extracted to {input_path}")

    # Transcription
    segments = transcribe_audio(input_path, model_name=args.model)
    # Speaker diarization
    diarization = diarize_audio(input_path)
    # Assign speaker to each transcript segment
    for seg in segments:
        seg_start = seg["start"]
        seg["speaker"] = next(
            (d["speaker"] for d in diarization if seg_start >= d["start"] and seg_start < d["end"]),
            "Unknown"
        )
    output = {"segments": segments}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Transcription saved to {args.output}")

if __name__ == "__main__":
    main()
