"""main.py

Orchestrates transcription pipeline and outputs JSON transcript.
Includes parallel processing capabilities for maximum GPU utilization."""
import json
import argparse
import time
import torch
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from transcribe import transcribe_audio
from diarize import diarize_audio
from extract_audio import extract_audio
# from silence_detect import detect_silences

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("main")

def main():
    parser = argparse.ArgumentParser(description="AI-driven transcription pipeline")
    parser.add_argument("input", help="Path to input audio/video file")
    parser.add_argument("-m", "--model", default="base", help="Whisper model name")
    parser.add_argument("-o", "--output", default="transcript.json", help="Output JSON file")
    
    # Performance options
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size for model inference (higher uses more GPU memory)")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.7,
                       help="Fraction of GPU memory to use (0.0-1.0)")
    args = parser.parse_args()
    
    # Log GPU information
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA-capable GPU(s)")
        for i in range(device_count):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Set maximum memory usage to protect GPU resources
        for i in range(device_count):
            try:
                torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, i)
                logger.info(f"Set GPU {i} memory fraction to {args.gpu_memory_fraction}")
            except:
                logger.warning(f"Could not set memory fraction on GPU {i}")
    else:
        logger.warning("No GPU detected! Processing will be slow.")
    
    start_time = time.time()
    
    # If input is not a .wav, extract audio
    input_path = args.input
    if not input_path.lower().endswith(".wav"):
        logger.info(f"Extracting audio from {input_path}...")
        input_path = extract_audio(input_path)
        logger.info(f"Audio extracted to {input_path}")

    # Process audio with GPU optimizations and monitoring - in parallel
    logger.info(f"Processing audio with model {args.model} in parallel mode")
    
    start_processing = time.time()
    
    # Run transcription and diarization concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        transcribe_future = executor.submit(transcribe_audio, input_path, args.model, args.batch_size)
        diarize_future = executor.submit(diarize_audio, input_path)
        
        # Wait for both to complete and get results
        segments = transcribe_future.result()
        diarization = diarize_future.result()
    
    parallel_duration = time.time() - start_processing
    logger.info(f"Parallel processing completed in {parallel_duration:.2f} seconds")
    logger.info(f"Found {len(segments)} transcript segments and {len(diarization)} speaker segments")
    
    # Assign speaker to each transcript segment
    for seg in segments:
        seg_start = seg["start"]
        seg["speaker"] = next(
            (d["speaker"] for d in diarization if seg_start >= d["start"] and seg_start < d["end"]),
            "Unknown"
        )
        
    output = {"segments": segments}
    
    # Calculate and log processing statistics
    duration = time.time() - start_time
    audio_duration = segments[-1]["end"] if segments else 0
    logger.info(f"Total processing completed in {duration:.2f} seconds")
    logger.info(f"Total segments: {len(segments)}")
    logger.info(f"Overall processing speed: {audio_duration/duration:.2f}x real-time")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Transcription saved to {args.output}")
    logger.info("Processing complete!")
    
    # Print a summary to standard output for easy visibility
    print(f"\nProcessing Summary:")
    print(f"  - Processing time: {duration:.2f} seconds")
    print(f"  - Audio duration: {audio_duration:.2f} seconds")
    print(f"  - Processing speed: {audio_duration/duration:.2f}x real-time")
    print(f"  - Output saved to: {args.output}")
    print(f"  - Total segments: {len(segments)}")
    print(f"  - Total speakers: {len(set(s['speaker'] for s in segments))}")
    if torch.cuda.is_available():
        print(f"  - Used GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  - Used CPU only")

if __name__ == "__main__":
    main()
