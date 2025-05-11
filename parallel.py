"""parallel.py

Provides parallel processing capabilities for audio transcription and diarization.
Enables chunking of large audio files for maximum GPU utilization.
"""
import os
import time
import torch
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment

from transcribe import transcribe_audio, log_gpu_memory as transcribe_log_gpu
from diarize import diarize_audio

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("parallel")


def split_audio(
    audio_path: str, 
    chunk_duration_ms: int = 300000,  # 5 minutes in milliseconds
    overlap_ms: int = 5000  # 5 seconds overlap between chunks
) -> List[Tuple[str, int]]:
    """
    Split a large audio file into smaller chunks for parallel processing.
    
    Args:
        audio_path: Path to the original audio file
        chunk_duration_ms: Duration of each chunk in milliseconds
        overlap_ms: Overlap between chunks in milliseconds to avoid boundary issues
        
    Returns:
        List of tuples containing (chunk_path, start_ms)
    """
    logger.info(f"Splitting audio file {audio_path} into chunks of {chunk_duration_ms/1000}s with {overlap_ms/1000}s overlap")
    
    # Load audio file
    audio = AudioSegment.from_file(audio_path)
    total_duration_ms = len(audio)
    logger.info(f"Audio duration: {total_duration_ms/1000:.2f} seconds")
    
    # Create temp directory for chunks
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory for chunks: {temp_dir}")
    
    # Create chunks with overlap
    chunks = []
    for start_ms in range(0, total_duration_ms, chunk_duration_ms - overlap_ms):
        # Ensure we don't go beyond the audio duration
        end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
        
        # Get chunk audio segment
        chunk = audio[start_ms:end_ms]
        
        # Generate output filename
        chunk_path = os.path.join(temp_dir, f"chunk_{start_ms//1000}_{end_ms//1000}.wav")
        
        # Export chunk to WAV file
        chunk.export(chunk_path, format="wav")
        
        # Store chunk info
        chunks.append((chunk_path, start_ms))
        
        # If we've reached the end of the audio, stop
        if end_ms >= total_duration_ms:
            break
    
    logger.info(f"Created {len(chunks)} audio chunks")
    return chunks


def process_chunk(
    chunk_info: Tuple[str, int],
    model_name: str = "base",
    batch_size: int = 16
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a single audio chunk with transcription and diarization.
    
    Args:
        chunk_info: Tuple containing (chunk_path, offset_ms)
        model_name: Whisper model name
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (transcription_segments, diarization_segments) with adjusted timestamps
    """
    chunk_path, offset_ms = chunk_info
    offset_sec = offset_ms / 1000.0
    
    logger.info(f"Processing chunk {chunk_path} (offset: {offset_sec:.2f}s)")
    
    # Run transcription and diarization in parallel using threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        transcribe_future = executor.submit(
            transcribe_audio, chunk_path, model_name, batch_size
        )
        diarize_future = executor.submit(
            diarize_audio, chunk_path
        )
        
        # Get results
        transcription = transcribe_future.result()
        diarization = diarize_future.result()
    
    # Adjust timestamps by adding offset
    for segment in transcription:
        segment["start"] += offset_sec
        segment["end"] += offset_sec
    
    for segment in diarization:
        segment["start"] += offset_sec
        segment["end"] += offset_sec
    
    logger.info(f"Completed chunk {chunk_path}: {len(transcription)} transcript segments, "
                f"{len(set(s['speaker'] for s in diarization))} speakers")
    
    return transcription, diarization


def process_audio_in_parallel(
    audio_path: str,
    model_name: str = "base",
    max_workers: Optional[int] = None,
    chunk_duration_minutes: int = 5,
    batch_size: int = 16
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process an audio file in parallel chunks.
    
    Args:
        audio_path: Path to the audio file
        model_name: Whisper model name to use
        max_workers: Maximum number of parallel workers (defaults to number of cores)
        chunk_duration_minutes: Duration of each chunk in minutes
        batch_size: Batch size for model inference
        
    Returns:
        Tuple of (transcription_segments, diarization_segments)
    """
    start_time = time.time()
    
    # Auto-detect optimal number of workers based on available resources
    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        # Use at most CPU count or GPU count + 1, whichever is smaller
        max_workers = min(cpu_count, max(1, gpu_count + 1))
    
    logger.info(f"Starting parallel processing with {max_workers} workers")
    
    # Log GPU information
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        transcribe_log_gpu("Initial: ")
    
    # Split audio into chunks
    chunk_duration_ms = chunk_duration_minutes * 60 * 1000
    chunks = split_audio(audio_path, chunk_duration_ms)
    
    # Process chunks in parallel
    all_transcriptions = []
    all_diarizations = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunk processing tasks
        futures = {
            executor.submit(
                process_chunk, chunk_info, model_name, batch_size
            ): chunk_info 
            for chunk_info in chunks
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            chunk_info = futures[future]
            try:
                transcription, diarization = future.result()
                all_transcriptions.extend(transcription)
                all_diarizations.extend(diarization)
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_info[0]}: {str(e)}")
    
    # Sort results by start time
    all_transcriptions.sort(key=lambda x: x["start"])
    all_diarizations.sort(key=lambda x: x["start"])
    
    # Log processing statistics
    duration = time.time() - start_time
    audio_duration = max(
        [s["end"] for s in all_transcriptions] if all_transcriptions else [0] +
        [s["end"] for s in all_diarizations] if all_diarizations else [0]
    )
    logger.info(f"Parallel processing completed in {duration:.2f} seconds")
    logger.info(f"Total audio duration: {audio_duration:.2f} seconds")
    logger.info(f"Processing speed: {audio_duration/duration:.2f}x real-time")
    logger.info(f"Total transcript segments: {len(all_transcriptions)}")
    logger.info(f"Total unique speakers: {len(set(s['speaker'] for s in all_diarizations))}")
    
    return all_transcriptions, all_diarizations


def merge_speaker_segments(diarization: List[Dict]) -> List[Dict]:
    """
    Merge consecutive segments from the same speaker.
    
    Args:
        diarization: List of diarization segments with {start, end, speaker}
        
    Returns:
        List of merged diarization segments
    """
    if not diarization:
        return []
    
    # Sort by start time
    sorted_segments = sorted(diarization, key=lambda x: x["start"])
    
    # Initialize with first segment
    merged = [dict(sorted_segments[0])]
    
    # Iterate through remaining segments
    for segment in sorted_segments[1:]:
        last = merged[-1]
        
        # If same speaker and continuous, merge
        if (segment["speaker"] == last["speaker"] and 
            segment["start"] - last["end"] < 0.5):  # 500ms tolerance
            last["end"] = segment["end"]
        else:
            # Add as new segment
            merged.append(dict(segment))
    
    return merged
