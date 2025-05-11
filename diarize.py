"""diarize.py

Handles speaker diarization using pyannote.audio with GPU acceleration and performance monitoring."""
import os
import time
import torch
from typing import List, Dict
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline, Annotation

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("diarize")

def log_gpu_memory(prefix: str = ""):
    """Log GPU memory usage if GPU is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"{prefix}GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def load_diarization_pipeline(model_name: str = "pyannote/speaker-diarization") -> Pipeline:
    """Load pyannote speaker diarization pipeline with GPU support if available."""
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"Diarization using GPU: {torch.cuda.get_device_name(0)}")
        log_gpu_memory("Before diarization model load: ")
    else:
        logger.info("GPU not available for diarization, using CPU only (this will be slow)")
    
    # Set device explicitly in environment
    os.environ["PYTORCH_DEVICE"] = device
    
    # Load pipeline
    pipeline = Pipeline.from_pretrained(model_name)
    
    # Log memory after model load
    if device == "cuda":
        log_gpu_memory("After diarization model load: ")
        
    return pipeline

def diarize_audio(audio_path: str, model_name: str = "pyannote/speaker-diarization", segmentation_batch_size: int = 32) -> List[Dict]:
    """Perform speaker diarization and return list of segments.

    Each segment: {start, end, speaker}
    
    Args:
        audio_path: Path to the audio file
        model_name: Name of the diarization model to use
        segmentation_batch_size: Batch size for segmentation (higher values use more GPU memory)
    """
    logger.info(f"Starting diarization of {audio_path}")
    start_time = time.time()
    
    # Load pipeline
    pipeline = load_diarization_pipeline(model_name)
    
    # Log pre-inference GPU state
    if torch.cuda.is_available():
        log_gpu_memory("Before diarization inference: ")
    
    # Configure segmentation batch size if using GPU
    if torch.cuda.is_available():
        # Set segmentation batch size (if the model supports it)
        try:
            pipeline.segmentation_batch_size = segmentation_batch_size
            logger.info(f"Set segmentation_batch_size to {segmentation_batch_size}")
        except:
            logger.warning("Could not set segmentation_batch_size - model may not support it")
    
    # Run diarization
    diarization = pipeline(audio_path)
    
    # Log post-inference GPU state
    if torch.cuda.is_available():
        log_gpu_memory("After diarization inference: ")
        # Clear cache to free memory
        torch.cuda.empty_cache()
        log_gpu_memory("After diarization cache clear: ")
    
    # Extract segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    
    # Log performance metrics
    duration = time.time() - start_time
    audio_duration = max([s["end"] for s in segments]) if segments else 0
    logger.info(f"Diarization completed in {duration:.2f} seconds")
    logger.info(f"Processing speed: {audio_duration/duration:.2f}x real-time")
    logger.info(f"Identified {len(set(s['speaker'] for s in segments))} unique speakers")
    
    return segments
