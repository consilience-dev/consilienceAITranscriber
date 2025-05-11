"""transcribe.py

Handles transcription of audio/video using OpenAI Whisper.
Includes GPU monitoring and optimization for batch processing."""
import whisper
import torch
import time
from typing import List, Dict

# Set up basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("transcribe")

def log_gpu_memory(prefix: str = ""):
    """Log GPU memory usage if GPU is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
        logger.info(f"{prefix}GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def load_model(model_name: str = "base") -> whisper.Whisper:
    """Load Whisper model with GPU support if available."""
    # Log device information
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        log_gpu_memory("Before model load: ")
    else:
        logger.info("GPU not available, using CPU only (this will be slow)")
    
    # Load model
    model = whisper.load_model(model_name)
    model = model.to(device)
    
    # Log memory after model load
    if device == "cuda":
        log_gpu_memory("After model load: ")
    
    return model

def transcribe_audio(audio_path: str, model_name: str = "base", batch_size: int = 16) -> List[Dict]:
    """Transcribe an audio or video file and return a list of segments.
    
    Each segment is a dict with keys: start (float), end (float), text (str).
    
    Args:
        audio_path: Path to the audio file
        model_name: Whisper model name (tiny, base, small, medium, large-v3, etc.)
        batch_size: Number of audio segments to process in parallel (higher values use more GPU memory)
    """
    logger.info(f"Starting transcription of {audio_path} with model {model_name}")
    start_time = time.time()
    
    # Load model and monitor GPU
    model = load_model(model_name)
    
    # Log pre-inference GPU state
    if torch.cuda.is_available():
        log_gpu_memory("Before inference: ")
    
    # Run inference with fp16 if GPU available 
    # Note: Whisper handles batching internally and doesn't expose a batch_size parameter
    # The actual batch size depends on the model and available memory
    logger.info(f"Running inference with fp16={torch.cuda.is_available()}")
    result = model.transcribe(
        audio_path, 
        fp16=torch.cuda.is_available()
    )
    
    # Log post-inference GPU state and timing
    if torch.cuda.is_available():
        log_gpu_memory("After inference: ")
        # Explicitly clear CUDA cache to free memory
        torch.cuda.empty_cache()
        log_gpu_memory("After cache clear: ")
    
    # Process segments
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })
    
    duration = time.time() - start_time
    audio_seconds = segments[-1]["end"] if segments else 0
    logger.info(f"Transcription completed in {duration:.2f} seconds")
    logger.info(f"Processing speed: {audio_seconds/duration:.2f}x real-time")
    
    return segments
