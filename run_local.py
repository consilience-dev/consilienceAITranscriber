#!/usr/bin/env python3
"""
Local execution script for ConsilienceAITranscriber

This script provides an enhanced local testing environment that:
1. Auto-detects GPU availability and capabilities
2. Configures resource protection for local testing
3. Runs the transcription pipeline with the detected hardware configuration

Following the cardinal rule of 'Design for GPU Local Use First', this ensures
the code is testable on a developer's machine while protecting GPU resources.
"""
import os
import sys
import argparse
import subprocess
import logging
import json
from pathlib import Path
import shutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("transcriber")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ConsilienceAITranscriber - GPU-accelerated transcription & diarization"
    )
    parser.add_argument("input", help="Path to input audio/video file")
    parser.add_argument(
        "-m", "--model", 
        default="base", 
        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v3", "turbo"],
        help="Whisper model size"
    )
    parser.add_argument("-o", "--output", default=None, help="Output JSON file (default: <input>-transcript.json)")
    parser.add_argument("--max-gpu-memory", type=float, default=0.7, help="Maximum GPU memory fraction to use (0.0-1.0)")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--chunk-size", type=int, default=5, help="Chunk size in minutes for large files")
    
    return parser.parse_args()

def detect_hardware():
    """Detect available hardware and capabilities"""
    hardware_info = {
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "total_gpu_memory_gb": 0,
        "usable_gpu_memory_gb": 0,
        "cpu_count": os.cpu_count() or 1,
    }
    
    # Check for GPU availability
    if HAS_TORCH and torch.cuda.is_available():
        hardware_info["gpu_available"] = True
        hardware_info["gpu_count"] = torch.cuda.device_count()
        
        for i in range(hardware_info["gpu_count"]):
            gpu_name = torch.cuda.get_device_name(i)
            hardware_info["gpu_names"].append(gpu_name)
            
            # Get GPU memory in GB
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            hardware_info["total_gpu_memory_gb"] += gpu_memory
    
    return hardware_info

def process_file(args, hardware_info):
    """Process the audio/video file with the detected hardware"""
    input_path = Path(args.input)
    
    # Set output path if not specified
    if args.output:
        output_path = Path(args.output)
    else:
        # Create output filename by replacing the suffix and adding -transcript
        stem = input_path.stem
        output_path = input_path.parent / f"{stem}-transcript.json"
    
    # Set environment variables based on hardware
    env = os.environ.copy()
    
    if hardware_info["gpu_available"] and not args.force_cpu:
        logger.info(f"Using GPU acceleration with {hardware_info['gpu_count']} GPU(s)")
        logger.info(f"Available GPUs: {', '.join(hardware_info['gpu_names'])}")
        
        # Calculate usable memory based on max_gpu_memory setting
        usable_memory = hardware_info["total_gpu_memory_gb"] * args.max_gpu_memory
        hardware_info["usable_gpu_memory_gb"] = usable_memory
        
        logger.info(f"Reserving {usable_memory:.2f} GB of {hardware_info['total_gpu_memory_gb']:.2f} GB GPU memory")
        
        # Set CUDA visible devices (all by default)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(hardware_info["gpu_count"]))
    else:
        if args.force_cpu:
            logger.info("Forcing CPU usage as requested")
        else:
            logger.info("No GPU detected, using CPU")
        env["CUDA_VISIBLE_DEVICES"] = ""
    
    # Run the main transcription pipeline
    cmd = [
        sys.executable, 
        "main.py", 
        str(input_path),
        "--model", args.model,
        "--output", str(output_path)
    ]
    
    # Note: main.py doesn't currently support chunk-size
    # When you add parallelization to main.py, uncomment this:
    #
    # if hardware_info["gpu_available"] and not args.force_cpu:
    #     cmd.extend(["--chunk-size", str(args.chunk_size)])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, env=env, check=True)
        logger.info(f"Transcription completed successfully. Output saved to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Transcription failed with exit code {e.returncode}")
        return False

def main():
    """Main function"""
    args = parse_args()
    
    logger.info("Starting ConsilienceAITranscriber in local mode")
    
    # Detect hardware
    hardware_info = detect_hardware()
    
    # Log hardware information
    logger.info(f"Hardware detection complete:")
    logger.info(f"  CPU cores: {hardware_info['cpu_count']}")
    if hardware_info["gpu_available"]:
        logger.info(f"  GPU count: {hardware_info['gpu_count']}")
        logger.info(f"  GPU models: {', '.join(hardware_info['gpu_names'])}")
        logger.info(f"  Total GPU memory: {hardware_info['total_gpu_memory_gb']:.2f} GB")
    else:
        logger.info("  No GPU detected")
    
    # Process the file
    success = process_file(args, hardware_info)
    
    if success:
        logger.info("Processing completed successfully")
        sys.exit(0)
    else:
        logger.error("Processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
