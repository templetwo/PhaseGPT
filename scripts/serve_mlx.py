#!/usr/bin/env python3
"""
scripts/serve_mlx.py

A production-grade utility for the lifecycle management of PhaseGPT Oracle 
and other LLMs on Apple Silicon via MLX.

Features:
- Automated Model Retrieval: Fetches from Hugging Face Hub.
- Intelligent Conversion: Detects if conversion is needed; handles 4-bit/8-bit quantization.
- OpenAI Compatibility: Serves the model via an API compatible with OpenAI clients.
- Resource Isolation: Runs conversion and serving in separate process spaces to manage UMA effectively.

Author: Domain Expert / MLX Architect
"""

import argparse
import sys
import os
import shutil
import logging
import subprocess
import json
from pathlib import Path

# Configure structured logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def check_environment():
    """
    Verifies the runtime environment for critical dependencies.
    """
    try:
        import mlx_lm
        import huggingface_hub
        import mlx.core
        logger.info(f"Environment Check Passed: MLX {mlx.core.__version__} | MLX-LM Installed")
    except ImportError as e:
        logger.error(f"Critical Dependency Missing: {e.name}")
        logger.error("Please install required packages: pip install mlx-lm huggingface_hub")
        sys.exit(1)

def validate_local_model(path: Path) -> bool:
    """
    Validates if a directory contains a functional MLX model.
    A valid model requires:
    1. config.json
    2. tokenizer_config.json (usually)
    3. Weights (.npz or .safetensors)
    """
    if not path.exists() or not path.is_dir():
        return False
    
    required_files = ["config.json"]
    has_config = all((path / f).exists() for f in required_files)
    
    # Check for weights
    has_weights = any(path.glob("*.npz")) or any(path.glob("*.safetensors"))
    
    if has_config and has_weights:
        logger.info(f"Valid MLX model detected at {path}")
        return True
    return False

def get_model_path(args) -> str:
    """
    Determines the target model path.
    If args.model is a local path, use it.
    If it's a Repo ID, generate a local cache path based on the model name and quantization.
    """
    # Check if the input is already a valid local path
    input_path = Path(args.model).resolve()
    if input_path.exists():
        # If it's a raw HF/PyTorch model, we might need to convert it to a new location
        # If it's already MLX format (checked via validation), return it.
        if validate_local_model(input_path):
            return str(input_path)
        else:
            # It exists but isn't MLX ready (likely standard HF format).
            # We will convert it to ./mlx_models/...
            logger.info(f"Local path {input_path} found but requires conversion/quantization.")
            # Fallthrough to generate a destination path
            pass

    # Construct destination path for conversion
    # Sanitize name: "organization/model" -> "organization_model" or local path sanitation
    sanitized_name = str(args.model).replace("/", "_").replace(".", "_")
    if sanitized_name.startswith("_"): sanitized_name = sanitized_name[1:]
    
    quant_suffix = f"_{args.bits}bit" if not args.no_quantize else "_fp16"
    model_dir_name = f"{sanitized_name}{quant_suffix}"
    
    # Use a specific 'mlx_models' directory in the current working path
    base_dir = Path(os.getcwd()) / "mlx_models"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    return str(base_dir / model_dir_name)

def perform_conversion(source_path: str, dest_path: str, args):
    """
    Executes the mlx_lm.convert utility.
    """
    logger.info(f"Initiating Conversion Pipeline for: {source_path}")
    logger.info(f"Destination: {dest_path}")
    
    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", source_path,
        "--mlx-path", dest_path
    ]

    if not args.no_quantize:
        logger.info(f"Quantization Enabled: {args.bits}-bit (Group Size: {args.group_size})")
        cmd.append("-q")
        cmd.extend(["--q-bits", str(args.bits)])
        cmd.extend(["--q-group-size", str(args.group_size)])
    else:
        logger.info("Quantization Disabled: Keeping original precision (FP16/BF16)")

    try:
        logger.info("Starting conversion process... (This may take time based on model size)")
        subprocess.run(cmd, check=True)
        logger.info("Conversion completed successfully.")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion process failed with exit code {e.returncode}")
        logger.error("Ensure you have sufficient disk space and memory.")
        sys.exit(1)

def launch_server(model_path: str, args):
    """
    Launches the mlx_lm.server module.
    """
    logger.info("-" * 50)
    logger.info("PhaseGPT Oracle Serving Initialization")
    logger.info("-" * 50)
    logger.info(f"Model Source: {model_path}")
    logger.info(f"Network: http://{args.host}:{args.port}")
    
    cmd = [
        sys.executable, "-m", "mlx_lm.server",
        "--model", model_path,
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    if args.adapter:
        logger.info(f"Applying LoRA Adapter: {args.adapter}")
        cmd.extend(["--adapter-file", args.adapter])

    try:
        # Set environment variables for better performance/logging if needed
        env = os.environ.copy()
        
        # Execute the server
        subprocess.run(cmd, check=True, env=env)
        
    except KeyboardInterrupt:
        logger.info("\nServer shutdown requested by user.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Server process crashed with exit code {e.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Serve PhaseGPT Oracle on Apple Silicon (MLX Integration)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core Model Arguments
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Hugging Face Repo ID (e.g., 'Qwen/Qwen2.5-1.5B-Instruct') or local directory path."
    )
    
    # Quantization Controls
    parser.add_argument(
        "--no-quantize", 
        action="store_true", 
        help="Disable quantization. WARNING: Requires massive memory."
    )
    parser.add_argument(
        "--bits", 
        type=int, 
        default=4, 
        choices=[4, 8], 
        help="Bit-depth for quantization (4 or 8)."
    )
    parser.add_argument(
        "--group-size", 
        type=int, 
        default=64, 
        help="Group size for quantization parameter scaling."
    )

    # Server Configuration
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1", 
        help="Host interface to bind."
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080, 
        help="Listening port for the API."
    )
    
    # Advanced Options
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to a fine-tuned LoRA adapter (.npz) to load."
    )

    args = parser.parse_args()

    # Step 1: Check Dependencies
    check_environment()

    # Step 2: Resolve Paths
    # Source path is simply args.model
    # Destination path is calculated for conversion
    dest_path = get_model_path(args)
    
    # Step 3: Conversion Logic
    # If the destination already exists and is valid, skip conversion
    if validate_local_model(Path(dest_path)):
        logger.info(f"Optimized model found at {dest_path}. Skipping conversion.")
        final_model_path = dest_path
    else:
        # We need to convert from args.model to dest_path
        perform_conversion(args.model, dest_path, args)
        final_model_path = dest_path

    # Step 4: Launch Server
    launch_server(final_model_path, args)

if __name__ == "__main__":
    main()
