import modal

# Define Modal Image with complete conda environment
image = modal.Image.debian_slim().pip_install([
    # Core ML and Data Science
    "torch",  # Will get CUDA-enabled version automatically
    "numpy==2.1.3",
    "pandas==2.2.3",
    "scikit-learn==1.5.2",
    "scipy==1.14.1",
    "networkx==3.2.1",
    "numexpr==2.10.1",
    
    # Deep Learning and NLP
    "transformers==4.45.2",
    "tokenizers==0.20.4",
    "sentencepiece==0.2.0",
    "safetensors==0.4.5",
    "regex==2024.9.11",
    
    # Data Processing
    "pyarrow==16.1.0",
    "duckdb==1.1.3",
    "xxhash>=3.4.1",  # Updated to latest compatible version
    
    # Utilities
    "tqdm==4.66.5",
    "pyyaml==6.0.2",
    "typing-extensions==4.12.2",
    "requests==2.32.3",
    "aiohttp==3.11.10",
    "urllib3==2.2.3",
    "pydantic==2.10.4",
    "pydantic-core==2.27.2",
    
    # Image Processing
    "pillow==11.0.0",
    "torchvision",  # Will get CUDA-enabled version automatically
    "torchaudio",   # Will get CUDA-enabled version automatically
    
    # Development Tools
    "psutil==5.9.0",
    "packaging==24.1",
    "setuptools",  # No version constraint
    
    # Math and Stats
    "mpmath==1.3.0",
    
    # Additional Dependencies
    "openai==1.52.1",
    "multiprocess==0.70.15",
    "propcache==0.2.0",
    "python-snappy",  # Changed from snappy to python-snappy
    
    # Core Dependencies
    "modal==0.70.1",
    "fastapi==0.115.6",
    "fsspec==2024.2.0",
    "grpclib==0.4.7",
    
    # ML and Deep Learning
    "linformer-pytorch==0.19.3",
    "xformers==0.0.28.post3",
    
    # Development Tools
    "rich==13.9.4",
    "typer==0.15.1",
    "watchfiles==1.0.3",
    "shellingham==1.5.4",
    "sympy==1.13.1",
    "markdown-it-py==3.0.0",
    "mdurl==0.1.2",
    
    # HTTP and Networking
    "h2==4.1.0",
    "hpack==4.0.0",
    "hyperframe==6.0.1",
    
    # Type Checking
    "types-toml==0.10.8.20240310",
    "types-certifi==2021.10.8.3",
    
    # Additional Utilities
    "sigtools==4.0.1",
    "starlette==0.41.3",
    "synchronicity==0.9.8",
    "toml==0.10.2",
    "protobuf==5.29.2"
]).apt_install([
    "git",
    "libsnappy-dev",  # for snappy
    "zstd",           # for zstd
    "liblz4-dev",     # for lz4
])

# Create the Modal app
app = modal.App("blt-env-app")

# Test CUDA availability
@app.function(gpu="any", image=image)  # Request any available GPU
def test_cuda():
    import torch
    return f"CUDA available: {torch.cuda.is_available()}"

# Function to test environment setup
@app.function(gpu="any", image=image)  # Request any available GPU
def check_environment():
    import sys
    import torch
    import transformers
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import duckdb
    import openai
    from PIL import Image
    
    versions = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pyarrow_version": pa.__version__,
        "duckdb_version": duckdb.__version__,
        "openai_version": openai.__version__,
        "pillow_version": Image.__version__
    }
    print("Environment versions:")
    for k, v in versions.items():
        print(f"{k}: {v}")
    return versions
