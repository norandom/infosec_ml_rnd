"""
https://www.because-security.com/blog/serverless-gpu-computing-with-modal-for-custom-models
"""


import modal
import os
import subprocess

# Define Modal Volume and mount path
MOUNT_PATH = "/data"
volume = modal.Volume.from_name("because-sec-lab-1", create_if_missing=True)

def setup_environment():
    # Clone repositories
    subprocess.run(['git', 'clone', 'https://github.com/norandom/infosec_ml_rnd.git'], check=True)
    subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/blt.git', 'infosec_ml_rnd/bpe_vs_blt_log_vec/blt'], check=True)
    
    # Add to Python path
    import sys
    sys.path.append('/root/infosec_ml_rnd')
    sys.path.append('/root/infosec_ml_rnd/bpe_vs_blt_log_vec')
    
    # Download tokenizer using existing function
    # from blt_wrapper.blt_wrapper import dl_blt_tokenizer
    # print("Token: " + os.environ["HF_TOKEN"])
    # dl_blt_tokenizer(os.environ["HF_TOKEN"])

# Create secret from name
secret = modal.Secret.from_name("hf-token")

# Define Modal Image with complete conda environment
image = (modal.Image.debian_slim()
    .apt_install([
        "git",
        "libsnappy-dev",  # for snappy
        "zstd",           # for zstd
        "liblz4-dev",     # for lz4
    ])
    .pip_install([
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
        "requests",  # Added for downloading files
        
        # Data Processing
        "pyarrow==16.1.0",
        "duckdb==1.1.3",
        "xxhash>=3.4.1",  # Updated to latest compatible version
        
        # Utilities
        "tqdm==4.66.5",
        "pyyaml==6.0.2",
        "typing-extensions==4.12.2",
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
        "protobuf==5.29.2",
        "huggingface_hub"  # Added for downloading tokenizer
    ])
    .run_function(setup_environment, secrets=[secret])
)

# Create the Modal app
app = modal.App("blt-env-app")

# Test CUDA availability
@app.function(gpu="any", image=image)  # Request any available GPU
def test_cuda():
    import torch
    return f"CUDA available: {torch.cuda.is_available()}"

# Function to test environment setup
@app.function(gpu="any", image=image, volumes={MOUNT_PATH: volume})
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
    import json
    
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
    
    # Write versions to file in volume
    with open(f"{MOUNT_PATH}/environment_versions.json", "w") as f:
        json.dump(versions, f, indent=2)
    
    volume.commit()  # Commit changes to volume
    return versions

# Function to process embeddings with BLT
@app.function(
    gpu="A10G",
    image=image,
    volumes={MOUNT_PATH: volume},
    secrets=[secret],
    timeout=3600  # 1 hour timeout
)
def process_blt_embeddings():
    import sys
    import duckdb
    import requests
    from pathlib import Path
    import numpy as np
    import time
    
    # Add paths
    sys.path.append('/root/infosec_ml_rnd')
    sys.path.append('/root/infosec_ml_rnd/bpe_vs_blt_log_vec')
    
    from blt_wrapper.blt_wrapper import create_blt_model, get_blt_embedding, load_blt_lib
    
    # Initialize BLT library
    blt_dir = '/root/infosec_ml_rnd/bpe_vs_blt_log_vec/blt'
    load_blt_lib(blt_dir)

    # Download the database
    print("\033[92mDownloading database...\033[0m")  
    db_url = "https://data.because-security.com/datasets/embeddings_1k.duckdb"
    db_path = f"{MOUNT_PATH}/embeddings_1k.duckdb"
    
    response = requests.get(db_url)
    with open(db_path, "wb") as f:
        f.write(response.content)
    
    # Initialize BLT model
    print("\033[92mInitializing BLT model...\033[0m")
    
    # Print GPU and CPU info
    import subprocess
    try:
        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=gpu_name --format=csv,noheader", shell=True).decode()
        print(f"GPU: {gpu_info.strip()}")
    except:
        print("No GPU info available")
        
    try:
        cpu_info = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | uniq", shell=True).decode()
        print(f"CPU: {cpu_info.split(':')[1].strip()}")
    except:
        print("No CPU info available")
    
    model = create_blt_model()
    tokenizer_path = "/root/infosec_ml_rnd/bpe_vs_blt_log_vec/tokenizers/tokenizer.model"
    
    # Connect to database
    conn = duckdb.connect(db_path)
    
    # Create embedding column if it doesn't exist
    conn.execute("""
    ALTER TABLE message_embeddings 
    ADD COLUMN IF NOT EXISTS embedding_blt DOUBLE[];
    """)
    
    # Get total count
    total_count = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0]
    
    # Process in batches
    batch_size = 100
    offset = 0
    processed = 0
    shape_printed = False
    time_printed = False
    start_time = None
    db_start_time = None
    
    while offset < total_count:
        # Get batch of messages
        messages = conn.execute("""
            SELECT message 
            FROM message_embeddings 
            ORDER BY message 
            LIMIT ? OFFSET ?
        """, [batch_size, offset]).fetchall()
        
        if not messages:
            break
            
        # Start timing for first batch
        if processed == 0:
            start_time = time.time()
            
        # Process batch
        new_embeddings = []
        for msg, in messages:
            embedding = get_blt_embedding(
                text=msg,
                tokenizer_path=tokenizer_path,
                model=model,
                verbose=False
            )
            # Print shape only once at the start
            if not shape_printed:
                print("\nFinal embeddings shape:", embedding['final'].shape)
                shape_printed = True
            
            # Convert to numpy array and get final token embedding
            embedding = embedding['final'].cpu().numpy()
            embedding = embedding[0, -1, :]  # Get last hidden state
            new_embeddings.append(embedding)

        # Print vectorization time for first batch
        if not time_printed and start_time is not None:
            vectorization_time = time.time() - start_time
            print(f"\nVectorization took: {vectorization_time:.2f} seconds for {len(messages)} messages")
            print(f"Average time per message: {vectorization_time/len(messages):.2f} seconds")
            db_start_time = time.time()
            time_printed = True

        # Update database
        update_query = """
        UPDATE message_embeddings 
        SET embedding_blt = ?
        WHERE message = ?
        """
        # Start transaction for batch update
        conn.execute("BEGIN TRANSACTION")
        for i, (msg,) in enumerate(messages):
            embedding_list = new_embeddings[i].tolist()
            conn.execute(update_query, [embedding_list, msg])
        conn.execute("COMMIT")
            
        # Print database update time for first batch
        if db_start_time is not None:
            db_time = time.time() - db_start_time
            print(f"Database update took: {db_time:.2f} seconds for {len(messages)} messages")
            print(f"Average time per update: {db_time/len(messages):.2f} seconds")
            db_start_time = None
        
        processed += len(messages)
        print(f"Progress: {(processed/total_count)*100:.2f}% ({processed}/{total_count})")
        offset += batch_size
    
    conn.close()
    volume.commit()
    return "Completed BLT embedding processing"
