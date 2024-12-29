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
        "huggingface_hub",  # Added for downloading tokenizer
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
    gpu="any", 
    image=image, 
    volumes={MOUNT_PATH: volume},
    secrets=[secret]
)
def process_blt_embeddings():
    import sys
    import duckdb
    import requests
    from pathlib import Path
    import numpy as np
    from tqdm import tqdm
    
    # Add paths
    sys.path.append('/root/infosec_ml_rnd')
    sys.path.append('/root/infosec_ml_rnd/bpe_vs_blt_log_vec')
    
    from blt_wrapper.blt_wrapper import create_blt_model, get_text_embedding, load_blt_lib
    
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
    model = create_blt_model()
    tokenizer_path = "/root/infosec_ml_rnd/bpe_vs_blt_log_vec/tokenizers/original"
    
    # Connect to database
    conn = duckdb.connect(db_path)
    
    # Get total count for progress bar
    total_count = conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone()[0]
    
    # Process in batches
    batch_size = 100
    offset = 0
    
    with tqdm(total=total_count, desc="Processing embeddings", colour='green') as pbar:
        while True:
            # Get batch of messages
            query = f"""
            SELECT message 
            FROM message_embeddings 
            LIMIT {batch_size} 
            OFFSET {offset}
            """
            messages = conn.execute(query).fetchall()
            
            if not messages:
                break
                
            # Process batch
            new_embeddings = []
            for msg, in messages:
                embedding = get_text_embedding(msg, model, tokenizer_path)
                # Convert to numpy array
                embedding = embedding.cpu().numpy().squeeze()
                new_embeddings.append(embedding)
            
            # Update database
            for i, (msg,) in enumerate(messages):
                update_query = """
                UPDATE message_embeddings 
                SET embedding_blt = ?
                WHERE message = ?
                """
                conn.execute(update_query, [new_embeddings[i], msg])
            
            offset += batch_size
            pbar.update(len(messages))
    
    conn.close()
    volume.commit()
    return "Completed BLT embedding processing"

# Function to test BLT embeddings
@app.function(gpu="any", image=image)
def test_blt_embeddings():
    import sys
    import torch
    from pathlib import Path
    
    # Add paths
    sys.path.append('/root/infosec_ml_rnd')
    sys.path.append('/root/infosec_ml_rnd/bpe_vs_blt_log_vec')
    
    from blt_wrapper.blt_wrapper import create_blt_model, process_text_to_embeddings, load_blt_lib
    
    # Initialize BLT library
    blt_dir = '/root/infosec_ml_rnd/bpe_vs_blt_log_vec/blt'
    load_blt_lib(blt_dir)
    
    # Initialize model and tokenizer
    model = create_blt_model()
    tokenizer_path = "/root/infosec_ml_rnd/bpe_vs_blt_log_vec/tokenizers/original"
    
    # Test text
    test_text = "Hello, this is a test."
    
    # Get embeddings
    embeddings = process_text_to_embeddings(
        text=test_text,
        tokenizer_path=tokenizer_path,
        model=model,
        verbose=False
    )
    
    # Print analysis
    print("\n=== BLT Embedding Analysis ===")
    print(f"\nInput Text: \"{test_text}\"")
    
    for name, embedding in embeddings.items():
        print(f"\n=== {name.upper()} Embeddings ===")
        print(f"Shape: {embedding.shape}")
        print(f"• Batch size: {embedding.shape[0]}")
        print(f"• Sequence length: {embedding.shape[1]}")
        print(f"• Embedding dimension: {embedding.shape[2]}")
        params = embedding.shape[1] * embedding.shape[2]
        print(f"• Parameters: {params:,}")
        total_params = sum(e.shape[1] * e.shape[2] for e in embeddings.values())
        print(f"• Share of total: {params/total_params:.2%}")
        
        print("\nStatistics:")
        print(f"• Mean: {embedding.float().mean().item():.3f}")
        print(f"• Std: {embedding.float().std().item():.3f}")
        print(f"• Min: {embedding.float().min().item():.3f}")
        print(f"• Max: {embedding.float().max().item():.3f}")
    
    print("\n=== Overall Statistics ===")
    total_params = sum(e.shape[1] * e.shape[2] for e in embeddings.values())
    print(f"Total parameters across all embeddings: {total_params:,}")
    
    return embeddings
