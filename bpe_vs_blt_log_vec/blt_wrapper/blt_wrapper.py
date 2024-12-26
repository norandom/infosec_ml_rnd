def load_config(config_path='../config.json'):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the config.json file. Defaults to '../config.json'
        
    Returns:
        tuple: (huggingface_token, blt_dir) from config
    """
    import json
    from pathlib import Path

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
        hf_token = config.get("huggingface_token")  # Load HF token from config
        blt_dir = config.get("blt_dir")
        
        if not hf_token:
            raise ValueError("huggingface_token not found in config")
        if not blt_dir:
            raise ValueError("blt_dir not found in config")

    return hf_token, blt_dir


def dl_blt_tokenizer(hf_token : str):

    from huggingface_hub import hf_hub_download

    # Llama2 tokenizer details from the setup script
    repo_id = "meta-llama/Llama-2-7b"
    filename = "tokenizer.model"
    path_to_save = "./tokenizers"  # You can change this path as needed

    # Download the tokenizer
    output_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=path_to_save,
        token=hf_token
    )

    print(f"Successfully downloaded tokenizer to: {output_path}")


def load_blt_lib(blt_dir : str):
    import sys
    import os
    from pathlib import Path

    blt_dir = Path(os.path.expanduser(blt_dir))
    print("Loading from blt_dir: " + str(blt_dir))
    # Add the BLT directory to Python path
    sys.path.append(str(blt_dir))

    # Now you can import bytelatent
    from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
    from bytelatent.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer

    # Test the imports
    print("BLT imports successful!")


def init_and_test_tokenizers():
    from pathlib import Path
    from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
    from bytelatent.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer

    # Initialize tokenizers
    tokenizer_path = Path("tokenizers/tokenizer.model")  # Path where we downloaded the tokenizer

    try:
        bpe_tokenizer = SentencePieceTokenizer(model_path=str(tokenizer_path))
        blt_tokenizer = BltTokenizer(
            bpe_delim=True,
            bpe_tokenizer_path=str(tokenizer_path),
            add_bos=True,
            add_eos=True
        )
        print("Tokenizers initialized successfully!")
        
        # Test with a simple string
        test_text = "Hello, this is a test."
        bpe_tokens = bpe_tokenizer.encode(test_text)
        blt_tokens = blt_tokenizer.encode(test_text)
        
        print(f"\nTest results:")
        print(f"Original text: {test_text}")
        print(f"BPE tokens: {bpe_tokens}")
        print(f"BLT tokens: {blt_tokens}")
        
    except Exception as e:
        print(f"Error initializing tokenizers: {e}")

    print("Sentence length:    %d" % len(test_text))
    print("BPE tokens created: %d" % len(bpe_tokens))
    print("BLT tokens created: %d" % len(blt_tokens))


def init_and_test_byte_patcher(verbose=True):
    from bytelatent.data.patcher import Patcher, PatcherArgs
    from bytelatent.model.local_models import LocalEncoder
    import numpy as np
    import torch

    # Initialize patcher with entropy-based patching
    patcher_args = PatcherArgs(
        patching_mode="entropy",  # Uses entropy for dynamic patching
        threshold=1.335442066192627,  # Default entropy threshold
        patch_size=4.5,  # Average target patch size
        patching_batch_size=1,
        device="cpu"  # Using CPU for testing
    )
    patcher = Patcher(patcher_args)
    if verbose:
        print("\nByte Patcher initialized successfully!")

    # Test with a sample binary data
    test_data = b"Hello World! This is some test binary data with varying entropy levels."
    # Convert bytes to numpy array then to tensor, ensuring values are in range [0, 259]
    data_array = np.frombuffer(test_data, dtype=np.uint8).copy()  # Make array writable
    data_tensor = torch.from_numpy(data_array).unsqueeze(0)  # Add batch dimension
    
    try:
        # Create patches using static patch size since entropy-based needs more setup
        patch_size = 8  # Static patch size for testing
        num_patches = (len(test_data) + patch_size - 1) // patch_size
        patch_lengths = torch.full((1, num_patches), patch_size, dtype=torch.long)
        # Adjust the last patch length if needed
        if len(test_data) % patch_size != 0:
            patch_lengths[0, -1] = len(test_data) % patch_size
            
        if verbose:
            print(f"\nByte Patcher test results:")
            print(f"Original data length: {len(test_data)} bytes")
            print(f"Batch shape: {data_tensor.shape}")
            print(f"Number of patches: {num_patches}")
        
            # Print patch information
            cumsum = 0
            for i, length in enumerate(patch_lengths[0]):
                length = length.item()
                if length > 0:  # Skip zero-length patches
                    patch_data = test_data[cumsum:cumsum + length]
                    print(f"Patch {i}: Length={length}, Data={patch_data}")
                    cumsum += length
        
        return patcher, patch_lengths
        
    except Exception as e:
        if verbose:
            print(f"Error during patching: {e}")
            import traceback
            traceback.print_exc()
        raise


def init_and_test_bpe_patcher(verbose=True):
    from bytelatent.data.patcher import Patcher, PatcherArgs
    import torch

    # Initialize patcher with BPE patching
    patcher_args = PatcherArgs(
        patching_mode="bpe",  # Use BPE-based patching
        patch_size=4.5,  # Average target patch size
        patching_batch_size=1,
        device="cpu"  # Using CPU for testing
    )
    patcher = Patcher(patcher_args)
    if verbose:
        print("\nBPE Patcher initialized successfully!")

    # Test with a sample text
    test_text = "Hello, this is a test for BPE patching!"
    
    try:
        # First get BLT tokens using the previously initialized tokenizer
        from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
        from pathlib import Path
        
        tokenizer_path = Path("tokenizers/tokenizer.model")
        blt_tokenizer = BltTokenizer(
            bpe_delim=True,
            bpe_tokenizer_path=str(tokenizer_path),
            add_bos=True,
            add_eos=True
        )
        
        # Get BLT tokens
        blt_tokens = blt_tokenizer.encode(test_text)
        
        # Convert tokens to tensor and add batch dimension
        tokens_tensor = torch.tensor(blt_tokens).unsqueeze(0)

        # Apply patcher
        patch_lengths = patcher.patch(
            tokens=tokens_tensor,
            include_next_token=False
        )[0]  # Get first batch

        if verbose:
            print(f"\nBPE Patcher test results:")
            print(f"Original text: {test_text}")
            print(f"BLT tokens: {blt_tokens}")
            print(f"Patch lengths: {patch_lengths.tolist()}")
            
            # Print reconstructed patches
            cumsum = 0
            patch_lengths_list = patch_lengths[0].tolist()  # Get the first batch's patch lengths
            for i, length in enumerate(patch_lengths_list):
                if length > 0:  # Skip zero-length patches
                    patch_tokens = blt_tokens[cumsum:cumsum + length]
                    print(f"Patch {i}: Length={length}, Tokens={patch_tokens}")
                    cumsum += length
        
        return patcher, blt_tokenizer, patch_lengths, blt_tokens
                
    except Exception as e:
        if verbose:
            print(f"Error during BPE patching: {e}")
            import traceback
            traceback.print_exc()
        raise


def create_blt_model(checkpoint_path=None, model_dim=1024, verbose=True):
    """
    Create a BLT model with the specified configuration.

    Args:
        checkpoint_path (str, optional): Path to a model checkpoint to load.
        model_dim (int): Model dimension. Default is 1024.
        verbose (bool): Whether to print model details.

    Returns:
        ByteLatentTransformer: The initialized model.
    """
    from torch import nn
    from bytelatent.model.utils import downsample
    import torch
    from bytelatent.model.blt import ByteLatentTransformer, ByteLatentTransformerArgs

    # Calculate dependent parameters
    n_heads = model_dim // 64  # Keep head_dim=64 constant
    if verbose:
        print(f"Initializing model with:")
        print(f"- Model dimension: {model_dim}")
        print(f"- Number of heads: {n_heads}")
    
    vocab_size = model_dim  # Set vocab_size to model_dim to get desired output dimension
    max_seqlen = 10_000

    args = ByteLatentTransformerArgs(
        dim=model_dim,  # Model's internal dimension
        n_layers=12,
        n_heads=n_heads,  # Scales with dim to maintain head_dim=64
        vocab_size=vocab_size,  # Set vocab_size to model_dim to get desired output dimension
        patch_size=4,
        patching_mode="bpe",
        downsampling_by_pooling="avg",
        max_seqlen=max_seqlen,
        max_length=max_seqlen,
        max_encoder_seq_length=max_seqlen,
        weight_tying=True,
        sliding_window=None,
        cross_attn_encoder=False,
        cross_attn_decoder=False,
        cross_attn_k=4,
        cross_attn_window_encoder=None,
        cross_attn_window_decoder=None,
        cross_attn_use_flex_attention=False,
        encoder_hash_byte_group_size=[4],
        encoder_hash_byte_group_vocab=256,
        encoder_hash_byte_group_nb_functions=4,
        encoder_enable_byte_ngrams=False,
        data_loader_patching=False,
        patching_threshold=0.5,
        patching_threshold_add=0.0,
        monotonicity=False,
        max_patch_length=None,
        init_std_factor="disabled",
        init_base_std=0.02,
        head_dim=64,  # Keep constant
        rope_theta=10000.0,
        use_rope=True,
        dropout=0.0,
        norm_eps=1e-5,
        pm_size=0,
        efficient_attn="fmha",
        use_local_encoder_transformer=True,
        patch_only_encoder=False,
        patch_only_decoder=False,
        share_encoder_decoder_emb=False,
        cross_attn_nheads=n_heads,  # Scales with n_heads
        cross_attn_all_layers_encoder=False,
        cross_attn_all_layers_decoder=False,
        cross_attn_init_by_pooling=False,
        entropy_model_checkpoint_dir=None
    )

    try:
        # Create model
        if verbose:
            print("Creating model...")
        
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
            
        model = ByteLatentTransformer(args)
        
        # Add output projection to get desired dimension
        model.output_proj = nn.Linear(vocab_size, model_dim).to(device=device, dtype=dtype)
        
        model = model.to(device=device, dtype=dtype)
        model.device = device  # Store device for later use
        
        if verbose:
            print(f"Model initialized successfully on {device}")
            
        return model
        
    except Exception as e:
        if verbose:
            print(f"Error creating model: {e}")
            import traceback
            traceback.print_exc()
        raise


def init_and_test_embeddings(verbose=True):
    from torch import nn
    from bytelatent.model.utils import downsample
    import torch
    from bytelatent.model.blt import ByteLatentTransformer, ByteLatentTransformerArgs

    try:
        model = create_blt_model(model_dim=1024, verbose=verbose)
        return model
    except Exception as e:
        if verbose:
            print(f"Error in init_and_test_embeddings: {e}")
        raise


def process_text_to_embeddings(text, tokenizer_path, model, verbose=False):
    """
    Process text through the BLT model to generate embeddings.
    
    Args:
        text (str): Input text to process
        tokenizer_path (Path): Path to the tokenizer model
        model: Initialized BLT model
        verbose (bool): Whether to print processing details
    
    Returns:
        dict: Dictionary of different embedding types
    """
    from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
    import torch
    
    # Initialize tokenizer
    blt_tokenizer = BltTokenizer(
        bpe_delim=True,
        bpe_tokenizer_path=str(tokenizer_path),
        add_bos=True,
        add_eos=True
    )
    
    # Tokenize text
    tokens = blt_tokenizer.encode(text)
    if verbose:
        print(f"Tokens: {tokens}")
    
    # Convert to tensor and add batch dimension
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)
    device = getattr(model, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    tokens_tensor = tokens_tensor.to(device)
    
    # Generate embeddings
    with torch.no_grad():
        # Get token embeddings from the model's forward pass
        embeddings = model(tokens_tensor)
        
        # Extract the hidden states
        if hasattr(embeddings, 'hidden_states'):
            token_embeddings = embeddings.hidden_states[-1]  # Last layer
        else:
            token_embeddings = embeddings  # Assume it's the final embeddings
            
        # For position and final embeddings, use the same since we don't have access to intermediates
        position_embeddings = token_embeddings.clone()  # Clone to avoid modification
        final_embeddings = token_embeddings
    
    return {
        'token': token_embeddings,
        'position': position_embeddings,
        'final': final_embeddings
    }


def get_model_hidden_states(text, model, tokenizer_path, verbose=True):
    """
    Get and analyze the model's hidden states for given text.
    
    Args:
        text (str): Input text to process
        model: BLT model instance
        tokenizer_path (Path): Path to tokenizer
        verbose (bool): Whether to print analysis details
    
    Returns:
        torch.Tensor: Last hidden state with shape [batch_size, seq_len, hidden_dim]
    """
    from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
    import torch
    
    if verbose:
        print("\n=== Analyzing Model Hidden States ===")
        print(f"Input text: '{text}'")
    
    # Initialize tokenizer and process text
    blt_tokenizer = BltTokenizer(
        bpe_delim=True,
        bpe_tokenizer_path=str(tokenizer_path),
        add_bos=True,
        add_eos=True
    )
    tokens = blt_tokenizer.encode(text)
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)
    device = getattr(model, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokens_tensor = tokens_tensor.to(device=device)
    
    # Get model outputs
    with torch.no_grad():
        # Forward pass through model
        base_output = model(tokens_tensor)
        outputs = model.output_proj(base_output.to(dtype))  # Project to desired dimension with correct dtype
        
        if verbose:
            print(f"\nBase output shape: {base_output.shape}")
            print(f"Final output shape: {outputs.shape}")
            print(f"• Batch size: {outputs.shape[0]}")
            print(f"• Sequence length: {outputs.shape[1]}")
            print(f"• Hidden dimension: {outputs.shape[2]}")
            print(f"• Device: {outputs.device}")
            print(f"• Dtype: {outputs.dtype}")
            
            print("\nStatistics:")
            print(f"• Mean: {outputs.float().mean().item():.3f}")
            print(f"• Std: {outputs.float().std().item():.3f}")
            print(f"• Min: {outputs.float().min().item():.3f}")
            print(f"• Max: {outputs.float().max().item():.3f}")
            
            # Distribution analysis
            print("\nValue Distribution:")
            percentiles = [0, 25, 50, 75, 100]
            for p in percentiles:
                val = torch.quantile(outputs.float(), p/100)
                print(f"• {p}th percentile: {val.item():.3f}")
        
    return outputs  # Return projected hidden states


def get_text_embedding(text, model, tokenizer_path):
    """
    Get a 1024-dimensional embedding vector for the input text.
    
    Args:
        text (str): Input text to vectorize
        model: Initialized BLT model
        tokenizer_path (Path): Path to the tokenizer model
    
    Returns:
        torch.Tensor: 1024-dimensional embedding vector
    """
    # Get hidden states (last hidden state)
    hidden_states = get_model_hidden_states(text, model, tokenizer_path, verbose=False)
    
    # Average across sequence dimension to get a single vector
    embedding = hidden_states.mean(dim=1)
    
    return embedding.squeeze()


def get_linformer_embedding(text, tokenizer_path):
    """
    Get a 1024-dimensional embedding vector using Linformer with BPE tokenization.
    
    Args:
        text (str): Input text to vectorize
        tokenizer_path (Path): Path to the tokenizer model
    
    Returns:
        torch.Tensor: 1024-dimensional embedding vector
    """
    from bytelatent.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer
    from linformer import Linformer
    import torch
    import torch.nn as nn
    
    # Initialize BPE tokenizer
    bpe_tokenizer = SentencePieceTokenizer(model_path=str(tokenizer_path))
    tokens = bpe_tokenizer.encode(text)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    # Create tensors on the correct device
    tokens_tensor = torch.tensor(tokens, device=device).unsqueeze(0)
    
    # Initialize Linformer model
    config = {
        'input_size': 10_000,  # Dynamic sequence length
        'channels': 1024,  # Match BLT dimension
        'dim_k': 128,     # Reduced attention dimension
        'dim_ff': 2048,   # Feed-forward dimension
        'nhead': 16,      # Match BLT heads
        'depth': 12,      # Match BLT layers
        'dropout': 0.1
    }
    
    # Create Linformer model and move to device
    model = Linformer(
        dim=config['channels'],
        seq_len=config['input_size'],
        depth=config['depth'],
        heads=config['nhead'],
        k=config['dim_k'],
        one_kv_head=False,  # Enable multi-head attention
        share_kv=False      # Don't share key/value projections
    ).to(device=device, dtype=dtype)
    
    # Create input embeddings on device
    embedding = nn.Embedding(32000, config['channels']).to(device=device, dtype=dtype)
    embedded = embedding(tokens_tensor)
    
    # Get model output
    with torch.no_grad():
        output = model(embedded)
        # Average across sequence dimension
        embedding = output.mean(dim=1)
    
    return embedding.squeeze()


def make_blt_embeddings(verbose=False):
    """
    Test function to create and analyze BLT embeddings.
    
    Args:
        verbose (bool): Whether to print detailed information during processing.
        
    Returns:
        dict: Dictionary containing token, position and final embeddings
    """
    import torch
    from pathlib import Path
    
    if verbose:
        print("\nInitializing BLT embedding generation...")
        
    # Initialize model and tokenizer
    tokenizer_path = Path("tokenizers/tokenizer.model")
    if verbose:
        print(f"Using tokenizer from: {tokenizer_path}")
        
    model = create_blt_model()
    
    # Test text
    test_text = "Hello, this is a test."
    if verbose:
        print(f"\nProcessing text: '{test_text}'")
    
    # Get embeddings
    embeddings = process_text_to_embeddings(
        text=test_text,
        tokenizer_path=tokenizer_path,
        model=model,
        verbose=verbose
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
        
        if verbose:
            print("\nTensor Properties:")
            print(f"• Device: {embedding.device}")
            print(f"• Dtype: {embedding.dtype}")
        
        print("\nStatistics:")
        print(f"• Mean: {embedding.float().mean().item():.3f}")
        print(f"• Std: {embedding.float().std().item():.3f}")
        print(f"• Min: {embedding.float().min().item():.3f}")
        print(f"• Max: {embedding.float().max().item():.3f}")
        
        if verbose:
            print("\nValue Distribution:")
            percentiles = [0, 25, 50, 75, 100]
            for p in percentiles:
                val = torch.quantile(embedding.float(), p/100)
                print(f"• {p}th percentile: {val.item():.3f}")
    
    print("\n=== Overall Statistics ===")
    total_params = sum(e.shape[1] * e.shape[2] for e in embeddings.values())
    print(f"Total parameters across all embeddings: {total_params:,}")
    
    if verbose:
        print("\nMemory Usage:")
        memory_bytes = sum(e.element_size() * e.nelement() for e in embeddings.values())
        print(f"• Total memory: {memory_bytes/1024/1024:.2f} MB")
        
    return embeddings


def main(config_path='../config.json'):
    """
    Main function to test the BLT wrapper functionality.
    
    Args:
        config_path (str): Path to the config.json file. Defaults to '../config.json'
    """
    from pathlib import Path  # Add this import
    import torch
    
    # Load configuration
    hf_token, blt_dir = load_config(config_path)
    
    # Initialize libraries and tokenizer
    dl_blt_tokenizer(hf_token)
    load_blt_lib(blt_dir)
    init_and_test_tokenizers()
    init_and_test_byte_patcher()
    init_and_test_bpe_patcher()
    
    # Create model and test embeddings
    tokenizer_path = Path("tokenizers/tokenizer.model")
    model = create_blt_model()
    test_text = "Hello, this is a test."
    
    # Get embeddings from both methods
    print("\n=== Comparing Embeddings ===")
    print(f"Input text: '{test_text}'")
    
    # BLT Embeddings
    print("\n1. BLT Embeddings:")
    blt_embedding = get_text_embedding(test_text, model, tokenizer_path)
    print(f"Shape: {blt_embedding.shape}")
    print(f"Mean: {blt_embedding.mean().item():.4f}")
    print(f"Std: {blt_embedding.std().item():.4f}")
    
    # Linformer Embeddings
    print("\n2. Linformer Embeddings:")
    linformer_embedding = get_linformer_embedding(test_text, tokenizer_path)
    print(f"Shape: {linformer_embedding.shape}")
    print(f"Mean: {linformer_embedding.mean().item():.4f}")
    print(f"Std: {linformer_embedding.std().item():.4f}")
    
    # Compute cosine similarity
    cos = torch.nn.CosineSimilarity(dim=0)
    similarity = cos(blt_embedding, linformer_embedding)
    print(f"\nCosine Similarity between embeddings: {similarity.item():.4f}")

if __name__ == "__main__":
    import sys
    
    # Get config path from command line args if provided
    config_path = sys.argv[1] if len(sys.argv) > 1 else '../config.json'
    main(config_path)