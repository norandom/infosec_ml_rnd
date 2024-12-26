import torch
import duckdb
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import time
from data_manager.data_manager import load_filtered_messages
from blt_wrapper.blt_wrapper import (
    load_config, 
    load_blt_lib,
    create_blt_model,
    get_text_embedding,
    get_linformer_embedding
)

def get_config_path():
    """Get absolute path to config.json"""
    current_dir = Path(__file__).parent
    config_path = current_dir / 'config.json'
    return str(config_path)

def init_models_and_db():
    """Initialize models and database connection"""
    # Load configuration and BLT library
    config_path = get_config_path()
    hf_token, blt_dir = load_config(config_path)
    load_blt_lib(blt_dir)
    
    # Initialize BLT model
    model = create_blt_model(model_dim=1024, verbose=False)
    tokenizer_path = Path("tokenizers/tokenizer.model")
    
    # Initialize DuckDB
    con = duckdb.connect('embeddings_1k.duckdb')
    con.execute("""
        CREATE TABLE IF NOT EXISTS message_embeddings (
            message VARCHAR,
            embedding_blt DOUBLE[],
            embedding_bpe DOUBLE[]
        )
    """)
    
    return model, tokenizer_path, con

def process_batch(messages, model, tokenizer_path):
    """Process a batch of messages to create embeddings with performance metrics"""
    blt_embeddings = []
    bpe_embeddings = []
    blt_times = []
    bpe_times = []
    
    for msg in messages:
        # Get BLT embedding with timing
        blt_start = time.perf_counter()
        blt_emb = get_text_embedding(msg, model, tokenizer_path)
        blt_emb = blt_emb.detach().cpu().numpy()
        blt_end = time.perf_counter()
        blt_times.append(blt_end - blt_start)
        
        # Get Linformer embedding with timing
        bpe_start = time.perf_counter()
        bpe_emb = get_linformer_embedding(msg, tokenizer_path)
        bpe_emb = bpe_emb.detach().cpu().numpy()
        bpe_end = time.perf_counter()
        bpe_times.append(bpe_end - bpe_start)
        
        blt_embeddings.append(blt_emb)
        bpe_embeddings.append(bpe_emb)
        
        # Clear CUDA cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return blt_embeddings, bpe_embeddings, blt_times, bpe_times

def vectorize_messages(batch_size=50):
    """Vectorize messages in batches and store in DuckDB"""
    print("Initializing models and database...")
    model, tokenizer_path, con = init_models_and_db()
    
    # Load messages
    df = load_filtered_messages()
    messages = df['filtered_message'].tolist()
    total_messages = len(messages)
    
    print(f"Processing {total_messages} messages in batches of {batch_size}...")
    
    # Process in batches
    for i in tqdm(range(0, total_messages, batch_size)):
        batch = messages[i:i + batch_size]
        
        # Get embeddings for batch
        blt_embeddings, bpe_embeddings, blt_times, bpe_times = process_batch(batch, model, tokenizer_path)
        
        # Calculate performance metrics
        avg_blt_time = sum(blt_times) / len(blt_times)
        avg_bpe_time = sum(bpe_times) / len(bpe_times)
        total_blt_time = sum(blt_times)
        total_bpe_time = sum(bpe_times)
        
        print("\nPerformance Metrics:")
        print(f"BLT Vectorization:")
        print(f"  - Average time per message: {avg_blt_time:.4f} seconds")
        print(f"  - Total time for {len(batch)} messages: {total_blt_time:.4f} seconds")
        print(f"  - Messages per second: {len(batch)/total_blt_time:.2f}")
        
        print(f"\nLinformer (BPE) Vectorization:")
        print(f"  - Average time per message: {avg_bpe_time:.4f} seconds")
        print(f"  - Total time for {len(batch)} messages: {total_bpe_time:.4f} seconds")
        print(f"  - Messages per second: {len(batch)/total_bpe_time:.2f}")
        
        # Insert into DuckDB
        for msg, blt_emb, bpe_emb in zip(batch, blt_embeddings, bpe_embeddings):
            con.execute("""
                INSERT INTO message_embeddings (message, embedding_blt, embedding_bpe)
                VALUES (?, ?, ?)
            """, [msg, blt_emb.tolist(), bpe_emb.tolist()])
    
    print("Creating indexes...")
    con.execute("CREATE INDEX IF NOT EXISTS msg_idx ON message_embeddings (message)")
    
    # Close connection
    con.close()
    print("Processing complete!")

def test_vectorization():
    """Run a test vectorization on 10 messages with performance metrics"""
    print("Running test vectorization with 10 messages...")
    model, tokenizer_path, con = init_models_and_db()
    
    # Load messages and take first 10
    df = load_filtered_messages()
    messages = df['filtered_message'].head(10).tolist()
    
    # Process messages
    blt_embeddings, bpe_embeddings, blt_times, bpe_times = process_batch(messages, model, tokenizer_path)
    
    # Calculate performance metrics
    avg_blt_time = sum(blt_times) / len(blt_times)
    avg_bpe_time = sum(bpe_times) / len(bpe_times)
    total_blt_time = sum(blt_times)
    total_bpe_time = sum(bpe_times)
    
    print("\nPerformance Metrics:")
    print(f"BLT Vectorization:")
    print(f"  - Average time per message: {avg_blt_time:.4f} seconds")
    print(f"  - Total time for {len(messages)} messages: {total_blt_time:.4f} seconds")
    print(f"  - Messages per second: {len(messages)/total_blt_time:.2f}")
    
    print(f"\nLinformer (BPE) Vectorization:")
    print(f"  - Average time per message: {avg_bpe_time:.4f} seconds")
    print(f"  - Total time for {len(messages)} messages: {total_bpe_time:.4f} seconds")
    print(f"  - Messages per second: {len(messages)/total_bpe_time:.2f}")
    
    # Insert into DuckDB
    for msg, blt_emb, bpe_emb in zip(messages, blt_embeddings, bpe_embeddings):
        con.execute("""
            INSERT INTO message_embeddings (message, embedding_blt, embedding_bpe)
            VALUES (?, ?, ?)
        """, [msg, blt_emb.tolist(), bpe_emb.tolist()])
    
    # Print sample results
    print("\nSample from database:")
    result = con.execute("""
        SELECT 
            message,
            array_length(embedding_blt) as blt_dim,
            array_length(embedding_bpe) as bpe_dim
        FROM message_embeddings 
        LIMIT 3
    """).fetchall()
    
    for row in result:
        print(f"\nMessage: {row[0][:100]}...")
        print(f"BLT embedding dimensions: {row[1]}")
        print(f"BPE embedding dimensions: {row[2]}")
    
    con.close()
    print("\nTest complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run test mode with 10 messages')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for full processing')
    args = parser.parse_args()
    
    if args.test:
        test_vectorization()
    else:
        vectorize_messages(batch_size=args.batch_size)