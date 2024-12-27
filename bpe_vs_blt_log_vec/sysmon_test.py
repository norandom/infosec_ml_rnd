import torch
import duckdb
import numpy as np
from pathlib import Path
import os
import time
from tqdm import tqdm
from data_manager.data_manager import load_filtered_messages
from blt_wrapper.blt_wrapper import (
    load_config, 
    load_blt_lib,
    create_blt_model,
    get_text_embedding,
    get_linformer_embedding
)
import openai
from time import sleep
import json

def get_config_path():
    """Get absolute path to config.json"""
    current_dir = Path(__file__).parent
    config_path = current_dir / 'config.json'
    return str(config_path)

def get_db_name(mode='full', message_count=None):
    """Get database name based on mode and message count"""
    if mode == 'test':
        return f'embeddings_test_{message_count}.duckdb'
    elif mode == 'test_similarity':
        return f'embeddings_test_sim_{message_count}.duckdb'
    else:
        return 'embeddings_1k.duckdb'

def init_models_and_db(mode='full', message_count=None):
    """Initialize models and database connection"""
    # Load configuration and BLT library
    config_path = get_config_path()
    hf_token, blt_dir, _ = load_config(config_path)
    load_blt_lib(blt_dir)
    
    # Initialize BLT model
    model = create_blt_model(model_dim=1024, verbose=False)
    tokenizer_path = Path("tokenizers/tokenizer.model")
    
    # Initialize DuckDB with mode-specific database
    con = duckdb.connect(get_db_name(mode, message_count), read_only=False)
    
    # Create table if not exists
    con.execute("""
        CREATE TABLE IF NOT EXISTS message_embeddings (
            message VARCHAR,
            embedding_blt DOUBLE[],
            embedding_bpe DOUBLE[]
        )
    """)
    
    return model, tokenizer_path, con

def cosine_similarity_sql():
    """Get SQL expression for cosine similarity between two vectors"""
    return """
        LIST_DOT_PRODUCT(v1, v2) /
        (
            SQRT(LIST_DOT_PRODUCT(v1, v1)) *
            SQRT(LIST_DOT_PRODUCT(v2, v2))
        )
    """

def find_similar_vectors(con, embeddings_blt, embeddings_bpe, similarity_threshold=0.99):
    """Find existing messages with similar vectors using cosine similarity on BPE embeddings only"""
    if not embeddings_bpe:
        return set()
    
    # Convert embeddings to string format for SQL
    vectors_bpe = [str(emb.tolist()) for emb in embeddings_bpe]
    
    # Create temporary table for the new vectors
    con.execute("CREATE TEMP TABLE IF NOT EXISTS temp_vectors_bpe (embedding DOUBLE[])")
    
    # Insert the new vectors
    for bpe in vectors_bpe:
        con.execute("INSERT INTO temp_vectors_bpe VALUES (?)", [bpe])
    
    # Perform similarity search using BPE embeddings
    similar_indices = con.execute(f"""
        WITH similarities AS (
            SELECT 
                t.rowid as new_idx,
                e.rowid as existing_idx,
                LIST_DOT_PRODUCT(t.embedding, e.embedding_bpe) /
                (
                    SQRT(LIST_DOT_PRODUCT(t.embedding, t.embedding)) *
                    SQRT(LIST_DOT_PRODUCT(e.embedding_bpe, e.embedding_bpe))
                ) as similarity
            FROM temp_vectors_bpe t
            CROSS JOIN message_embeddings e
        )
        SELECT DISTINCT new_idx
        FROM similarities
        WHERE similarity >= {similarity_threshold}
    """).fetchall()
    
    # Clean up temporary table
    con.execute("DROP TABLE IF EXISTS temp_vectors_bpe")
    
    return {row[0] for row in similar_indices}

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

def vectorize_messages(batch_size=50, skip_similar=True, similarity_threshold=0.99):
    """Vectorize messages in batches"""
    print("Initializing models and database...")
    model, tokenizer_path, con = init_models_and_db()
    
    # Load messages
    df = load_filtered_messages()
    messages = df['filtered_message'].tolist()
    total_messages = len(messages)
    
    # Get already processed messages
    existing_messages = set(row[0] for row in con.execute("SELECT message FROM message_embeddings").fetchall())
    messages_to_process = [msg for msg in messages if msg not in existing_messages]
    skipped_count = total_messages - len(messages_to_process)
    
    print(f"Found {len(messages_to_process)} messages to process (skipping {skipped_count} already processed)")
    print(f"Processing in batches of {batch_size}...")
    
    # Process in batches
    for i in tqdm(range(0, len(messages_to_process), batch_size)):
        batch = messages_to_process[i:i + batch_size]
        if not batch:
            break
            
        # Get embeddings for the batch
        blt_embeddings, bpe_embeddings, _, _ = process_batch(batch, model, tokenizer_path)
        
        # Insert into database
        for msg, blt_emb, bpe_emb in zip(batch, blt_embeddings, bpe_embeddings):
            con.execute("""
                INSERT INTO message_embeddings (message, embedding_blt, embedding_bpe)
                VALUES (?, ?, ?)
            """, [msg, blt_emb.tolist(), bpe_emb.tolist()])
    
    con.close()
    print("\nProcessing complete!")

def test_vectorization(message_count=10):
    """Run a basic test vectorization"""
    print(f"Running basic test with {message_count} messages...")
    model, tokenizer_path, con = init_models_and_db(mode='test', message_count=message_count)
    
    # Load messages
    df = load_filtered_messages()
    messages = df['filtered_message'].head(message_count).tolist()
    
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

def test_vectorization_similarity(message_count=10):
    """Run a test vectorization with similarity detection"""
    print(f"\nRunning similarity test with {message_count} messages...")
    model, tokenizer_path, con = init_models_and_db(mode='test_similarity', message_count=message_count)
    
    # Load messages
    df = load_filtered_messages()
    messages = df['filtered_message'].head(message_count).tolist()
    
    # Process all messages first to get embeddings
    print("Computing embeddings for all messages...")
    blt_embeddings, bpe_embeddings, blt_times, bpe_times = process_batch(messages, model, tokenizer_path)
    
    # Calculate initial stats
    avg_msg_length = sum(len(msg) for msg in messages) / len(messages)
    print(f"\nInitial Statistics:")
    print(f"Number of messages: {len(messages)}")
    print(f"Average message length: {avg_msg_length:.1f} characters")
    
    # Insert first message as baseline
    con.execute("""
        INSERT INTO message_embeddings (message, embedding_blt, embedding_bpe)
        VALUES (?, ?, ?)
    """, [messages[0], blt_embeddings[0].tolist(), bpe_embeddings[0].tolist()])
    
    # Process remaining messages with similarity detection
    unique_count = 1
    similar_groups = {0: [0]}  # Group similar messages by their first occurrence
    
    print("\nChecking for similar messages...")
    for i in tqdm(range(1, len(messages))):
        # Check similarity with existing messages
        similar_indices = find_similar_vectors(
            con, 
            [blt_embeddings[i]], 
            [bpe_embeddings[i]], 
            similarity_threshold=0.99  # High threshold for log messages
        )
        
        if not similar_indices:
            # No similar messages found, add to database
            con.execute("""
                INSERT INTO message_embeddings (message, embedding_blt, embedding_bpe)
                VALUES (?, ?, ?)
            """, [messages[i], blt_embeddings[i].tolist(), bpe_embeddings[i].tolist()])
            unique_count += 1
            similar_groups[i] = [i]
        else:
            # Add to existing group
            for existing_idx in similar_groups:
                if any(j in similar_indices for j in similar_groups[existing_idx]):
                    similar_groups[existing_idx].append(i)
                    break
    
    # Print results
    print("\nResults:")
    print(f"Total messages processed: {len(messages)}")
    print(f"Unique message patterns: {unique_count}")
    print(f"Reduction ratio: {(len(messages) - unique_count) / len(messages):.1%}")
    
    # Print some example groups
    print("\nExample similar message groups:")
    group_count = 0
    for base_idx, similar_indices in similar_groups.items():
        if len(similar_indices) > 1:  # Only show groups with similar messages
            group_count += 1
            if group_count > 3:  # Show only first 3 groups
                break
            print(f"\nGroup {group_count} (Base message + 2 similar):")
            print(f"Base: {messages[base_idx][:100]}...")
            for idx in similar_indices[1:3]:  # Show only 2 similar messages
                print(f"Similar: {messages[idx][:100]}...")
    
    con.close()
    print("\nTest complete!")

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path) as f:
        config = json.load(f)
    return config.get('hf_token'), config.get('blt_dir'), config.get('openai_key')

def calculate_openai_cost(num_messages, model="text-embedding-3-large"):
    """Calculate estimated cost for OpenAI embeddings"""
    # Current pricing for text-embedding-3-large is $0.00013 per 1K tokens
    avg_tokens_per_message = 100  # Estimate
    total_tokens = num_messages * avg_tokens_per_message
    cost_per_1k = 0.00013
    estimated_cost = (total_tokens / 1000) * cost_per_1k
    return estimated_cost, total_tokens

def add_openai_embeddings(batch_size=50):
    """Add OpenAI embeddings to existing database"""
    print("Loading configuration...")
    config_path = get_config_path()
    _, _, openai_key = load_config(config_path)
    if not openai_key:
        print("Error: OpenAI API key not found in config.json")
        return
    
    openai.api_key = openai_key
    
    # Connect to database
    con = duckdb.connect('embeddings_1k.duckdb', read_only=False)
    
    # Add column if it doesn't exist
    con.execute("""
        ALTER TABLE message_embeddings 
        ADD COLUMN IF NOT EXISTS embedding_openai DOUBLE[];
    """)
    
    # Get count of messages without OpenAI embeddings
    count = con.execute("""
        SELECT COUNT(*) 
        FROM message_embeddings 
        WHERE embedding_openai IS NULL
    """).fetchone()[0]
    
    if count == 0:
        print("All messages already have OpenAI embeddings.")
        con.close()
        return
    
    # Calculate and display cost estimate
    estimated_cost, total_tokens = calculate_openai_cost(count)
    print(f"\nProcessing {count} messages:")
    print(f"Estimated tokens: {total_tokens:,}")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    
    response = input("\nDo you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        con.close()
        return
    
    # Process messages in batches
    messages = con.execute("""
        SELECT rowid, message 
        FROM message_embeddings 
        WHERE embedding_openai IS NULL
    """).fetchall()
    
    print(f"\nProcessing {len(messages)} messages in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(messages), batch_size)):
        batch = messages[i:i + batch_size]
        if not batch:
            break
        
        # Extract messages and IDs
        batch_ids = [row[0] for row in batch]
        batch_texts = [row[1] for row in batch]
        
        try:
            # Get embeddings from OpenAI
            response = openai.embeddings.create(
                model="text-embedding-3-large",
                input=batch_texts
            )
            
            # Update database with new embeddings
            for idx, embedding_data in zip(batch_ids, response.data):
                embedding = embedding_data.embedding
                con.execute("""
                    UPDATE message_embeddings 
                    SET embedding_openai = ?
                    WHERE rowid = ?
                """, [embedding, idx])
            
            # Small delay to avoid rate limits
            sleep(0.1)
            
        except Exception as e:
            print(f"\nError processing batch: {str(e)}")
            print("Saving progress and exiting...")
            break
    
    # Commit changes and close connection
    con.close()
    print("\nProcessing complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run basic test mode')
    parser.add_argument('--test-similarity', action='store_true', help='Run similarity test')
    parser.add_argument('--message-count', type=int, default=10, help='Number of messages to process in test modes')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--skip-similar', action='store_true', default=True, help='Skip messages that are similar to existing ones')
    parser.add_argument('--similarity-threshold', type=float, default=0.99, help='Similarity threshold for skipping messages')
    parser.add_argument('--add-openai-embeddings', action='store_true', help='Add OpenAI embeddings to existing messages')
    args = parser.parse_args()
    
    if args.test:
        test_vectorization(args.message_count)
    elif args.test_similarity:
        test_vectorization_similarity(args.message_count)
    elif args.add_openai_embeddings:
        add_openai_embeddings(batch_size=args.batch_size)
    else:
        vectorize_messages(batch_size=args.batch_size, skip_similar=args.skip_similar, similarity_threshold=args.similarity_threshold)