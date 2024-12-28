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
        return f'test_{message_count}_messages.duckdb'
    else:
        return 'embeddings_1k.duckdb'

def init_models_and_db(mode='full', message_count=None, linformer_only=False):
    """Initialize models and database connection"""
    # Load configuration and BLT library
    config_path = get_config_path()
    
    if not linformer_only:
        hf_token, blt_dir, _ = load_config(config_path)
        load_blt_lib(blt_dir)
        # Initialize BLT model
        model = create_blt_model(model_dim=1024, verbose=False)
    else:
        model = None
        
    tokenizer_path = Path("tokenizers/tokenizer.model")
    
    # Initialize DuckDB with mode-specific database
    con = duckdb.connect(get_db_name(mode, message_count), read_only=False)
    
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
    """
    Process a batch of messages to create embeddings with performance metrics
    """
    blt_embeddings = []
    bpe_embeddings = []
    blt_times = []
    bpe_times = []

    
    tokenizer_path = Path(__file__).parent / "tokenizers" / "original"

    
    for msg in messages:
        # BLT embedding
        start_time = time.time()
        blt_emb = get_text_embedding(msg, model, tokenizer_path)
        blt_time = time.time() - start_time
        blt_times.append(blt_time)
        blt_embeddings.append(blt_emb)
        
        # Linformer embedding
        start_time = time.time()
        bpe_emb = get_linformer_embedding(msg)
        bpe_time = time.time() - start_time
        bpe_times.append(bpe_time)
        bpe_embeddings.append(bpe_emb)
    
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
    
    # Initialize models and database
    model, tokenizer_path, con = init_models_and_db(mode='test', message_count=message_count)
    
    # Get test messages
    messages = load_filtered_messages()[:message_count]
    
    # Process messages
    blt_embeddings, bpe_embeddings, blt_times, bpe_times = process_batch(messages, model, tokenizer_path)
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print("BLT Vectorization:")
    print(f"  - Average time per message: {np.mean(blt_times):.4f} seconds")
    print(f"  - Total time for {message_count} messages: {sum(blt_times):.4f} seconds")
    print(f"  - Messages per second: {message_count/sum(blt_times):.2f}")
    
    print("\nLinformer (BPE) Vectorization:")
    print(f"  - Average time per message: {np.mean(bpe_times):.4f} seconds")
    print(f"  - Total time for {message_count} messages: {sum(bpe_times):.4f} seconds")
    print(f"  - Messages per second: {message_count/sum(bpe_times):.2f}")
    
    # Store results in database
    for msg, blt_emb, bpe_emb in zip(messages, blt_embeddings, bpe_embeddings):
        # Convert numpy arrays to lists and ensure 1D shape
        blt_list = blt_emb.tolist() if isinstance(blt_emb, np.ndarray) else blt_emb.cpu().numpy().squeeze().tolist()
        bpe_list = bpe_emb.tolist() if isinstance(bpe_emb, np.ndarray) else bpe_emb.cpu().numpy().squeeze().tolist()
        
        # Ensure we have 1024-dim vectors
        if len(blt_list) != 1024 or len(bpe_list) != 1024:
            print(f"\nError: Vector dimensions mismatch")
            print(f"BLT shape: {len(blt_list)}")
            print(f"BPE shape: {len(bpe_list)}")
            continue
            
        try:
            con.execute("""
                INSERT INTO message_embeddings (message, embedding_blt, embedding_bpe)
                VALUES (?, ?, ?)
            """, (msg, blt_list, bpe_list))
        except Exception as e:
            print(f"\nError storing vectors: {e}")
            print(f"BLT shape: {np.array(blt_list).shape}")
            print(f"BPE shape: {np.array(bpe_list).shape}")
            raise
    
    con.close()
    print("\nTest complete! Results stored in database.")

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

def test_update_linformer(message_count=2):
    """Test updating only Linformer vectors for a few messages"""
    print(f"\nTesting Linformer update with {message_count} messages...")
    
    # First create test data
    test_vectorization(message_count)
    
    print("\nNow testing Linformer update...")
    
    # Connect to database with test mode
    con = duckdb.connect(get_db_name(mode='test', message_count=message_count), read_only=False)
    
    try:
        # Get sample messages with their existing BLT vectors
        rows = con.execute(f"""
            SELECT rowid, message, embedding_blt 
            FROM message_embeddings 
            LIMIT {message_count}
        """).fetchall()
        
        print(f"Found {len(rows)} messages to update")
        
        # Process each message
        for rowid, message, blt_vector in rows:
            print(f"\nProcessing message {rowid}:")
            print(f"Message preview: {message[:100]}...")
            
            # Generate new Linformer embedding
            start_time = time.time()
            linformer_emb = get_linformer_embedding(message)
            process_time = time.time() - start_time
            
            print(f"Generated Linformer vector of shape: {linformer_emb.shape}")
            print(f"Processing time: {process_time:.2f} seconds")
            
            try:
                # Update only the Linformer vector
                con.execute("""
                    UPDATE message_embeddings 
                    SET embedding_bpe = ?
                    WHERE rowid = ?
                """, (linformer_emb.tolist(), rowid))
                
                # Verify the update
                result = con.execute("""
                    SELECT 
                        array_length(embedding_blt) as blt_len,
                        array_length(embedding_bpe) as bpe_len
                    FROM message_embeddings 
                    WHERE rowid = ?
                """, [rowid]).fetchone()
                
                if result:
                    blt_len = result[0] if result[0] is not None else 0
                    bpe_len = result[1] if result[1] is not None else 0
                    print(f"Verified dimensions - BLT: {blt_len}, Linformer: {bpe_len}")
                    if blt_len != 1024 or bpe_len != 1024:
                        print("Warning: Vector dimensions are not 1024!")
                else:
                    print("Warning: Could not verify vector dimensions after update")
                    
            except Exception as e:
                print(f"Error updating vector: {e}")
            
    finally:
        con.close()
    
    print("\nTest complete! Linformer vectors updated successfully.")

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

def update_linformer_vectors(batch_size=50):
    """
    Update only the Linformer vectors in the database, keeping BLT vectors unchanged.
    
    Args:
        batch_size (int): Number of messages to process at once
    """
    print("\nInitializing Linformer update...")
    
    # Check for state file
    state_file = Path("linformer_update_state.txt")
    last_processed_id = 0
    if state_file.exists():
        with open(state_file) as f:
            try:
                last_processed_id = int(f.read().strip())
                print(f"Resuming from row ID: {last_processed_id}\n")
            except ValueError:
                print("Invalid state file, starting from beginning\n")
    
    # Initialize database connection only, skip BLT model
    _, _, con = init_models_and_db(linformer_only=True)
    
    # Get total count for progress bar
    total_rows = con.execute(f"""
        SELECT COUNT(*) 
        FROM message_embeddings 
        WHERE rowid > {last_processed_id}
    """).fetchone()[0]
    print(f"Found {total_rows} messages to process\n")
    
    # Pre-initialize Linformer model to get GPU message before progress bar
    dummy_message = "Initializing model"
    _ = get_linformer_embedding(dummy_message)
    print("")  # Add space after GPU message
    
    print("Starting update process...\n")
    
    try:
        # Process in batches
        for offset in tqdm(range(0, total_rows, batch_size), desc="Processing messages"):
            # Get batch of messages
            batch = con.execute(f"""
                SELECT rowid, message 
                FROM message_embeddings 
                WHERE rowid > {last_processed_id}
                ORDER BY rowid
                LIMIT {batch_size}
            """).fetchall()
            
            if not batch:
                continue
                
            # Generate new Linformer embeddings
            linformer_embeddings = []
            for _, message in batch:
                emb = get_linformer_embedding(message)
                linformer_embeddings.append(emb.tolist())
            
            # Update database
            for (rowid, _), emb in zip(batch, linformer_embeddings):
                con.execute("""
                    UPDATE message_embeddings 
                    SET embedding_bpe = ?
                    WHERE rowid = ?
                """, (emb, rowid))
                last_processed_id = rowid
                
            # Save state after each batch
            with open(state_file, 'w') as f:
                f.write(str(last_processed_id))
                
            # Commit after each batch
            con.commit()
            
    except Exception as e:
        print(f"\nError during update: {e}")
        print(f"Last processed row ID: {last_processed_id}\n")
        raise
    finally:
        con.close()
        if total_rows == 0 or last_processed_id == con.execute("SELECT MAX(rowid) FROM message_embeddings").fetchone()[0]:
            print("\nUpdate fully completed!\n")
            # Clean up state file
            if state_file.exists():
                state_file.unlink()
        else:
            print(f"\nUpdate interrupted. Progress saved at row ID: {last_processed_id}")
            print("Run the same command again to resume from this point.\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Sysmon logs with different embedding models')
    
    # Mode selection
    parser.add_argument('--mode', 
                      choices=['vectorize', 'test', 'test_similarity', 'update_linformer', 'test_update_linformer'],
                      default='test',
                      help='Operation mode:\n'
                           'vectorize: Process all messages and create embeddings\n'
                           'test: Run basic test on a small number of messages\n'
                           'test_similarity: Test similarity detection\n'
                           'update_linformer: Update Linformer vectors in database\n'
                           'test_update_linformer: Test Linformer update on small dataset')
    
    # Processing parameters
    parser.add_argument('--batch_size', 
                      type=int, 
                      default=50,
                      help='Number of messages to process in each batch')
    
    # Test parameters
    parser.add_argument('--message_count', 
                      type=int, 
                      default=10,
                      help='Number of messages to process in test modes')
    
    # Optional parameters
    parser.add_argument('--similarity_threshold',
                      type=float,
                      default=0.99,
                      help='Threshold for considering vectors similar (0.0-1.0)')
    
    args = parser.parse_args()
    
    try:
        # Map modes to functions
        mode_functions = {
            'vectorize': lambda: vectorize_messages(
                batch_size=args.batch_size,
                similarity_threshold=args.similarity_threshold
            ),
            'test': lambda: test_vectorization(
                message_count=args.message_count
            ),
            'test_similarity': lambda: test_vectorization_similarity(
                message_count=args.message_count
            ),
            'update_linformer': lambda: update_linformer_vectors(
                batch_size=args.batch_size
            ),
            'test_update_linformer': lambda: test_update_linformer(
                message_count=args.message_count
            )
        }
        
        # Execute selected function
        print(f"\nRunning in {args.mode} mode...")
        mode_functions[args.mode]()
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise