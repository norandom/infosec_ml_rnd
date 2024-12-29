def get_parq_file_path():
    """Get path to parquet file"""
    from pathlib import Path
    import os
    parquet_path = Path(os.environ['USERPROFILE']) / "source/datasets/log2ml/lab_logs_blindtest_activity_sysmon_1000samples_july_28_2024_filtered_with_vectors_clean.parquet"

    if parquet_path.exists():
        return parquet_path
    else:
        raise FileNotFoundError(f"The JSON file does not exist at: {parquet_path}")

def load_filtered_messages():
    """
    Load filtered_message column from the parquet file into a pandas DataFrame
    
    Returns:
        pandas.DataFrame: DataFrame containing filtered_message column
    """
    import pandas as pd
    parquet_path = get_parq_file_path()
    return pd.read_parquet(parquet_path, columns=['filtered_message'])

def main():
    try:
        # Load messages from parquet file
        df = load_filtered_messages()
        print(f"Loaded {len(df)} messages from parquet file")
        
        # Print first few messages
        print("\nFirst few messages:")
        for msg in df['filtered_message'].head():
            print(f"- {msg}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()