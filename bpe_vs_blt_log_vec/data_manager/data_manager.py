from pathlib import Path
import os
import pandas as pd

def get_parq_file_path():
    # Define the file path using environment variable for user home
    parquet_path = Path(os.environ['USERPROFILE']) / "source/datasets/log2ml/lab_logs_blindtest_activity_sysmon_1000samples_july_28_2024_filtered_with_vectors_clean.parquet"

    if parquet_path.exists():
        return parquet_path
    else:
        raise FileNotFoundError(f"The JSON file does not exist at: {parquet_path}")

def load_filtered_messages():
    """
    Load only the filtered_message column from the parquet file into a pandas DataFrame
    
    Returns:
        pandas.DataFrame: DataFrame containing only the filtered_message column
    """
    parquet_path = get_parq_file_path()
    return pd.read_parquet(parquet_path, columns=['filtered_message'])

def main():
    try:
        df = load_filtered_messages()
        print(f"Successfully loaded {len(df)} messages")
        print("\nFirst 5 messages:")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()