import polars as pl
import os

def check_temp_batch_size_matches(path: str, batch_size: int) -> bool:
    # Get the list of all files in the folder
    files = os.listdir(path)

    # Filter out the parquet files and sort them
    parquet_files = sorted([f for f in files if f.endswith('.parquet')])

    # Read the first parquet file using polars
    first_file_path = os.path.join(path, parquet_files[0])
    
    df = pl.read_parquet(first_file_path)

    # # Display the dataframe
    return len(df[:,:1]) == batch_size

def remove_temp_files(path: str) -> None:
    for file in os.listdir(path):
        if file.endswith(".parquet"):
            os.remove(os.path.join(path, file))

# print(check_temp_batch_size_matches('./data/temp', 50))