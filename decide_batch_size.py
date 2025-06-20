import polars as pl


def decide_batch_size(df: pl.LazyFrame) -> int:
    # Dynamically determine mini_batch_size based on total rows and comment length
    lf_comments = df.select("comments")
    total_rows = lf_comments.select(pl.len()).collect().item()

    # Calculate total characters in the 'comments' column, handling potential nulls
    total_characters = lf_comments.select(
        pl.col("comments").str.len_bytes().fill_null(0).sum()
    ).collect().item()

    max_chars_per_batch = 4000 # Target maximum characters per batch, slightly below API limit
    min_batch_size = 1
    max_batch_size_api = 100 # Azure Translator API limit

    if total_rows == 0:
        automated_mini_batch_size = max_batch_size_api # Default for empty files
    else:
        average_comment_length = total_characters / total_rows
        if average_comment_length == 0: # Avoid division by zero if all comments are empty
            automated_mini_batch_size = max_batch_size_api
        else:
            # Calculate suggested batch size based on character limit
            suggested_batch_size_by_chars = max_chars_per_batch / average_comment_length
            
            # Combine with row-based logic and API limits
            # Ensure it's an integer and within min/max bounds
            automated_mini_batch_size = int(min(max(suggested_batch_size_by_chars, min_batch_size), max_batch_size_api))

    # Ensure batch size is not greater than total rows for small files
    automated_mini_batch_size = min(automated_mini_batch_size, total_rows) if total_rows > 0 else 1

    return automated_mini_batch_size

    # print(f"Total rows: {total_rows}, Total characters in 'comments': {total_characters}, Average comment length: {average_comment_length:.2f}")
    # print(f"Using dynamically determined mini_batch_size: {automated_mini_batch_size}")

# file = "Accenture TGPS FEB 2024- text comments"

# file_detected = f"./data/src/{file}_detected.csv"

# df = pl.scan_csv(file_detected)

# print(decide_batch_size(df))