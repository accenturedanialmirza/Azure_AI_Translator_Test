from translator_copilot_lazy import Translator
from detect_language import df_language_verified
from detect_spam import classify_comment
from split_texts import split_text, split_sentences_into_rows
import polars as pl
import re

if __name__ == "__main__":

    file = "MIS menuju SSOT JUL 2024- text comments"

    # detect language
    df = pl.scan_csv(f'./data/src/{file}.csv')
    df_language_verified(df).sink_csv(f'./data/src/{file}_detected.csv')

    file_detected = f"./data/src/{file}_detected.csv"

    # translate the sentences
    translator_instance = Translator(
        input_path=file_detected,
        mini_batch_size=100  # Set your desired mini-batch size here.
    )

    # Process the translation for the 'comments' column using our explicit mini-batch approach.
    processed_df =  translator_instance.process_translation_lazy(column="comments")

    processed_df = processed_df.with_columns([
                    pl.struct(["comments", "source_text_length"]).map_elements(lambda row: split_text(row["comments"], row["source_text_length"]), return_dtype=pl.List(pl.Utf8)).alias("source_split_texts")
                ]).with_columns([
                    pl.struct(["translated_text", "translated_text_length"]).map_elements(lambda row: split_text(row["translated_text"], row["translated_text_length"]), return_dtype=pl.List(pl.Utf8)).alias("translated_split_texts")
                ])

    # detect spam
    processed_df = processed_df.with_columns(pl.col("translated_text").map_elements(lambda text: classify_comment(text), return_dtype=pl.Boolean).alias("is_spam"))

    processed_df.write_parquet(f"./data/prod/{file}_translated_lazy.parquet")

    # split the sentences
    final_df = split_sentences_into_rows(processed_df, "source_split_texts", "translated_split_texts")
    final_df.write_parquet(f"./data/prod/{file}_translated_split_lazy.parquet")