from modules.decide_batch_size import decide_batch_size
from detect_language import df_language_verified
from modules.translator_df import Translator

from detect_non_informative import predict_non_informative_comment
from modules.split_texts import split_text, split_sentences_into_rows
import polars as pl
import re

if __name__ == "__main__":

    file = "Infinitas SEP 2023- text comments"
    file_detected = f"./data/src/{file}_detected.csv"

    # # detect language
    df = pl.scan_csv(f'./data/src/{file}.csv')
    # df_language_verified(df).sink_csv(file_detected)

    # decide batch size
    automated_mini_batch_size = decide_batch_size(df)

    # translate the sentences
    translator_instance = Translator(
        input_path=file_detected,
        mini_batch_size=automated_mini_batch_size # Use the automated batch size
    )

    # Process the translation for the 'comments' column using our explicit mini-batch approach.
    processed_df =  translator_instance.process_translation_lazy(column="comments")

    processed_df = processed_df.with_columns([
                    pl.struct(["comments", "source_text_length"]).map_elements(lambda row: split_text(row["comments"], row["source_text_length"]), return_dtype=pl.List(pl.Utf8)).alias("source_split_texts")
                ]).with_columns([
                    pl.struct(["translated_text", "translated_text_length"]).map_elements(lambda row: split_text(row["translated_text"], row["translated_text_length"]), return_dtype=pl.List(pl.Utf8)).alias("translated_split_texts")
                ])

    # detect spam
    processed_df = processed_df.with_columns([
                    pl.struct(["translated_text", "sentiment category"]).map_elements(lambda row: predict_non_informative_comment(row["translated_text"], row["sentiment category"]), return_dtype=pl.Boolean).alias("is_non_informative")
                ])

    processed_df.write_parquet(f"./data/prod/{file}_translated_lazy.parquet")

    # split the sentences
    final_df = split_sentences_into_rows(processed_df, "source_split_texts", "translated_split_texts")
    final_df.write_parquet(f"./data/prod/{file}_translated_split_lazy.parquet")