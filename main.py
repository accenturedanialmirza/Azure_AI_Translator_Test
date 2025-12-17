import os 
# os.chdir('/home/azureuser/cloudfiles/code/Users/danial.m.bin.madrawi/Azure_AI_Translator_Test')

from modules.decide_batch_size import decide_batch_size
from modules.translator_df import Translator
from PII_redactor import redact_pii, redact_pii_df

from detect_non_informative import predict_non_informative_comment
import polars as pl
import re

if __name__ == "__main__":

    file = "Infinitas SEP 2023- text comments"
    file_detected = f"./data/src/{file}_detected.csv"

    # detected language dataset 
    df = pl.scan_csv(file_detected)

    # decide batch size
    automated_mini_batch_size = decide_batch_size(df)

    # translate the sentences
    translator_instance = Translator(
        input_path=file_detected,
        mini_batch_size=automated_mini_batch_size # Use the automated batch size
    )

    # Process the translation for the 'comments' column using our explicit mini-batch approach.
    processed_df =  translator_instance.process_translation_lazy(column="comments")

    # redact translated comments
    redacted_processed_df = processed_df \
                .with_columns([
                    pl.struct(["translated_text"]).map_elements(lambda row: redact_pii(row["translated_text"]), return_dtype=pl.Utf8).alias("redacted_translated_text")
                ])
    
    predicted_redacted_processed_df = redacted_processed_df.with_columns([
        pl.struct(["question code", "translated_text"]).map_elements(lambda row: predict_non_informative_comment(row["question code"], row["translated_text"]), return_dtype=pl.Boolean).alias("is_non_informative")
    ])

    predicted_redacted_processed_df.write_csv(f'./data/prod/{file}_detected_translated_redacted_predicted.csv')