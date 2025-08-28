import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import polars as pl
from language_detection.detect_language import _detect_multiple_language_iso, df_language_verified
from multi_lang_det_utils import window_sliders, get_langs, detect_multi_lang, COUNTRY_TO_LANGS


def original_implementation(df: pl.DataFrame) -> None:
    df = df.lazy()
    lazydf = df_language_verified(df)
    
    lazydf.collect().select(['index', 'respondent id', 'comments', 'comments_language_id']).write_parquet(f'multiple_language_detection/{file}_multi_detect_ori.parquet')

def _windows_lang_detect(text: str, country: str) -> list:
    text_list = window_sliders(text, 1000)
    lang = get_langs(COUNTRY_TO_LANGS, country)

    return detect_multi_lang(text_list, lang)

# print(_windows_lang_detect('Hello', 'INDONESIA'))

def windows_implementation(df: pl.DataFrame) -> None:
    last_df = df.with_columns([
        pl.col("comments").map_elements(lambda x: _windows_lang_detect(x, 'THAILAND'), return_dtype=pl.List(pl.Utf8)).alias("comments_language_id")
    ])

    last_df.select(['index', 'respondent id', 'comments', 'comments_language_id']).write_parquet(f'multiple_language_detection/{file}_multi_detect_window.parquet')

file = 'Mitr Phol JUN 2025- text comments'
df = pl.read_excel(f'data/src/{file}.xlsx.xlsx').with_row_index()

# original_implementation(df)

# windows_implementation(df)

ori_df = pl.read_parquet('multiple_language_detection/Mitr Phol JUN 2025- text comments_multi_detect_ori.parquet')

windows_df = pl.read_parquet('multiple_language_detection/Mitr Phol JUN 2025- text comments_multi_detect_window.parquet')

ori_df.join(windows_df, on=['index']).write_parquet('multiple_language_detection/Mitr Phol JUN 2025- text comments_compare.parquet')