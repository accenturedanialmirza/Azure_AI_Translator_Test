import polars as pl
import os
import json

from lingua import Language, LanguageDetectorBuilder
# from language_detection_windows_geo import window_sliders, get_langs, detect_multi_lang, COUNTRY_TO_LANGS

with open('language_detection/country_languages.json') as file_text:
    COUNTRY_TO_LANGS = json.load(file_text)

languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, \
             Language.ITALIAN, Language.CHINESE, Language.JAPANESE, Language.PORTUGUESE, \
             Language.INDONESIAN, Language.THAI, Language.MALAY, Language.ARABIC, \
             Language.POLISH, Language.CZECH]
# detector = LanguageDetectorBuilder.from_languages(*Language.all()).build()
detector = LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(0.05).build()

def _detect_language_iso(text: str) -> str:
    if text is None or text.strip() == "":
        return "unknown"
    lang = detector.detect_language_of(text)
    return lang.iso_code_639_1.name.lower() if lang else "unknown"

def df_language_verified(df: pl.LazyFrame) -> pl.LazyFrame:
    # Load the LazyFrame and add the language detection column
    lf = df.with_columns([
            pl.col("comments").map_elements(_detect_language_iso, return_dtype=pl.String).alias("comments_language_id")
        ])

    return lf

def main():
    file = "MIS menuju SSoT JUL 2024- text comments"

    df = pl.read_excel(f'./data/src/{file}.xlsx.xlsx').lazy()

    df_language_verified(df).sink_csv(f'./data/src/{file}_detected.csv')

if __name__ == "__main__":
    main()  