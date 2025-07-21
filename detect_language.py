import polars as pl
from lingua import Language, LanguageDetectorBuilder

languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, \
             Language.ITALIAN, Language.CHINESE, Language.JAPANESE, Language.PORTUGUESE, \
             Language.INDONESIAN, Language.THAI, Language.MALAY, Language.ARABIC, \
             Language.POLISH, Language.CZECH]
# detector = LanguageDetectorBuilder.from_languages(*Language.all()).build()
detector = LanguageDetectorBuilder.from_languages(*languages).build()
# file = "Mitr Phol JUN 2025- text comments- v2"

# df = pl.scan_csv(f'./data/src/{file}.csv')

# df = pl.read_excel(f'./data/src/{file}.xlsx.xlsx').lazy()

# Define a function to detect language and return the ISO code
def _detect_language_iso(text: str) -> str:
    if text is None or text.strip() == "":
        return "unknown"
    lang = detector.detect_language_of(text)
    return lang.iso_code_639_1.name.lower() if lang else "unknown"

def _detect_multiple_language_iso(text: str) -> list[str]:
    if text is None or text.strip() == "":
        return ["unknown"]
    langs = detector.detect_multiple_languages_of(text)
    results = [lang.language.iso_code_639_1.name.lower() for lang in langs] or ["unknown"]
    return results

def df_language_verified(df: pl.LazyFrame) -> pl.LazyFrame:
    # Load the LazyFrame and add the language detection column
    lf = df.with_columns([
            pl.col("comments").map_elements(_detect_language_iso, return_dtype=pl.String).alias("comments_language_id")
        ])
    
    # lf = df.with_columns([
    #         pl.col("comments").map_elements(_detect_multiple_language_iso, return_dtype=pl.List(pl.Utf8)).alias("comments_language_id")
    #     ])

    return lf


# df_language_verified(df).sink_csv(f'./data/src/{file}_detected.csv')

# df_language_verified(df).collect().to_pandas().to_excel(f'./data/src/{file}_detected.xlsx', index=False)

# df_language_verified(df).sink_parquet(f'./data/src/{file}_multiple_detected.parquet')