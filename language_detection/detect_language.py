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

def _detect_multiple_language_iso(text: str) -> list[str]:
    if text is None or text.strip() == "":
        return ["unknown"]
    langs = detector.detect_multiple_languages_of(text)
    results = list(set([lang.language.iso_code_639_1.name.lower() for lang in langs] or ["unknown"]))
    return results

# def _windows_lang_detect(text: str, country: str) -> list:
    text_list = window_sliders(text, 1000)
    lang = get_langs(COUNTRY_TO_LANGS, country)

    return detect_multi_lang(text_list, lang)

def df_language_verified(df: pl.LazyFrame) -> pl.LazyFrame:
    # Load the LazyFrame and add the language detection column
    lf = df.with_columns([
            pl.col("comments").map_elements(_detect_language_iso, return_dtype=pl.String).alias("comments_language_id")
        ])
    
    # lf = df.with_columns([
    #         pl.col("comments").map_elements(_detect_multiple_language_iso, return_dtype=pl.List(pl.Utf8)).alias("comments_language_id")
    #     ])

    return lf

# def _lang_detect_country(text: str, country: str = None) -> list[str]:
    try:
        if text is None or text.strip() == "":
            return ["unknown"]
        if country == None or country.strip() == "" or country not in dict(COUNTRY_TO_LANGS).keys():
            return _detect_multiple_language_iso(text)
        else:
            return _windows_lang_detect(text, country)
    except Exception as e:
        return [e]

# def def_language_verified_country(df: pl.LazyFrame) ->  pl.LazyFrame:
    lf = df.with_columns(pl.col("EE Country").str.to_uppercase().alias("EE Country")) \
            .with_columns([
                pl.struct("comments", pl.col("EE Country")).map_elements(
                    lambda row: _lang_detect_country(row["comments"], row["EE Country"]), return_dtype=pl.List(pl.Utf8)).alias("comments_language_id")
            ])

    return lf

# def for_multiple_detection():
    file_text = pl.read_excel(
        'data/src/ABB HR Transformation- Cycle 3- text comments.xlsx.xlsx'
    )

    file_demographic = pl.read_excel('data/src/ABB HR Transformation- Cycle 3.xlsx',
                        engine="calamine",
                        read_options={"header_row": 1}
                    ).select(["respondent id", "EE Country"]).unique(subset=["respondent id", "EE Country"])
    df = file_text.join(file_demographic, on="respondent id").lazy()

    df_language_detected = def_language_verified_country(df).collect()

    df_language_detected.write_parquet('ABB HR Transformation- Cycle 3_detected.parquet')


def for_single_detection():
    file = "Vertex Procurement Dexterity Raw Scores- text comments"

    # df = pl.scan_csv(f'./data/src/{file}.csv')

    df = pl.read_excel(f'./data/src/{file}.xlsx.xlsx').lazy()

    # df_language_verified(df).sink_csv(f'./data/src/{file}_detected.csv')

    # df_language_verified(df).collect().to_pandas().to_excel(f'./data/src/{file}_detected.xlsx', index=False)

    df_language_verified(df).sink_csv(f'./data/src/{file}_detected.csv')

    # print(_detect_multiple_language_iso("Sudah baik tetapi harus terus ditingkatkan"))

for_single_detection()