import pandas as pd
import spacy
from date_spacy import find_dates
import re
import polars as pl

# Load the spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_trf")

ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True}, before="ner")

# Add Social Security Number
ssn_pattern_regex = {
    "label": "SSN",
    "pattern": [
        {"TEXT": {"REGEX": "\\d{3}"}},
        {"TEXT": "-"},
        {"TEXT": {"REGEX": "\\d{2}"}},
        {"TEXT": "-"},
        {"TEXT": {"REGEX": "\\d{4}"}}
    ]
}
# Add gender/sex pattern
gender_pattern_regex = {
    "label": "GENDER",
    "pattern": [
        {"LOWER": {"REGEX": "\\b(gender|sex|male|female|man|woman|boy|girl|he|she|him|her)\\b"}}
    ]
}
email_pattern_regex = {
    "label": "EMAIL",
    "pattern": [
        {"TEXT": {"REGEX": "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}}
    ]
}

ruler.add_patterns([ssn_pattern_regex, gender_pattern_regex, email_pattern_regex])

def regex_name_fallback(text, redacted_text):
    # Match words that look like names (e.g., lowercase words in a sentence)
    name_like_words = re.findall(r'\b[a-z][a-z]+\b', text)
    for word in name_like_words:
        if word in text and word not in redacted_text:
            redacted_text = re.sub(rf'\b{word}\b', '[]', redacted_text)
    return redacted_text

# Define a function to redact PII using spaCy NER
def redact_pii(text):
    doc = nlp(text)
    redacted_text = text
    for ent in doc.ents:
        # if ent.label_ in ["SSN", "GENDER", "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:
        if ent.label_ in ["PERSON"]:
            redacted_text = redacted_text.replace(ent.text, "[]")
    redacted_text = regex_name_fallback(text, redacted_text)
    return redacted_text

# print(redact_pii("There's no overall DT project progress shared with regions with andrew, regions are asking more but no regular official information provided."))
def redact_pii_df(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col(''))
    return df