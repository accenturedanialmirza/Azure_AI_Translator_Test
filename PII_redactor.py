import pandas as pd
import spacy
from date_spacy import find_dates

# Load the spaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

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
ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True}, before="ner")
ruler.add_patterns([ssn_pattern_regex])

# Add gender/sex pattern
gender_pattern_regex = {
    "label": "GENDER",
    "pattern": [
        {"LOWER": {"REGEX": "\\b(gender|sex|male|female|man|woman|boy|girl|he|she|him|her)\\b"}}
    ]
}
ruler.add_patterns([gender_pattern_regex])

# Add date finder
nlp.add_pipe('find_dates', after="ner")

# Define a function to redact PII using spaCy NER
def redact_pii(text):
    doc = nlp(text)
    redacted_text = text
    for ent in doc.ents:
        if ent.label_ in ["SSN", "GENDER", "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:
            redacted_text = redacted_text.replace(ent.text, "[]")
        else:
            redacted_text = redacted_text.replace(ent.text, "[]")
    return redacted_text


