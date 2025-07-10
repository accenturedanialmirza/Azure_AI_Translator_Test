# %%
import pandas as pd
import spacy
from date_spacy import find_dates

# %% [markdown]
# !spacy download en_core_web_sm

# %%
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

# Add specific date patterns
date_pattern_regex = {
    "label": "DATE",
    "pattern": [
        {"TEXT": {"REGEX": "\\d{2}/\\d{2}/\\d{4}"}},
        {"TEXT": {"REGEX": "\\d{1,2}\\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\\s+\\d{4}"}}
    ]
}
ruler.add_patterns([date_pattern_regex])


# %%
# Sample dataframe
data = {'comments': [
    "My credit card number is 1234-5678-9012-3456",
    "My date of birth is 01/01/1990",
    "My driver's license number is A1234567",
    "My financial information includes bank account number 123456789",
    "The full name is John Doe",
    "My gender is male",
    "My mailing address is 123 Main St, Anytown, USA",
    "My medical records show I have diabetes",
    "My passport information is passport number 987654321",
    "My place of birth is Anytown, USA",
    "My race is Caucasian",
    "My religion is Christianity",
    "My Social Security number (SSN) is 123-45-6789",
    "My ZIP code is 12345"
]}


# %%
df = pd.DataFrame(data)
df

# %%
# Define a function to redact PII using spaCy NER

"""
    Entity Label	Description
    PERSON	People, including fictional
    NORP	Nationalities, religious and political groups
    FAC	Facilities (e.g., buildings, airports, highways)
    ORG	Organizations (e.g., companies, institutions)
    GPE	Countries, cities, states
    LOC	Non-GPE locations (e.g., mountains, bodies of water)
    PRODUCT	Products (e.g., phones, cars, food)
    EVENT	Named events (e.g., wars, sports events)
    WORK_OF_ART	Titles of books, songs, etc.
    LAW	Named documents made into laws
    LANGUAGE	Any named language
    DATE	Absolute or relative dates
    TIME	Times smaller than a day
    PERCENT	Percentage values
    MONEY	Monetary values
    QUANTITY	Measurements (e.g., weight, distance)
    ORDINAL	"First", "second", etc.
    CARDINAL	Numerals that do not fall under another type
"""
def redact_pii(text):
    doc = nlp(text)
    redacted_text = text
    for ent in doc.ents:
        if ent.label_ in ["SSN", 
                          "GENDER", 
                          "PERSON", 
                          "NORP", 
                          "FAC", 
                          "ORG", 
                          "GPE", 
                          "LOC", 
                          "PRODUCT", 
                          "EVENT", 
                          "WORK_OF_ART", 
                          "LAW", 
                          "LANGUAGE", 
                          "DATE", 
                          "TIME", 
                          "PERCENT", 
                          "MONEY", 
                          "QUANTITY",
                          "ORDINAL",
                          "CARDINAL"]:
            redacted_text = redacted_text.replace(ent.text, "[REDACTED]")
        else:
            redacted_text = redacted_text.replace(ent.text, "[REDACTED]")
    return redacted_text


# %%
# Apply the function to the comments column
df['comments'] = df['comments'].apply(redact_pii)

df

# %%
doc = nlp("The social Security number (SSN) is 123-45-6789")
for ent in doc.ents:
    print(ent.text, ent.label_)

# %%



