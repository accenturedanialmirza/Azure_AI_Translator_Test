import numpy as np
import polars as pl
import requests, uuid, json
from typing import List, Tuple, Optional
import os
from dotenv import load_dotenv
import time
from tqdm import tqdm

load_dotenv('.env')

# Add your key and endpoint
key = os.getenv("AZURE_TEXT_TRANSLATION_KEY")
endpoint = os.getenv("AZURE_TEXT_TRANSLATION_ENDPOINT")

location = "eastus"
path = "/translate"
constructed_url = endpoint + path

class Translator():
    def __init__(self, input: str, output: str):
        self.input = input
        self.output = output
    
    def create_df(self) -> pl.DataFrame:
        df = pl.read_csv(self.input)
        return df

    def translate_text(self, text: Optional[str], source_language_id: str, translate_to_language: List[str] = ['en']) -> Tuple[Optional[str], List[int]]:
        if text is None: # Handles nulls passed from Polars columns more directly
            return None, [0]
            
        params = {
            'api-version': '3.0',
            'to': translate_to_language, # Translate to English
            'includeSentenceLength': True
        }

        params['from'] = source_language_id if source_language_id != 'unknown' else 'en'

        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Ocp-Apim-Subscription-Region': location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

        # Create the body with the text to translate
        body = [{'text': text}]

        # Make the request to the Translator API
        request = requests.post(constructed_url, params=params, headers=headers, json=body)
        response = request.json()

        # Extract the translated text
        try:
            translated_text = response[0]['translations'][0]['text']
            source_text_length = response[0]['translations'][0]['sentLen']['srcSentLen']
            translated_text_length = response[0]['translations'][0]['sentLen']['transSentLen']
        except (IndexError, KeyError):
            translated_text = None  # Handle errors gracefully
            source_text_length = [0]
            translated_text_length = [0]

        return translated_text, source_text_length, translated_text_length



    def translate_dataframe_with_rate_limit(self, df: pl.DataFrame, delay: float = 1.0) -> pl.DataFrame:
        translated_texts = []
        source_lengths = []
        translated_lengths = []

        for row in tqdm(df.iter_rows(named=True)):
            text = row['comments']
            lang = row['comments_language_id']
            translated, src_len, trans_len = self.translate_text(text, lang)
            translated_texts.append(translated)
            source_lengths.append(src_len)
            translated_lengths.append(trans_len)
            time.sleep(delay)

        df = df.with_columns([
                pl.Series("translated_text", translated_texts),
                pl.Series("source_text_length", source_lengths),
                pl.Series("translated_text_length", translated_lengths)
            ])
        return df


    def split_text_by_lengths(self, text: str, lengths: list) -> List[str]:
        sentences = []
        start = 0
        for length in lengths:
            # Slice the string from the current start index to the next "start+length"
            sentences.append(text[start:start+length])
            start += length
        return sentences
    
    
if __name__ == "__main__":
    translator = Translator(input="./data/src/text-zh-simplified_detected.csv", output="./data/prod/text-zh-simplified_detected_translated.csv")
    df = translator.create_df()
    df = translator.translate_dataframe_with_rate_limit(df, delay=0.1)
    print(df)
