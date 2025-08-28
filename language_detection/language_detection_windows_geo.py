
from lingua import LanguageDetectorBuilder
from lingua import Language
import json

with open('language_detection/country_languages.json') as file:
    COUNTRY_TO_LANGS = json.load(file)

def window_sliders(text:str, size:int) -> list[str]:
    if not isinstance(text,str): raise ValueError("Text must be a string")
    if not isinstance(size,int): raise ValueError("Size must be an int")
    if size > len(text.split()): 
        return [text]
        # raise ValueError("Size must be greater than number of words in the text.")

    result = []
    text = text.split()
    last_idx = len(text) - size
    for i in range(0,last_idx+1):
        result.append((" ").join(text[i:i+size]))
    
    return result

def get_langs(country_to_langs:dict, country):
    if country not in country_to_langs.keys():
        raise ValueError("Country doesnt exist in the country_to_langs")
    return [Language.from_str(l) for l in country_to_langs[country]]

def detect_multi_lang(texts:list[str]|str, languages, t=.4, window=False, n=1) -> list:

    detector = LanguageDetectorBuilder.from_languages(*languages).with_minimum_relative_distance(t).build()

    if isinstance(texts,list):
        if any(not isinstance(x,str) for x in texts):
            raise ValueError("The input text contains non-string item, it must only contains string.")
        
        # Use window mode
        if window:
            result = []
            for s in texts:
                one_result = detector.detect_multiple_languages_in_parallel_of(window_sliders(s,n))
                r = set()
                for l in one_result:
                    for i in l:
                        r.add(i.language.iso_code_639_1.name.lower() )
                result.append(list(r))
                
            return result

        result_list = detector.detect_multiple_languages_in_parallel_of(texts)
        result = []
        for l in result_list:
            temp = [r.language.iso_code_639_1.name.lower()  for r in l]
            [result.append(x) for x in set(temp)]
        
        return result
        
    elif isinstance(texts, str):
        result_set = detector.detect_multiple_languages_of(texts)

        result = [r.language.iso_code_639_1.name.lower()  for r in result_set]
        
        return list(set(result))
    else:
        raise ValueError("The input text must be either a list of strings or a string.")
    
if __name__ == "__main__":
    text_list = window_sliders("Good morning, Apa khabar?", 1000)

    lang = get_langs(COUNTRY_TO_LANGS, "INDONESIA")

    print(detect_multi_lang(text_list, lang))

    # print(dict(COUNTRY_TO_LANGS).keys())