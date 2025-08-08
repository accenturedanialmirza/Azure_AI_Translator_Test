
from lingua import LanguageDetectorBuilder
from lingua import Language

def window_sliders(text:str, size:int) -> list[str]:
    """
    Generates a list of sliding windows (substrings) of a given word length from the input text.

    Parameters:
        text (str): The input string to slide over.
        size (int): The number of words in each sliding window.

    Returns:
        list[str]: A list of strings, each containing `size` consecutive words from the input text.

    Raises:
        ValueError: If `text` is not a string.
        ValueError: If `size` is not an integer.
        ValueError: If `size` is less than the number of words in the text.

    Example:
        >>> window_sliders("the quick brown fox jumps", 3)
        ['the quick brown', 'quick brown fox', 'brown fox jumps']
    """
    
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

COUNTRY_TO_LANGS = {
    "THAILAND": ["THAI", "CHINESE", "ENGLISH"],
    "SPAIN": ["SPANISH", "CATALAN", "BASQUE", "ENGLISH"],
    "INDIA": ["HINDI", "BENGALI", "MARATHI", "TAMIL", "TELUGU", "GUJARATI", "URDU", "ENGLISH"],
    "INDONESIA": ["INDONESIAN", "ENGLISH"],
    "JAPAN": ["JAPANESE", "ENGLISH", "CHINESE", "KOREAN"],
    "MALAYSIA": ["MALAY", "ENGLISH", "CHINESE", "TAMIL"],
}

def get_langs(country_to_langs:dict, country):
    """
    Retrieve a list of Language objects corresponding to the languages spoken in a given country.

    Args:
        country_to_langs (dict): A dictionary mapping country names (str) to lists of language codes or names (str).
        country (str): The name of the country for which to retrieve languages.

    Returns:
        list: A list of Language objects corresponding to the languages spoken in the specified country.

    Raises:
        ValueError: If the specified country does not exist in the country_to_langs dictionary.

    Example:
        >>> get_langs(COUNTRY_TO_LANGS, "INDIA")
        [Language.HINDI, Language.BENGALI, Language.MARATHI, Language.TAMIL, Language.TELUGU, Language.GUJARATI, Language.URDU, Language.ENGLISH]
    """

    if country not in country_to_langs.keys():
        raise ValueError("Country doesnt exist in the country_to_langs")
    return [Language.from_str(l) for l in country_to_langs[country]]

def detect_multi_lang(texts:list[str]|str, languages, t=.4, window=False, n=1) -> list:
    """
    Detects multiple languages present in a given text or a list of texts using the Lingua language detector.
    IMPORTANT: This function DEPENDS on the the `window_sliders` function to generate sliding windows for language detection.

    Parameters:
        texts (list[str] | str): A single string or a list of strings to detect languages from.
        languages (list): List of Language objects to consider for detection.
        t (float, optional): Minimum relative distance threshold for language detection. Default is 0.4.
        window (bool, optional): If True, uses sliding window approach for detection. Default is False.
        n (int, optional): Window size (number of words) for sliding window detection. Default is 1.

    Returns:
        list[list[IsoCode639_1]]: For a list of texts, returns a list of detected language ISO codes for each text.
        list[IsoCode639_1]: For a single string, returns a list of detected language ISO codes.

    Raises:
        ValueError: If input is not a string or list of strings.
        ValueError: If any element in the input list is not a string.

    Example:
        >>> detect_multi_lang("Parlez-vous français? Ich spreche Deutsch.", [Language.ENGLISH, Language.FRENCH, Language.GERMAN], t=0.5, window=True, n=3)
        [IsoCode639_1.FR, IsoCode639_1.DE]
    """

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
                        r.add(i.language.iso_code_639_1)
                result.append(list(r))
                
            return result

        result_list = detector.detect_multiple_languages_in_parallel_of(texts)
        result = []
        for l in result_list:
            temp = [r.language.iso_code_639_1 for r in l]
            result.append(list(set(temp)))
        
        return result
        
    elif isinstance(texts, str):
        result_set = detector.detect_multiple_languages_of(texts)

        result = [r.language.iso_code_639_1 for r in result_set]
        
        return list(set(result))
    else:
        raise ValueError("The input text must be either a list of strings or a string.")
    
if __name__ == "__main__":
    text_list = window_sliders("saya belum tau terkait program Management Information System (MIS) menuju Single Source of Truth (SSoT)", 1000)

    lang = get_langs(COUNTRY_TO_LANGS, "THAILAND")

    print(detect_multi_lang(text_list, lang))