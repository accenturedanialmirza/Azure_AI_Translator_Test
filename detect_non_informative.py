import joblib
import polars as pl
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

loaded_model = joblib.load("Users/danial.m.bin.madrawi/Azure_AI_Translator_Test/spam-detection/comment_hide_classifier.joblib")

def predict_non_informative_comment(comments: str) -> bool:

    input_data = pd.DataFrame({
        'comments': [comments],
    })
    
    # Make prediction
    prediction = loaded_model.predict(input_data)
    
    return bool(prediction[0])

# comment1 = "N/A"
# sentiment1 = "null"
# should_hide1 = predict_hide_comment(comment1, sentiment1)
# print(f"Comment: '{comment1}' | Sentiment: {sentiment1} | Hide? -> {should_hide1}")