import joblib
import polars as pl
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer


loaded_model = joblib.load("spam-detection/comment_hide_classifier.joblib")

def predict_hide_comment(comments: str, sentiment_category: str) -> bool:

    input_data = pd.DataFrame({
        'comments': [comments],
        'sentiment category': [sentiment_category],
    })
    
    # Make prediction
    prediction = loaded_model.predict(input_data)
    
    return bool(prediction[0])

# comment1 = "N/A"
# sentiment1 = "null"
# should_hide1 = predict_hide_comment(comment1, sentiment1)
# print(f"Comment: '{comment1}' | Sentiment: {sentiment1} | Hide? -> {should_hide1}")