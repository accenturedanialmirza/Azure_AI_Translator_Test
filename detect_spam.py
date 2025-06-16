import joblib
import polars as pl
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer from the files
# model = joblib.load('spam-detection/naive_bayes_model.pkl')
# vectorizer = joblib.load('spam-detection/vectorizer.pkl')

# def classify_comment(comment: str) -> bool:
#     # Transform the input text using the loaded vectorizer
#     comment_vec = vectorizer.transform([comment])
    
#     # Predict the class using the loaded model
#     prediction = model.predict(comment_vec)
    
#     # Return the prediction
#     return prediction[0] == 1 

# file = "data/prod/MIS menuju SSOT JUL 2024- text comments_detected-simplified_translated_lazy.parquet"

# df = pl.scan_parquet(file)

# df = df.with_columns(pl.col("translated_text").map_elements(lambda text: classify_comment(text), return_dtype=pl.Boolean).alias("is_spam"))

# print(df.filter(
#     pl.col("is_spam") == False).select("translated_text", "is_spam").collect().write_csv('checking_spam.csv'))

def _clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower().strip()
    return text

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