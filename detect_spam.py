import joblib
import polars as pl

# Load the model and vectorizer from the files
model = joblib.load('spam-detection/naive_bayes_model.pkl')
vectorizer = joblib.load('spam-detection/vectorizer.pkl')

def classify_comment(comment: str) -> bool:
    # Transform the input text using the loaded vectorizer
    comment_vec = vectorizer.transform([comment])
    
    # Predict the class using the loaded model
    prediction = model.predict(comment_vec)
    
    # Return the prediction
    return prediction[0] == 1 

# file = "data/prod/MIS menuju SSOT JUL 2024- text comments_detected-simplified_translated_lazy.parquet"

# df = pl.scan_parquet(file)

# df = df.with_columns(pl.col("translated_text").map_elements(lambda text: classify_comment(text), return_dtype=pl.Boolean).alias("is_spam"))

# print(df.filter(
#     pl.col("is_spam") == False).select("translated_text", "is_spam").collect().write_csv('checking_spam.csv'))