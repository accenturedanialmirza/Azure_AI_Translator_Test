# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import os
import glob
for dirname, _, filenames in os.walk('/spam-detection'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%
excel_files = glob.glob("*.xlsx")
excel_files

# %%
data = pd.concat([pd.read_excel(file) for file in excel_files]).reset_index()

# %%
# Fill NaN values in 'comments' with empty string
data.fillna({"comments": ""}, inplace=True)

# %%
# Define our features (X) and the target (y)
X = data[['comments', 'sentiment category']]
y = data['hide comment']

# %%
# This is crucial to evaluate how well our model generalizes to new, unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
# This technique reflects the importance of a word in a document within a collection of documents.
text_processor = TfidfVectorizer()

# %%
categorical_processor = OneHotEncoder(handle_unknown='ignore')

# %%

# Combine the processors into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_processor, 'comments'),
        ('category', categorical_processor, ['sentiment category'])
    ],
    remainder='passthrough' # Keep other columns if any (none in this case)
)


# %%
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# %%
model_pipeline.fit(X_train, y_train)

# %%
y_pred = model_pipeline.predict(X_test)

# %%
print(confusion_matrix(y_test, y_pred))

# %%
print(classification_report(y_test, y_pred))

# %%
print(accuracy_score(y_test, y_pred))

# %%
y_pred

# Create a DataFrame to compare actual and predicted values
comparison_df = pd.DataFrame({
    'comments': X_test['comments'],
    'sentiment_category': X_test['sentiment category'],
    'actual': y_test,
    'predicted': y_pred
})

# Identify false positives
false_positives = comparison_df[(comparison_df['actual'] == 0) & (comparison_df['predicted'] == 1)]

# Print false positives
print("False Positives:")
print(false_positives)

# %%
import joblib

joblib.dump(model_pipeline, "comment_hide_classifier.joblib")


# %% [markdown]
# import joblib
# 
# joblib.dump(model, 'model.pkl')
# joblib.dump(X.columns, 'model_features.pkl')

# %% [markdown]
# model = joblib.load('model.pkl')
# model_features = joblib.load('model_features.pkl')
# 
# # Function to make predictions
# def predict_hide_comment(comments, sentiment_category) -> bool:
#     # Create a DataFrame for the input data
#     input_data = pd.DataFrame({
#         # 'question code': [question_code],
#         'comments': [comments],
#         'sentiment category': [sentiment_category],
#     })
#     
#     # Convert categorical data to numerical data for 'question code'
#     input_data = pd.get_dummies(input_data, columns=['sentiment category'])
#     
#     # Vectorize 'comments' column using the loaded TF-IDF vectorizer
# 
#     comments_tfidf = vectorizer.transform(input_data['comments']).toarray()
#     
#     # Drop the original 'comments' column and add the TF-IDF features
#     input_data = input_data.drop(columns=['comments'])
#     input_data = pd.concat([input_data, pd.DataFrame(comments_tfidf, index=input_data.index)], axis=1)
# 
#     # Reindex input_data to match the columns used during training
#     # Fill any missing columns with 0
#     input_data = input_data.reindex(columns=model_features, fill_value=0)
#     
#     input_data.columns = input_data.columns.astype(str)
#     
#     # Make prediction
#     prediction = model.predict(input_data)
#     
#     return prediction[0]
# 

# %% [markdown]
# # Example usage
# # question_code = 'dq|change_observed'
# comments = "There has been an introduction to AI solutions and tools that have helped with workflow."
# sentiment_category = "neutral"
# 
# result = predict_hide_comment(comments, sentiment_category)
# print(f"Prediction: {result}")

# %%



