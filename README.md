# Project Documentation: Azure AI Translator and Spam Detection

This project demonstrates the use of Azure AI Translator API for text translation, integrates with Polars DataFrames for efficient data processing, and includes a spam detection module.

## 1. Project Objective

The primary objectives of this project are:

*   **Testing Azure AI Translator API**: To evaluate the capabilities of the Azure AI Translator API for language translation.
*   **Efficient Data Processing with Polars**: To leverage the Polars DataFrame library for high-performance data manipulation and analysis.
*   **Spam Detection**: To classify comments as spam or not spam using a pre-trained model.

## 2. Key Features and Results

*   **Successful Translation**: The project successfully translates Chinese text into English, including the ability to obtain sentence break lengths.
*   **Data Persistence**: Translated results are efficiently saved into Parquet files. CSV format is avoided due to its limitations with nested data structures.
*   **Language Detection**: Automatically detects the language of input text.
*   **Text Splitting**: Splits source and translated texts into individual sentences for detailed analysis.

## 3. Project Structure

The project is organized into several Python modules, each responsible for a specific part of the workflow:

*   [`main.py`](main.py): The main entry point of the application, orchestrating the language detection, translation, text splitting, and spam detection processes.
*   [`translator_copilot_lazy.py`](translator_copilot_lazy.py): Contains the `Translator` class responsible for handling Azure AI Translator API calls and processing translations in a lazy, mini-batch approach using Polars.
*   [`detect_language.py`](detect_language.py): Implements the `df_language_verified` function for detecting the language of text within a Polars DataFrame.
*   [`detect_spam.py`](detect_spam.py): Provides the `classify_comment` function for identifying spam comments using a pre-trained Naive Bayes model.
*   [`split_texts.py`](split_texts.py): Contains functions like `split_text` and `split_sentences_into_rows` for breaking down texts into sentences and restructuring DataFrames.
*   [`check_batch_size.py`](check_batch_size.py): (Presumed) Utility for checking or validating batch sizes for API calls.
*   [`translator_gemini.py`](translator_gemini.py): (Presumed) An alternative or experimental translator module, possibly using Google Gemini API.
*   [`translator.py`](translator.py): (Presumed) A base or alternative translator module.
*   [`test_azure_sample.ipynb`](test_azure_sample.ipynb): Jupyter notebook for testing Azure AI Translator API samples.
*   [`test_translator.py`](test_translator.py): Unit tests for the translator functionalities.

### Data Directory

*   `data/src/`: Contains source CSV files for processing.
*   `data/prod/`: Stores processed and translated data in Parquet format.

### Spam Detection Module

*   `spam-detection/`: Directory containing assets for the spam detection model.
    *   [`naive_bayes_model.pkl`](spam-detection/naive_bayes_model.pkl): The serialized Naive Bayes model.
    *   [`vectorizer.pkl`](spam-detection/vectorizer.pkl): The serialized TF-IDF vectorizer used for text preprocessing.
    *   [`naive-bayes-model.ipynb`](spam-detection/naive-bayes-model.ipynb): Jupyter notebook detailing the training and evaluation of the Naive Bayes model.
    *   [`youtube-comment-spam-detection-max-94-89.ipynb`](spam-detection/youtube-comment-spam-detection-max-94-89.ipynb): Another Jupyter notebook related to spam detection, possibly an earlier iteration or a different model.
    *   [`youtube-comments-spam-detection-f1-score-96.ipynb`](spam-detection/youtube-comments-spam-detection-f1-score-96.ipynb): Jupyter notebook focusing on achieving a high F1-score for spam detection.

## 4. Setup and Installation

To set up and run this project, follow these steps:

### Prerequisites

*   Python 3.8+
*   Azure subscription with Azure AI Translator resource configured.
*   Environment variables for Azure AI Translator API key and endpoint.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/Azure_AI_Translator_Test.git
    cd Azure_AI_Translator_Test
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

Create a `.env` file in the root directory of the project and add your Azure AI Translator API key and endpoint:

```
AZURE_TRANSLATOR_KEY="your_azure_translator_key"
AZURE_TRANSLATOR_ENDPOINT="your_azure_translator_endpoint"
AZURE_TRANSLATOR_REGION="your_azure_translator_region" # e.g., "eastus"
```

## 5. Usage

To run the main translation and spam detection workflow:

```bash
python main.py
```

This will:
1.  Read the input CSV file (`MIS menuju SSOT JUL 2024- text comments.csv`) from `./data/src/`.
2.  Detect the language of comments and save the result to `./data/src/MIS menuju SSOT JUL 2024- text comments_detected.csv`.
3.  Translate the comments using Azure AI Translator API.
4.  Split the source and translated texts into sentences.
5.  Classify translated comments as spam or not spam.
6.  Save the translated and spam-detected data to `./data/prod/MIS menuju SSOT JUL 2024- text comments_translated_lazy.parquet`.
7.  Further split the sentences into individual rows and save to `./data/prod/MIS menuju SSOT JUL 2024- text comments_translated_split_lazy.parquet`.

## 6. Dependencies

The project relies on the following key Python libraries, as specified in [`requirements.txt`](requirements.txt):

*   `polars`: For high-performance DataFrame operations.
*   `requests`: For making HTTP requests to the Azure AI Translator API.
*   `python-dotenv`: For loading environment variables.
*   `numpy`: Numerical computing.
*   `pandas`: Data manipulation and analysis (though Polars is preferred for core operations).
*   `pyarrow`: For Parquet file handling.
*   `tqdm`: For progress bars.
*   Other dependencies for Jupyter notebooks and development.