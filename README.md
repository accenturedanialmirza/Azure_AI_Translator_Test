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
*   [`check_batch_size.py`](check_batch_size.py): Utility for validating and managing batch sizes for API calls, ensuring efficient data transfer.
*   [`translator_gemini.py`](translator_gemini.py): An alternative translator module, potentially for integrating with Google Gemini API or other translation services.
*   [`translator.py`](translator.py): A foundational translator module, possibly serving as a base for other translator implementations.
*   [`test_azure_sample.ipynb`](test_azure_sample.ipynb): Jupyter notebook for testing Azure AI Translator API samples.
*   [`test_translator.py`](test_translator.py): Unit tests for the translator functionalities.

### Data Directory

*   `data/src/`: Contains source CSV files for processing.
*   `data/prod/`: Stores processed and translated data in Parquet format.

### Spam Detection Module

*   `spam-detection/`: Directory containing assets and notebooks for the spam detection module.
    *   [`comment_hide_classifier.joblib`](spam-detection/comment_hide_classifier.joblib): The serialized machine learning model used for classifying comments.
    *   [`model-category.ipynb`](spam-detection/model-category.ipynb): Jupyter notebook for exploring and categorizing spam detection models.
    *   [`model-pipeline-category.ipynb`](spam-detection/model-pipeline-category.ipynb): Jupyter notebook detailing the pipeline for model training and evaluation.
    *   [`youtube-comment-spam-detection-max-94-89.ipynb`](spam-detection/youtube-comment-spam-detection-max-94-89.ipynb): Jupyter notebook showcasing a spam detection model with a maximum accuracy of 94.89%.
    *   [`youtube-comments-spam-detection-f1-score-96.ipynb`](spam-detection/youtube-comments-spam-detection-f1-score-96.ipynb): Jupyter notebook focusing on a spam detection model achieving an F1-score of 96%.
    *   `catboost_info/`: Directory containing training logs and information for CatBoost models, if used.

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

*   `polars`: For high-performance DataFrame operations and efficient data manipulation.
*   `requests`: For making HTTP requests, primarily to the Azure AI Translator API.
*   `python-dotenv`: For loading environment variables from a `.env` file.
*   `numpy`: Fundamental package for numerical computing in Python.
*   `pandas`: Data manipulation and analysis, often used for data loading and initial processing.
*   `pyarrow`: Provides Python bindings for Apache Arrow, essential for Parquet file handling.
*   `tqdm`: For displaying progress bars during iterative processes.
*   `ipykernel`: IPython Kernel for Jupyter notebooks.
*   `jupyter_client`: Jupyter protocol client.
*   `jupyter_core`: Core utilities for Jupyter.
*   `matplotlib-inline`: Matplotlib backend for inline plots in Jupyter.
*   `mypy`: Optional static type checker for Python.
*   `packaging`: Core utilities for Python packages.
*   `psutil`: Cross-platform library for retrieving process and system utilization.
*   `pygments`: A generic syntax highlighter.
*   `python-dateutil`: Extensions to the standard `datetime` module.
*   `pytz`: World timezone definitions for Python.
*   `pyzmq`: Python bindings for ZeroMQ.
*   `setuptools`: Easily download, build, install, upgrade, and uninstall Python packages.
*   `six`: Python 2 and 3 compatibility utilities.
*   `tornado`: A Python web framework and asynchronous networking library.
*   `traitlets`: A configuration system for Python applications.
*   `urllib3`: A powerful, user-friendly HTTP client for Python.
*   `fastexcel`: For fast Excel file reading.
*   `scikit-learn`: Machine learning library, likely used for the spam detection model.
*   `joblib`: For serializing and deserializing Python objects, used for saving models.
*   `catboost`: Gradient boosting library, potentially used for advanced spam detection models.