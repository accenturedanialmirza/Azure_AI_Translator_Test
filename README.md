# Azure AI Translator Project

This project offers a robust and efficient solution for processing and translating text comments using Azure AI Translator services. It leverages the Polars data manipulation library for high-performance operations, making it suitable for large datasets. The pipeline integrates functionalities for language detection, text translation with intelligent mini-batch processing, spam detection, and precise text splitting.

## Project Overview

The core workflow is orchestrated by the [`main.py`](main.py) script and includes the following steps:

1. **Data Ingestion**: Reads input comment data from a specified CSV file.
2. **Language Detection**: Utilizes the `detect_language.py` module to identify the language of each comment. This step is crucial for optimizing translation API calls.
3. **Text Translation**: Dynamically determines an optimal mini-batch size using [`decide_batch_size.py`](modules/decide_batch_size.py) and then translates comments to English (or other specified languages) using Azure AI Translator. The [`modules/translator_df.py`](modules/translator_df.py) and [`modules/translator_multiple_df.py`](modules/translator_multiple_df.py) modules handle this efficiently through a lazy, mini-batch approach, which helps manage API rate limits and memory usage for large volumes of text.
4. **PII Redaction**: Sensitive information (Personally Identifiable Information) within the translated comments is identified and redacted using the [`PII_redactor.py`](PII_redactor.py) module, ensuring data privacy.
5. **Text Splitting**: The [`split_texts.py`](split_texts.py) module is used in two stages: first, to accurately split both original and translated texts into lists of individual sentences within the DataFrame, and then to expand these lists into new rows, creating a detailed sentence-level view while preserving context and order.
6. **Non-Informative Comment Detection**: Translated comments are then passed through a pre-trained machine learning model (from `detect_non_informative.py`) to classify them as non-informative or legitimate.
7. **Data Output**: The processed and enriched data is saved into two Parquet files in the `./data/prod/` directory: `_translated_lazy.parquet` (containing translated comments and non-informative classifications) and `_translated_split_lazy.parquet` (containing individual source and translated sentences). Additionally, an Excel file is generated with the translated comments and other relevant data.

## Setup

To get this project up and running, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone translation
    cd translation
    ```

2. **Create a virtual environment** (highly recommended to manage dependencies):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    All required Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Azure AI Translator**:
    * You need an Azure AI Translator key and endpoint. Obtain these from your Azure portal by creating or accessing an Azure AI Translator resource.
    * Create a new file named `.env` in the root directory of your project.
    * Add your Azure credentials to the `.env` file in the following format:
        ```
        AZURE_TEXT_TRANSLATION_KEY="YOUR_AZURE_TRANSLATOR_KEY"
        AZURE_TEXT_TRANSLATION_ENDPOINT="YOUR_AZURE_TRANSLATOR_ENDPOINT"
        ```
    * **Important**: Ensure the `location` variable within [`modules/translator_df.py`](modules/translator_df.py) and [`modules/translator_multiple_df.py`](modules/translator_multiple_df.py) (around line 20) is updated to match the region of your Azure Translator resource (e.g., "eastus", "westeurope"). This is critical for successful API communication.

## Usage

To execute the full translation and processing pipeline, run the [`main.py`](main.py) script from your project's root directory:

```bash
python main.py
```

Before running, you may need to adjust the input file path within [`main.py`](main.py). By default, it expects a CSV file (e.g., `Infinitas SEP 2023- text comments.csv`) located in the `./data/src/` directory.

Upon successful execution, the script will:
* Read the specified input CSV file.
* Generate intermediate Parquet files in the `./data/temp/` directory (these are cleaned up automatically upon completion).
* Produce final processed Parquet files in the `./data/prod/` directory, containing the translated comments and other relevant data.

## Custom Modules

### [`modules/translator_multiple_df.py`](modules/translator_multiple_df.py)

This module provides an alternative `Translator` class, offering a similar approach to `translator_df.py` but with additional functionalities for handling multiple target languages and more complex translation workflows.

* **`Translator` Class**:
    * **Initialization**: Accepts `input_path` for data handling and `mini_batch_size` for batch processing.
    * **`translate_series(self, s: pl.Series, translate_to_language: List[str] = ['en']) -> Tuple[pl.Series, pl.Series, pl.Series]`**:
        * **Purpose**: Translates a series of text comments using the Azure AI Translator API.
        * **Returns**: A tuple containing the translated text series, source sentence lengths, and translated sentence lengths.
    * **`process_translation_lazy(self, column: str) -> pl.DataFrame`**:
        * **Purpose**: Manages the translation workflow for a specified column, applying `translate_series` directly.
    * **`split_text(self, text: str, lengths: List[int]) -> List[str]`**:
        * **Purpose**: Splits text into sentences based on provided lengths.
    * **`split_sentences_into_rows(self, df: pl.DataFrame, source_split_column: str, translated_split_column: str) -> pl.DataFrame`**:
        * **Purpose**: Expands DataFrame rows for detailed sentence-level analysis, creating a new row for each individual sentence pair.

### [`modules/translator_gemini.py`](modules/translator_gemini.py)

This module provides an alternative `Translator` class, offering a simpler, non-lazy approach to translation for smaller datasets or specific use cases where the full lazy-loading and batching mechanism of `modules/translator_df.py` is not required. It includes basic functionalities for creating DataFrames, translating series, and splitting texts.

* **`Translator` Class**:
    * **Initialization**: Accepts `input_path` for data handling.
    * **`create_df(self) -> pl.DataFrame`**:
        * **Purpose**: Creates a Polars DataFrame from the input data.
    * **`translate_series(self, s: pl.Series, translate_to_language: List[str] = ['en']) -> Tuple[pl.Series, pl.Series, pl.Series]`**:
        * **Purpose**: Translates a series of text comments using the Azure AI Translator API.
        * **Returns**: A tuple containing the translated text series, source sentence lengths, and translated sentence lengths.
    * **`process_translation(self, column: str) -> pl.DataFrame`**:
        * **Purpose**: Manages the translation workflow for a specified column, applying `translate_series` directly.
    * **`split_text(self, text: str, lengths: List[int]) -> List[str]`**:
        * **Purpose**: Splits text into sentences based on provided lengths.
    * **`split_sentences_into_rows(self, df: pl.DataFrame, source_split_column: str, translated_split_column: str) -> pl.DataFrame`**:
        * **Purpose**: Expands DataFrame rows for detailed sentence-level analysis, creating a new row for each individual sentence pair.

This project is modularized into several Python scripts, each encapsulating specific functionalities.

### [`modules/translator_df.py`](modules/translator_df.py)

This module is the heart of the translation process, containing the `Translator` class responsible for efficient interaction with the Azure AI Text Translation API.

* **`Translator` Class**:
    * **Initialization**: Takes `input_path` (path to the source CSV file) and `mini_batch_size` as parameters. The `mini_batch_size` controls how many comments are sent per API request, crucial for managing API limits and optimizing performance.
    * **`translate_series(self, s: pl.Series, translate_to_language: List[str] = ['en']) -> Tuple[pl.Series, pl.Series, pl.Series, pl.Series, pl.Series]`**:
            * **Purpose**: Translates a Polars Series of texts using the Azure Text Translation API.
            * **Parameters**:
                * `s`: A Polars Series containing the text comments to be translated.
                * `translate_to_language`: A list of target language codes (default is `['en']` for English).
            * **Functionality**: Returns the translated text as a Polars Series, along with the original and translated sentence lengths (useful for text splitting), and the detected source language and its confidence score.
    * **`process_translation_lazy(self, column: str) -> pl.DataFrame`**:
            * **Purpose**: Manages the end-to-end translation workflow for a specified text column using Polars LazyFrames. This lazy approach is vital for handling datasets that exceed available memory.
            * **Parameters**:
                * `column`: The name of the column in the DataFrame that contains the text comments to be translated.
            * **Functionality**:
                * **Mini-Batch Processing**: Explicitly slices the LazyFrame into mini-batches. This strategy prevents API throttling and manages memory by processing data in manageable chunks.
                * **Conditional Translation**: Intelligently filters out rows that do not require translation (e.g., comments already detected as English if the target language is English, or comments whose language is 'unknown').
                * **Intermediate Storage**: Saves each processed mini-batch as a temporary Parquet file in the `./data/temp/` directory. This acts as a checkpointing mechanism, leveraging [`modules/check_batch_size.py`](modules/check_batch_size.py) to allow the process to resume or recover from interruptions without re-processing already completed batches, and to ensure temporary files are consistent with the current batch size.
                * **Final Concatenation**: After all batches are processed, it concatenates all intermediate Parquet files into a single, final Polars DataFrame, which is then returned.

### [`detect_language.py`](detect_language.py)

This module is dedicated to identifying the natural language of text comments using the high-performance `lingua-py` library.

* **`_detect_language_iso(text: str) -> str`**:
    * **Purpose**: An internal helper function to detect the language of a single text string.
    * **Parameters**:
        * `text`: The input text string.
    * **Returns**: The ISO 639-1 language code (e.g., "en" for English, "id" for Indonesian).
    * **Functionality**: Uses a pre-configured `LanguageDetectorBuilder` to efficiently detect languages from a predefined list, including English, French, German, Spanish, Italian, Chinese, Japanese, Portuguese, Indonesian, Thai, and Malay.
* **`df_language_verified(df: pl.LazyFrame) -> pl.LazyFrame`**:
    * **Purpose**: Applies language detection across a Polars LazyFrame.
    * **Parameters**:
        * `df`: The input Polars LazyFrame containing a `comments` column.
    * **Returns**: A new Polars LazyFrame with added language information.
    * **Functionality**:
        * Adds a new column named `comments_language_id` by applying the `_detect_language_iso` function to each entry in the `comments` column.

### [`detect_non_informative.py`](detect_non_informative.py)

This module is responsible for classifying comments as non-informative or not, utilizing a pre-trained machine learning model.

* **`predict_non_informative_comment(comments: str, sentiment_category: str) -> bool`**:
    * **Purpose**: Predicts whether a given comment should be classified as non-informative.
    * **Parameters**:
        * `comments`: The text of the comment to be classified.
        * `sentiment_category`: The sentiment category associated with the comment (used as a feature by the model).
    * **Returns**: A boolean value (`True` if the comment is predicted as non-informative and should be hidden, `False` otherwise).
    * **Functionality**: Loads a pre-trained `comment_hide_classifier.joblib` model (expected to be in the `spam-detection/` directory) and uses it to make a prediction based on the comment text and its sentiment category.

### [`split_texts.py`](split_texts.py)

This module provides essential utility functions for accurately splitting long texts into individual sentences, leveraging length information often provided by translation APIs.

* **`split_text(text: str, lengths: List[int]) -> List[str]`**:
    * **Purpose**: Splits a single text string into a list of sentences.
    * **Parameters**:
        * `text`: The complete text string to be split.
        * `lengths`: A list of integers, where each integer represents the length of a sentence within the `text`. This information is typically obtained from the translation API's sentence length breakdown.
    * **Returns**: A list of strings, where each string is a segmented sentence.
    * **Functionality**: Iterates through the `lengths` list, segmenting the `text` accordingly. It attempts to split at natural word boundaries (spaces) to avoid breaking words, but will force a split if no space is found within a segment to ensure all sentences are correctly extracted based on the provided lengths.
* **`split_sentences_into_rows(df: pl.DataFrame, source_split_column: str, translated_split_column: str) -> pl.DataFrame`**:
    * **Purpose**: Transforms a DataFrame by expanding rows, creating a new row for each individual sentence pair.
    * **Parameters**:
        * `df`: The input Polars DataFrame, expected to have columns containing lists of source and translated sentences.
        * `source_split_column`: The name of the column in `df` that contains lists of source sentences.
        * `translated_split_column`: The name of the column in `df` that contains lists of translated sentences.
    * **Returns**: A new Polars DataFrame where each original row has been expanded into multiple rows, one for each sentence pair.
    * **Functionality**: This function is crucial for detailed sentence-level analysis. The output DataFrame will typically include columns such as `respondent_id` (from the original comment), `sentence_index` (the index of the sentence within its original comment), `source_text` (the original sentence), and `translated_text` (the corresponding translated sentence).

### [`check_batch_size.py`](check_batch_size.py)

This module provides helper functions primarily for managing and verifying temporary batch files generated during the translation process, enhancing the robustness of the pipeline.

* **`check_temp_batch_size_matches(path: str, batch_size: int) -> bool`**:
    * **Purpose**: Verifies if a previously processed temporary batch file matches the expected batch size.
    * **Parameters**:
        * `path`: The directory path where temporary Parquet files are stored (e.g., `./data/temp/`).
        * `batch_size`: The expected number of rows (comments) in a batch.
    * **Returns**: `True` if the first Parquet file found in the specified `path` has a row count matching the `batch_size`, `False` otherwise.
    * **Functionality**: This function is used to determine if a specific batch has been successfully processed and saved in a previous run. This allows the main translation process to skip already completed batches, making the pipeline more resilient to interruptions and efficient for large datasets.
* **`remove_temp_files(path: str) -> None`**:
    * **Purpose**: Cleans up temporary Parquet files.
    * **Parameters**:
        * `path`: The directory path from which to remove files (e.g., `./data/temp/`).
    * **Functionality**: Deletes all files ending with `.parquet` within the specified directory. This is typically called after the entire translation process is complete to free up disk space, or when a new run with different parameters (like a new batch size) is initiated, requiring a fresh start.

### [`PII_redactor.py`](PII_redactor.py)

This module is responsible for identifying and redacting Personally Identifiable Information (PII) from text using the `spaCy` library and custom entity recognition rules.

* **`redact_pii(text: str) -> str`**:
    * **Purpose**: Redacts various types of PII from a given text string.
    * **Parameters**:
        * `text`: The input text string from which PII needs to be redacted.
    * **Returns**: A new string with identified PII replaced by `[REDACTED]`.
    * **Functionality**:
        * Loads a `spaCy` English model (`en_core_web_sm`).
        * Integrates custom `entity_ruler` patterns to detect specific entities like Social Security Numbers (SSN) and GENDER (male, female, etc.) using regular expressions.
        * Incorporates `date_spacy` to identify and redact date entities.
        * Iterates through detected entities (including standard spaCy NER labels like PERSON, ORG, GPE, DATE, etc., and custom ones) and replaces their text with `[REDACTED]`.

## Error Handling and Robustness

The project incorporates several features to ensure robustness and efficient handling of large datasets:
* **Mini-Batch Processing**: Prevents API rate limit issues and manages memory by processing data in smaller, controlled chunks.
* **Intermediate File Storage**: Saving temporary batches to disk (`./data/temp/`) acts as a checkpointing mechanism, allowing the process to resume from the last completed batch in case of interruptions.
* **LazyFrame Operations**: Utilizing Polars LazyFrames ensures that data is processed efficiently without loading the entire dataset into memory, which is critical for very large inputs.

## Future Enhancements

* **Support for Multiple Target Languages**: Extend the `Translator` class to easily support translation into multiple languages simultaneously in a single run.
* **Configurability**: Externalize more parameters (e.g., input/output paths, column names, supported languages for detection) into a configuration file (e.g., YAML or JSON) for easier customization without code modification.
* **Performance Monitoring**: Integrate logging and metrics to monitor API call performance, processing times, and resource utilization.
* **Advanced Non-Informative Comment Detection**: Explore integrating more sophisticated models or real-time feedback loops for non-informative comment detection.
* **Dockerization**: Provide a Dockerfile for easy containerization and deployment of the application.