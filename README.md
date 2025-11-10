# Azure AI Translator Project

Lightweight pipeline for language detection, translation, PII redaction and comment-quality classification using Azure AI Translator and Polars.

Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running the pipeline](#running-the-pipeline)
- [Modules](#modules)
- [Outputs](#outputs)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Future work](#future-work)

## Overview

The core entrypoint is [`main.py`](main.py:1). The pipeline:
1. ingest CSV input
2. detect language with [`language_detection/detect_language.py`](language_detection/detect_language.py:1)
3. translate using modules in [`modules/`](modules/:1) (lazy mini-batching to handle large datasets)
4. redact PII with [`PII_redactor.py`](PII_redactor.py:1)
5. split sentences with [`modules/split_texts.py`](modules/split_texts.py:1)
6. detect non-informative comments with [`detect_non_informative.py`](detect_non_informative.py:1)

## Features

- Scales to large datasets using Polars LazyFrames and mini-batch translation
- Checkpointing via temporary Parquet batches (`./data/temp/`)
- PII redaction powered by spaCy and custom rules
- Sentence-level splitting and expanded outputs
- Non-informative comment classification using a pre-trained joblib model

## Quick Start

1. Clone repository:
```bash
git clone <repo-url>
cd Azure_AI_Translator_Test
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with:
```
AZURE_TEXT_TRANSLATION_KEY="YOUR_AZURE_TRANSLATOR_KEY"
AZURE_TEXT_TRANSLATION_ENDPOINT="YOUR_AZURE_TRANSLATOR_ENDPOINT"
```
Update the `location` variable in [`modules/translator_df.py`](modules/translator_df.py:1) and [`modules/translator_multiple_df.py`](modules/translator_multiple_df.py:1) to match your resource region (e.g., "eastus").

Optional files:
- Pretrained classifier: [`spam-detection/comment_hide_classifier.joblib`](spam-detection/comment_hide_classifier.joblib:1)

## Running the pipeline

Run:
```bash
python main.py
```
By default the script expects a source CSV in `./data/src/`. Edit [`main.py`](main.py:1) to change input/output paths or parameters.

## Modules (short reference)

- [`modules/translator_df.py`](modules/translator_df.py:1): Lazy, mini-batch Translator (recommended for large datasets)
- [`modules/translator_multiple_df.py`](modules/translator_multiple_df.py:1): Multi-target language support
- [`modules/translator_gemini.py`](modules/translator_gemini.py:1): Simpler non-lazy translator for small datasets
- [`language_detection/detect_language.py`](language_detection/detect_language.py:1): Language detection using lingua-py
- [`PII_redactor.py`](PII_redactor.py:1): PII detection and redaction using spaCy
- [`detect_non_informative.py`](detect_non_informative.py:1): Loads joblib classifier to flag non-informative comments
- [`modules/check_batch_size.py`](modules/check_batch_size.py:1): Temp-file checkpoint utilities

## Outputs

- Temporary batches: `./data/temp/`
- Final Parquet outputs: `./data/prod/_translated_lazy.parquet` and `_translated_split_lazy.parquet`
- Excel export: generated alongside Parquet outputs

## Testing

- Unit tests: [`test_translator.py`](test_translator.py:1)
- Example model files in `spam-detection/`

## Troubleshooting

- API errors: confirm keys, endpoint and `location` region match.
- Restarting with a different batch size: remove `./data/temp/` or use [`modules/check_batch_size.py`](modules/check_batch_size.py:1) helpers.
- spaCy model errors: install `en_core_web_sm` via `python -m spacy download en_core_web_sm`

## Future work

- Add a Dockerfile for containerized runs
- Add logging/metrics and CI tests

## License

MIT