# Breast Cancer Detection API (Flask + TensorFlow)

This Flask API predicts breast cancer from images using:
- A simple binary classification model
- A detailed multi-output model

Models are downloaded from Google Drive using `gdown`.

## Endpoints

- `/predict/simple`: Simple Cancer Detection
- `/predict/detailed`: Multi-label detailed analysis

## Setup

```bash
pip install -r requirements.txt
python app.py
