# SAMBA Stock Price Prediction System

This project implements the **SAMBA (State-space Mamba with Graph Neural Networks)** architecture for stock price forecasting. It includes a complete pipeline for downloading financial data, training a model with hyperparameter tuning, evaluating performance, and generating future stock price predictions.

## 📋 Table of Contents
1. Installation
2. Project Structure
3. Step 1: Create Dataset
4. Step 2: Training & Development
5. Step 3: Testing & Evaluation
6. Step 4: Production Training
7. Step 5: Future Prediction

---

## 1. Installation

Ensure you have Python installed (3.8+ recommended).

1. Clone the repository (if applicable) or navigate to your project folder.
2. Install dependencies:

pip install -r requirements.txt

---

## 2. Project Structure

.
├── create_feature_dataset.py   # Script to download and process raw data
├── main.py                     # Main entry point for Train/Test/Production
├── predict_future.py           # Script to forecast next 7 days
├── paper_config.py             # Configuration matching the SAMBA paper
├── requirements.txt            # Python dependencies
├── Dataset/                    # Folder where processed CSVs are stored
├── final_model_outputs/        # Folder where models and params are saved
├── models/                     # SAMBA and Mamba model definitions
├── trainer/                    # Training loop implementation
└── utils/                      # Utility functions (metrics, logging, etc.)

---

## 3. Step 1: Create Dataset

This script downloads raw financial data from Yahoo Finance (tickers like ^NSEI, ^GSPC, INR=X, etc.) and computes technical indicators (RSI, MACD, Bollinger Bands).

Run:

python create_feature_dataset.py

Output:
Creates a file at:

Dataset/NIFTY50_features_wide.csv

This dataset contains NIFTY50 data merged with global indices and technical indicators.

---

## 4. Step 2: Training & Development

This phase performs:

- Train/Dev split (80% Dev, 20% Test)
- Hyperparameter tuning
- Dev model training

Run:

python main.py --mode train

What happens:

- Data loaded from Dataset/NIFTY50_features_wide.csv
- Grid search runs for learning rate, hidden_dim, etc.
- Best parameters saved to:

final_model_outputs/best_params.json

- Dev model saved to:

final_model_outputs/dev_samba_model.pth

---

## 5. Step 3: Testing & Evaluation

Evaluate the Dev-trained model on the unseen 20% Test split.

Run:

python main.py --mode test

This step:

- Loads dev_samba_model.pth
- Computes MAE, RMSE, IC (Information Coefficient)
- Saves prediction plots such as:

final_model_outputs/test_plot_day_1.png

---

## 6. Step 4: Production Training

Train the model using 100% of available data.

Run:

python main.py --mode production

This step:

- Loads best hyperparameters found in Step 2
- Trains full SAMBA model on the entire dataset
- Saves production model to:

final_model_outputs/production_samba_model.pth

---

## 7. Step 5: Future Prediction

Predict the next 7 business days using the trained production model.

Run:

python predict_future.py

This script:

- Loads production_samba_model.pth
- Loads the full dataset
- Extracts the latest 60-day window (lookback)
- Predicts the next 7 business days
- Saves results to:

final_model_outputs/next_week_forecast.csv

---

## ✔ Summary

This repository provides a fully automated pipeline:

- Dataset creation
- Hyperparameter tuning
- Model evaluation
- Full-data production training
- Future forecasting

All outputs (models, metrics, plots, forecasts) are stored in final_model_outputs/.

