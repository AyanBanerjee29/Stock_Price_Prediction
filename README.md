# SAMBA Stock Price Prediction System

This project implements the **SAMBA (State-space Mamba with Graph Neural Networks)** architecture for stock price forecasting.  
It provides a complete pipeline for:

- Downloading financial market data  
- Creating feature-rich datasets  
- Training with hyperparameter tuning  
- Evaluating model performance  
- Training a final production model  
- Predicting future stock prices  

------------------------------------------------------------

## 📋 Table of Contents
1. Installation  
2. Project Structure  
3. Step 1: Create Dataset  
4. Step 2: Training & Development  
5. Step 3: Testing & Evaluation  
6. Step 4: Production Training  
7. Step 5: Future Prediction  

------------------------------------------------------------

## 1. Installation

Ensure you have Python **3.8+** installed.

Steps:

1. Clone the repository or navigate to your project directory.  
2. Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------

## 2. Project Structure

    .
    ├── create_feature_dataset.py   # Script to download & process raw data
    ├── main.py                     # Entry point for Train/Test/Production
    ├── predict_future.py           # Script to forecast the next 7 days
    ├── paper_config.py             # Configuration (from SAMBA research)
    ├── requirements.txt            # Python dependencies
    ├── Dataset/                    # Stores processed CSV datasets
    ├── final_model_outputs/        # Saved models, params, and plots
    ├── models/                     # SAMBA + Mamba model definitions
    ├── trainer/                    # Training loop implementation
    └── utils/                      # Utility tools (metrics, logger, etc.)

------------------------------------------------------------

## 3. Step 1: Create Dataset

This script:

- Downloads stock data from Yahoo Finance  
  (tickers include **^NSEI, ^GSPC, INR=X**, etc.)
- Computes technical indicators such as  
  **RSI, MACD, Bollinger Bands**
- Builds a combined feature dataset

Run:

    python create_feature_dataset.py

Output created:

    Dataset/NIFTY50_features_wide.csv

This dataset contains NIFTY50 market data enriched with global indices and technical indicators.

------------------------------------------------------------

## 4. Step 2: Training & Development

This step performs:

- **Train/Dev split:** 80% Dev, 20% Test  
- **Grid search hyperparameter tuning**  
- **Training a development SAMBA model**

Run:

    python main.py --mode train

The process:

- Loads `Dataset/NIFTY50_features_wide.csv`
- Performs hyperparameter tuning  
- Saves best parameters:

      final_model_outputs/best_params.json

- Trains a model using the best config and saves:

      final_model_outputs/dev_samba_model.pth

------------------------------------------------------------

## 5. Step 3: Testing & Evaluation

Evaluate the previously trained Dev model on the unseen Test split.

Run:

    python main.py --mode test

This step:

- Loads `dev_samba_model.pth`
- Computes key metrics:
  - **MAE**
  - **RMSE**
  - **IC (Information Coefficient)**
- Generates prediction plots and saves them to:

      final_model_outputs/test_plot_day_1.png

------------------------------------------------------------

## 6. Step 4: Production Training

Once the model is validated:

- Load best hyperparameters  
- Train on **100% of the available data**

Run:

    python main.py --mode production

Saves the final production model:

    final_model_outputs/production_samba_model.pth

------------------------------------------------------------

## 7. Step 5: Future Prediction

Predict the next **7 business days** using the production model.

Run:

    python predict_future.py

This script:

- Loads `production_samba_model.pth`
- Loads the complete dataset
- Extracts the **most recent 60-day window**
- Predicts the next 7 days
- Saves results to:

      final_model_outputs/next_week_forecast.csv

------------------------------------------------------------

## ✔ Summary

The SAMBA system provides a fully automated end-to-end forecasting pipeline:

- ✔ Dataset creation  
- ✔ Hyperparameter tuning  
- ✔ Evaluation & visualization  
- ✔ Production training  
- ✔ Future forecasting  

All results, models, plots, and metrics are stored in:

    final_model_outputs/

------------------------------------------------------------
