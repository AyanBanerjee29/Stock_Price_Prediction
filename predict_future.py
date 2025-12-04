# -*- coding: utf-8 -*-
"""
Script to predict NEXT WEEK's stock prices using the Production SAMBA Model.
1. Loads 'production_samba_model.pth'
2. Loads the LATEST data from 'Dataset/NIFTY50_features_wide.csv'
3. Grabs the last 60 days
4. Predicts the next 7 days (or configured horizon)
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# --- Import project modules ---
from paper_config import get_paper_config
from models import SAMBA
from utils.data_utils import load_raw_data

# --- Configuration ---
MODEL_PATH = "final_model_outputs/production_samba_model.pth"
PARAMS_PATH = "final_model_outputs/best_params.json"
DATASET_PATH = "Dataset/NIFTY50_features_wide.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_business_days_future(start_date, days):
    """Generates the next 'days' business days (skipping weekends)"""
    future_dates = []
    current_date = start_date
    while len(future_dates) < days:
        current_date += timedelta(days=1)
        # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
        if current_date.weekday() < 5: 
            future_dates.append(current_date)
    return future_dates

def main():
    print("🔮 SAMBA Future Predictor")
    print("=========================")

    # 1. Check Files
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PARAMS_PATH):
        print(f"❌ Error: Model files not found in 'final_model_outputs/'.")
        print("   Please run: python main.py --mode production")
        return

    # 2. Load Configuration & Model Params
    with open(PARAMS_PATH, 'r') as f:
        best_params = json.load(f)
    print(f"✅ Loaded Params: {best_params}")

    model_args, config = get_paper_config()
    
    # 3. Load Data to get Feature Count & Last Sequence
    print(f"   Loading data from {DATASET_PATH}...")
    df_full, price_index = load_raw_data(DATASET_PATH, target_col_name='close')
    
    # Update vocab_size based on actual data features
    num_features = len(df_full.columns)
    model_args.vocab_size = num_features
    
    # 4. Prepare the LAST Sequence (The "Lookback")
    window_size = config.lag
    if len(df_full) < window_size:
        print("❌ Error: Not enough data points for one sequence.")
        return

    # Extract the very last 'window_size' rows
    last_window_df = df_full.iloc[-window_size:]
    last_date = last_window_df.index[-1]
    print(f"📅 Last Data Point Date: {last_date.date()}")
    
    # --- PER-WINDOW SCALING (Crucial Step) ---
    # We must scale this window exactly how we scaled the training windows.
    # 1. Convert to numpy
    raw_seq = last_window_df.to_numpy(dtype=np.float32) # Shape: (60, features)
    
    # 2. Calculate Local Min/Max
    local_min = np.min(raw_seq, axis=0)
    local_max = np.max(raw_seq, axis=0)
    denom = local_max - local_min + 1e-8
    
    # 3. Scale the Input
    scaled_seq = (raw_seq - local_min) / denom
    
    # 4. Convert to Tensor [Batch=1, Seq=60, Feat=38]
    input_tensor = torch.tensor(scaled_seq).unsqueeze(0).to(device)
    
    # 5. Load the Production Model
    print("   Loading Production Model...")
    model = SAMBA(
        model_args, 
        best_params['hid'], 
        config.lag, 
        config.horizon, 
        best_params['embed_dim'], 
        best_params["cheb_k"]
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 6. Predict
    print("🚀 Generating Forecast...")
    with torch.no_grad():
        # Output Shape: [1, Horizon, 1] (Scaled 0-1)
        scaled_prediction = model(input_tensor) 
    
    # 7. Inverse Transform
    # We only care about the Price column (price_index) for the inverse
    target_min = local_min[price_index]
    target_max = local_max[price_index]
    
    scaled_pred_val = scaled_prediction.cpu().numpy().squeeze() # Shape: (Horizon,)
    
    # Real = Scaled * (Max - Min) + Min
    real_prices = scaled_pred_val * (target_max - target_min) + target_min
    
    # 8. Display Results
    future_dates = get_business_days_future(last_date, len(real_prices))
    
    print("\n🔮 PREDICTED PRICES (Next 7 Business Days)")
    print("-" * 45)
    print(f"{'Date':<15} | {'Predicted Close Price (₹)':<25}")
    print("-" * 45)
    
    for date, price in zip(future_dates, real_prices):
        print(f"{date.strftime('%Y-%m-%d'):<15} | ₹ {price:,.2f}")
    print("-" * 45)
    
    # Save to CSV
    output_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': real_prices
    })
    save_path = "final_model_outputs/next_week_forecast.csv"
    output_df.to_csv(save_path, index=False)
    print(f"\n✅ Forecast saved to {save_path}")

if __name__ == "__main__":
    main()
