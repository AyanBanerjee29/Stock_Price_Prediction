# -*- coding: utf-8 -*-
"""
Main script for SAMBA stock price forecasting
Modes:
1. 'train':      Split 80/20. Tune Hypers. Train on 80%. Save 'best_params.json' & 'dev_model.pth'.
2. 'test':       Load 'dev_model.pth'. Evaluate on 20% Test Set.
3. 'production': Load 'best_params.json'. TRAIN ON 100% DATA. Save 'production_model.pth'.
"""
import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

# --- Import project modules ---
from paper_config import get_paper_config, get_dataset_info
from models import SAMBA
from utils import (
    init_seed, pearson_correlation, rank_information_coefficient, All_Metrics, get_logger
)
from utils.data_utils import load_raw_data, create_per_window_sequences, data_loader
from trainer import Trainer

# --- GPU/CPU Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_loader_with_inverse(model, loader):
    """
    Custom evaluation loop that handles Per-Window Inverse Transformation.
    """
    model.eval()
    y_pred_real_list = []
    y_true_real_list = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            data = batch[0].to(device)
            target = batch[1].to(device)
            mm = batch[2].to(device)

            output = model(data) 
            
            # Inverse Transform: Real = Scaled * (Max - Min) + Min
            batch_min = mm[:, 0].view(-1, 1, 1) 
            batch_max = mm[:, 1].view(-1, 1, 1) 
            
            pred_real = output * (batch_max - batch_min) + batch_min
            target_real = target * (batch_max - batch_min) + batch_min
            
            y_pred_real_list.append(pred_real.cpu())
            y_true_real_list.append(target_real.cpu())
            
    y_p = torch.cat(y_pred_real_list, dim=0)
    y_t = torch.cat(y_true_real_list, dim=0)
    
    return y_p, y_t

def main(cli_args):
    # --- 1. Initial Setup ---
    model_args, config = get_paper_config()
    init_seed(config.seed)
    
    output_dir = "final_model_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # File paths
    best_params_file = os.path.join(output_dir, "best_params.json")
    dev_model_file = os.path.join(output_dir, "dev_samba_model.pth")       # 80% Data Model
    prod_model_file = os.path.join(output_dir, "production_samba_model.pth") # 100% Data Model
    
    print("🚀 SAMBA: Stock Prediction System")
    print(f"Using device: {device}")
    print(f"Current Mode: {cli_args.mode.upper()}")
    
    # --- 2. Load Data ---
    print("\n>> Loading and Windowing Data...")
    dataset_file = "Dataset/NIFTY50_features_wide.csv"
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset {dataset_file} not found! Run create_feature_dataset.py first.")
        return

    df_full, price_index = load_raw_data(dataset_file, target_col_name='close')
    num_features = len(df_full.columns)
    model_args.vocab_size = num_features
    config.num_nodes = num_features
    
    window = config.lag
    predict = config.horizon
    XX_all, YY_all, MM_all = create_per_window_sequences(df_full, window, predict, price_index)
    
    # --- 3. Split Logic ---
    total_samples = len(XX_all)
    
    # If Production: Use 100% for Training. 
    # If Train/Test: Use 80% Dev, 20% Test.
    
    test_split_ratio = 0.10
    test_size = int(total_samples * test_split_ratio)
    dev_size = total_samples - test_size
    
    XX_dev = XX_all[:dev_size]
    YY_dev = YY_all[:dev_size]
    MM_dev = MM_all[:dev_size]
    
    XX_test = XX_all[dev_size:]
    YY_test = YY_all[dev_size:]
    MM_test = MM_all[dev_size:]
    
    print(f"Total sequences available: {total_samples}")

    args = config.to_dict()

    # ==========================================
    # MODE: TRAIN (Hyperparameter Tuning + 80% Train)
    # ==========================================
    if cli_args.mode == 'train':
        print("\n===== HYPERPARAMETER TUNING (on 80% Dev Set) =====")

        param_grid = {
            'lr_init': [0.001, 0.0005],
            'hid': [32, 64],
            'embed_dim': [10],
            'cheb_k': [2, 3]
        }
        
        param_list = list(ParameterGrid(param_grid))
        results = []
        
        for i, params in enumerate(param_list):
            print(f"\n--- Tuning Run {i+1}/{len(param_list)}: {params} ---")
            fold_scores = []
            tscv = TimeSeriesSplit(n_splits=3)
            
            for fold, (train_index, val_index) in enumerate(tscv.split(XX_dev)):
                X_train_fold, y_train_fold, mm_train_fold = XX_dev[train_index], YY_dev[train_index], MM_dev[train_index]
                X_val_fold, y_val_fold, mm_val_fold = XX_dev[val_index], YY_dev[val_index], MM_dev[val_index]

                # Inner split for early stopping
                inner_split = int(len(X_train_fold) * 0.85)
                if inner_split < 10: continue

                X_t_in, y_t_in, mm_t_in = X_train_fold[:inner_split], y_train_fold[:inner_split], mm_train_fold[:inner_split]
                X_v_in, y_v_in, mm_v_in = X_train_fold[inner_split:], y_train_fold[inner_split:], mm_train_fold[inner_split:]

                train_loader = data_loader(X_t_in, y_t_in, mm_t_in, 64, shuffle=True)
                val_loader_in = data_loader(X_v_in, y_v_in, mm_v_in, 64, shuffle=False)
                fold_val_loader = data_loader(X_val_fold, y_val_fold, mm_val_fold, 64, shuffle=False)

                model = SAMBA(model_args, params['hid'], window, predict, params['embed_dim'], params["cheb_k"]).to(device)
                for p in model.parameters():
                    if p.dim() > 1: nn.init.xavier_uniform_(p)
                    else: nn.init.uniform_(p)

                loss_fn = torch.nn.MSELoss().to(device)
                optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr_init'])
                
                trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader_in, args)
                best_model_state, _ = trainer.train()
                
                # Evaluate
                model.load_state_dict(best_model_state)
                y_p_real, y_t_real = evaluate_loader_with_inverse(model, fold_val_loader)
                mae, _, _ = All_Metrics(y_p_real, y_t_real, None, None)
                fold_scores.append(mae.item())
                
                del model, trainer, optimizer
                torch.cuda.empty_cache()

            if fold_scores:
                avg_score = np.mean(fold_scores)
                print(f"--- Avg. MAE: {avg_score:.4f} ---")
                results.append({'params': params, 'score': avg_score})

        if not results: return

        best_result = min(results, key=lambda x: x['score'])
        best_params = best_result['params']
        
        print(f"\n✅ Best Params Found: {best_params}")
        with open(best_params_file, 'w') as f:
            json.dump(best_params, f)

        # Train "Dev" Model (80% Data) for Validation Purposes
        print("\n>> Training Validation Model (80% Data)...")
        
        # Split Dev into Train/Val for Early Stopping
        final_split = int(len(XX_dev) * 0.85)
        X_train_final = XX_dev[:final_split]
        y_train_final = YY_dev[:final_split]
        mm_train_final = MM_dev[:final_split]
        X_val_final = XX_dev[final_split:]
        y_val_final = YY_dev[final_split:]
        mm_val_final = MM_dev[final_split:]
        
        t_loader = data_loader(X_train_final, y_train_final, mm_train_final, 64, shuffle=True)
        v_loader = data_loader(X_val_final, y_val_final, mm_val_final, 64, shuffle=False)
        
        final_model = SAMBA(model_args, best_params['hid'], window, predict, best_params['embed_dim'], best_params["cheb_k"]).to(device)
        for p in final_model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.uniform_(p)

        optimizer = torch.optim.Adam(params=final_model.parameters(), lr=best_params['lr_init'])
        loss_fn = torch.nn.MSELoss().to(device)
        
        final_trainer = Trainer(final_model, loss_fn, optimizer, t_loader, v_loader, args)
        best_state, _ = final_trainer.train()
        torch.save(best_state, dev_model_file)
        print(f"✅ Validation Model saved to {dev_model_file}")
        print("Next Step: Run '--mode test' to see accuracy, or '--mode production' to train for future.")

    # ==========================================
    # MODE: TEST (Evaluate 80% Model on 20% Data)
    # ==========================================
    elif cli_args.mode == 'test':
        print("\n===== EVALUATING ON 20% TEST SET =====")
        if not os.path.exists(dev_model_file):
            print("❌ No trained validation model found. Run '--mode train' first.")
            return
        
        with open(best_params_file, 'r') as f: best_params = json.load(f)
        
        model = SAMBA(model_args, best_params['hid'], window, predict, best_params['embed_dim'], best_params["cheb_k"]).to(device)
        model.load_state_dict(torch.load(dev_model_file, map_location=device))
        
        test_loader = data_loader(XX_test, YY_test, MM_test, 64, shuffle=False)
        y_p_real, y_t_real = evaluate_loader_with_inverse(model, test_loader)
        
        mae, rmse, _ = All_Metrics(y_p_real, y_t_real, None, None)
        ic = pearson_correlation(y_t_real, y_p_real)
        
        print(f"\n📊 Test Set Metrics:")
        print(f"MAE:  {mae.item():.4f}")
        print(f"RMSE: {rmse.item():.4f}")
        print(f"IC:   {ic.item():.4f}")
        
        # Plotting
        y_t_plot = y_t_real.squeeze().numpy()
        y_p_plot = y_p_real.squeeze().numpy()
        if y_t_plot.ndim == 1: y_t_plot = y_t_plot.reshape(-1, 1)
        if y_p_plot.ndim == 1: y_p_plot = y_p_plot.reshape(-1, 1)

        for i in range(config.horizon):
            plt.figure(figsize=(12, 6))
            plt.plot(y_t_plot[:, i], label='Actual', alpha=0.7)
            plt.plot(y_p_plot[:, i], label='Predicted', alpha=0.7)
            plt.title(f"Test Set Evaluation (Day {i+1})")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"test_plot_day_{i+1}.png"))
            plt.close()
        print(f"Plots saved to {output_dir}")

    # ==========================================
    # MODE: PRODUCTION (Train on 100% Data)
    # ==========================================
    elif cli_args.mode == 'production':
        print("\n===== TRAINING PRODUCTION MODEL (100% DATA) =====")
        if not os.path.exists(best_params_file):
            print("❌ No best params found. Run '--mode train' first to optimize parameters.")
            return
            
        with open(best_params_file, 'r') as f: best_params = json.load(f)
        print(f"Using Best Params: {best_params}")
        
        # Use ALL data for training
        # We still create a small "validation" set just to satisfy the Trainer class (usually last 5% or so)
        # But effectively, the model sees almost everything.
        val_split_idx = int(len(XX_all) * 0.95)
        
        X_prod_train = XX_all[:val_split_idx]
        y_prod_train = YY_all[:val_split_idx]
        mm_prod_train = MM_all[:val_split_idx]
        
        X_prod_val = XX_all[val_split_idx:] # Tiny validation just to monitor loss
        y_prod_val = YY_all[val_split_idx:]
        mm_prod_val = MM_all[val_split_idx:]
        
        train_loader = data_loader(X_prod_train, y_prod_train, mm_prod_train, 64, shuffle=True)
        val_loader = data_loader(X_prod_val, y_prod_val, mm_prod_val, 64, shuffle=False)
        
        model = SAMBA(model_args, best_params['hid'], window, predict, best_params['embed_dim'], best_params["cheb_k"]).to(device)
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            else: nn.init.uniform_(p)
            
        optimizer = torch.optim.Adam(params=model.parameters(), lr=best_params['lr_init'])
        loss_fn = torch.nn.MSELoss().to(device)
        
        # Optional: Increase epochs for production or keep same
        trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader, args)
        best_state, _ = trainer.train()
        
        torch.save(best_state, prod_model_file)
        print(f"\n✅ PRODUCTION MODEL SAVED: {prod_model_file}")
        print("⚠️ This model is trained on all available data.")
        print("⚠️ Use this model file to predict next week's prices.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'production'])
    cli_args = parser.parse_args()
    main(cli_args)
