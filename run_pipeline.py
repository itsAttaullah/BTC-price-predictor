"""
Complete ML Pipeline Execution Script
Runs the entire pipeline from data fetching to model training
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

from data_fetcher import BinanceDataFetcher
from feature_engineering import FeatureEngineer
from model_training import BTCPricePredictor, trading_viability_analysis
import json


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80 + "\n")


def main():
    """Execute complete ML pipeline"""
    
    print_header("BTC PRICE PREDICTION - COMPLETE PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Step 1: Data Fetching
    print_header("STEP 1: DATA FETCHING")
    print("Fetching 5 years of BTC hourly data from Binance API...")
    
    fetcher = BinanceDataFetcher(symbol='BTCUSDT', interval='1h')
    
    # Check if data already exists
    if os.path.exists('data/btc_hourly_5y.csv'):
        print("Data file already exists. Loading from disk...")
        df_raw = fetcher.load_data('data/btc_hourly_5y.csv')
    else:
        print("Fetching fresh data from Binance...")
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        df_raw = fetcher.fetch_data(start_date, end_date)
        fetcher.save_data(df_raw)
    
    print(f"\n[OK] Data loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    print(f"  Date range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
    
    # Step 2: Feature Engineering
    print_header("STEP 2: FEATURE ENGINEERING")
    print("Creating technical indicators...")
    
    engineer = FeatureEngineer()
    
    # Check if features already exist
    if os.path.exists('data/btc_features.csv'):
        print("Feature file already exists. Loading from disk...")
        import pandas as pd
        df_features = pd.read_csv('data/btc_features.csv')
        df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
    else:
        print("Creating features from scratch...")
        df_features = engineer.create_features(df_raw)
        df_features = engineer.create_target(df_features, horizon=24)
        df_features.to_csv('data/btc_features.csv', index=False)
    
    feature_cols = engineer.get_feature_columns(df_features)
    print(f"\n[OK] Features created: {len(feature_cols)} features")
    print(f"  Final dataset: {df_features.shape[0]} rows")
    
    # Step 3: Model Training
    print_header("STEP 3: MODEL TRAINING")
    
    predictor = BTCPricePredictor()
    
    # Prepare data
    X_train, X_test, y_train, y_test, train_dates, test_dates = predictor.prepare_data(
        df_features, test_size=0.2
    )
    
    # Hyperparameter tuning
    print("\nPerforming hyperparameter tuning...")
    best_params = predictor.tune_hyperparameters(X_train, y_train, n_splits=5)
    
    # Train model
    print("\nTraining final model...")
    predictor.train(X_train, y_train, params=best_params)
    
    # Step 4: Evaluation
    print_header("STEP 4: MODEL EVALUATION")
    
    metrics = predictor.evaluate(X_test, y_test, save_plots=True)
    
    # Save metrics
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n[OK] Evaluation complete")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score (UP): {metrics['f1_up']:.4f}")
    print(f"  F1 Score (DOWN): {metrics['f1_down']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Step 5: Trading Viability Analysis
    print_header("STEP 5: TRADING VIABILITY ANALYSIS")
    trading_viability_analysis(metrics)
    
    # Step 6: Save Model
    print_header("STEP 6: SAVING MODEL")
    predictor.save_model('models/btc_xgboost_model.joblib')
    print("[OK] Model saved successfully")
    
    # Summary
    print_header("PIPELINE COMPLETE")
    print("[OK] All steps completed successfully!\n")
    print("Generated files:")
    print("  - data/btc_hourly_5y.csv (raw data)")
    print("  - data/btc_features.csv (processed features)")
    print("  - models/btc_xgboost_model.joblib (trained model)")
    print("  - results/confusion_matrix.png")
    print("  - results/feature_importance.png")
    print("  - results/feature_importance.csv")
    print("  - results/metrics.json")
    print("\nNext steps:")
    print("  1. Review the results in the 'results' folder")
    print("  2. Start the API: python src/api.py")
    print("  3. Test the API: python test_api.py")
    print("  4. Access Swagger UI: http://localhost:8000/docs")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Pipeline failed with exception:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

