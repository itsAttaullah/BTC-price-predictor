"""
Model Training Module
Trains XGBoost classifier with time-based validation and hyperparameter tuning

Key Principles:
1. Time-based split (NO random shuffling) - maintains temporal order
2. Walk-forward validation for realistic performance estimation
3. Hyperparameter tuning with time-series cross-validation
4. Comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_score, recall_score, f1_score,
    roc_auc_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BTCPricePredictor:
    """XGBoost model for BTC price direction prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.best_params = None
        self.training_history = {}
        
    def prepare_data(self, df, test_size=0.2):
        """
        Prepare data with TIME-BASED split (NO shuffling)
        
        Why time-based split matters:
        - Financial data has temporal dependencies
        - Random split would leak future information into training
        - We must train on past and test on future (like real trading)
        - Prevents overfitting to future patterns that wouldn't be known
        
        Args:
            df: DataFrame with features and target
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test, train_dates, test_dates
        """
        # Remove rows with missing values
        df = df.dropna()
        
        # Get feature columns
        from feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        feature_cols = engineer.get_feature_columns(df)
        
        self.feature_columns = feature_cols
        
        # Sort by timestamp to ensure temporal order
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate split point
        split_idx = int(len(df) * (1 - test_size))
        
        # Time-based split
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Separate features and target
        X_train = train_df[feature_cols]
        y_train = train_df['target']
        X_test = test_df[feature_cols]
        y_test = test_df['target']
        
        train_dates = train_df['timestamp']
        test_dates = test_df['timestamp']
        
        print("="*60)
        print("DATA SPLIT SUMMARY")
        print("="*60)
        print(f"Total samples: {len(df)}")
        print(f"Training samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Testing samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        print(f"\nTraining period: {train_dates.min()} to {train_dates.max()}")
        print(f"Testing period: {test_dates.min()} to {test_dates.max()}")
        print(f"\nNumber of features: {len(feature_cols)}")
        print(f"\nTraining target distribution:")
        print(y_train.value_counts())
        print(f"\nTesting target distribution:")
        print(y_test.value_counts())
        print("="*60)
        
        return X_train, X_test, y_train, y_test, train_dates, test_dates
    
    def tune_hyperparameters(self, X_train, y_train, n_splits=5):
        """
        Hyperparameter tuning with TimeSeriesSplit cross-validation
        
        Tuning approach:
        1. Use TimeSeriesSplit for cross-validation (respects temporal order)
        2. Test different combinations of key hyperparameters:
           - max_depth: Controls tree complexity and overfitting
           - learning_rate: Controls step size in gradient descent
           - n_estimators: Number of boosting rounds
           - min_child_weight: Minimum sum of instance weight in a child
           - subsample: Fraction of samples used for each tree
           - colsample_bytree: Fraction of features used for each tree
        3. Evaluate on validation sets and select best combination
        
        Args:
            X_train: Training features
            y_train: Training target
            n_splits: Number of time-series splits for CV
            
        Returns:
            Best parameters dictionary
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # For efficiency, we'll test a subset of combinations
        # In production, you might want to use GridSearchCV or RandomizedSearchCV
        param_combinations = [
            {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 
             'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 200,
             'min_child_weight': 3, 'subsample': 0.9, 'colsample_bytree': 0.9},
            {'max_depth': 7, 'learning_rate': 0.01, 'n_estimators': 300,
             'min_child_weight': 5, 'subsample': 1.0, 'colsample_bytree': 1.0},
            {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 200,
             'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.9},
            {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 150,
             'min_child_weight': 2, 'subsample': 0.85, 'colsample_bytree': 0.85},
        ]
        
        # TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        best_score = 0
        best_params = None
        results = []
        
        for i, params in enumerate(param_combinations, 1):
            print(f"\nTesting combination {i}/{len(param_combinations)}")
            print(f"Parameters: {params}")
            
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = xgb.XGBClassifier(
                    **params,
                    objective='binary:logistic',
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
                
                model.fit(X_tr, y_tr, verbose=False)
                y_pred = model.predict(X_val)
                
                # Use F1 score as primary metric (balanced for imbalanced classes)
                score = f1_score(y_val, y_pred)
                cv_scores.append(score)
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            print(f"CV F1 Score: {mean_score:.4f} (+/- {std_score:.4f})")
            
            results.append({
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        
        print("\n" + "="*60)
        print("BEST PARAMETERS")
        print("="*60)
        print(f"Best CV F1 Score: {best_score:.4f}")
        print(f"Best Parameters: {best_params}")
        print("="*60)
        
        self.best_params = best_params
        self.tuning_results = results
        
        return best_params
    
    def train(self, X_train, y_train, params=None):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            params: Hyperparameters (if None, uses best_params from tuning)
        """
        if params is None:
            if self.best_params is None:
                # Default parameters if no tuning was done
                params = {
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'n_estimators': 200,
                    'min_child_weight': 3,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9
                }
            else:
                params = self.best_params
        
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        print(f"Parameters: {params}")
        
        self.model = xgb.XGBClassifier(
            **params,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        self.model.fit(X_train, y_train, verbose=True)
        
        print("\nTraining completed!")
        print("="*60)
        
        return self.model
    
    def evaluate(self, X_test, y_test, save_plots=True):
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test target
            save_plots: Whether to save visualization plots
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_0 = precision_score(y_test, y_pred, pos_label=0)
        precision_1 = precision_score(y_test, y_pred, pos_label=1)
        recall_0 = recall_score(y_test, y_pred, pos_label=0)
        recall_1 = recall_score(y_test, y_pred, pos_label=1)
        f1_0 = f1_score(y_test, y_pred, pos_label=0)
        f1_1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['DOWN (0)', 'UP (1)']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Detailed metrics
        metrics = {
            'accuracy': accuracy,
            'precision_down': precision_0,
            'precision_up': precision_1,
            'recall_down': recall_0,
            'recall_up': recall_1,
            'f1_down': f1_0,
            'f1_up': f1_1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }
        
        print("\n" + "="*60)
        print("DETAILED METRICS")
        print("="*60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClass 0 (DOWN):")
        print(f"  Precision: {precision_0:.4f}")
        print(f"  Recall: {recall_0:.4f}")
        print(f"  F1-Score: {f1_0:.4f}")
        print(f"\nClass 1 (UP):")
        print(f"  Precision: {precision_1:.4f}")
        print(f"  Recall: {recall_1:.4f}")
        print(f"  F1-Score: {f1_1:.4f}")
        print(f"\nROC AUC: {roc_auc:.4f}")
        print("="*60)
        
        if save_plots:
            self._plot_confusion_matrix(cm)
            self._plot_feature_importance()
        
        return metrics
    
    def plot_training_progress(self, X_train, y_train, X_test, y_test, train_dates=None, test_dates=None, close_prices=None):
        """Public method to plot training progress"""
        self._plot_training_progress(X_train, y_train, X_test, y_test, train_dates, test_dates, close_prices)
    
    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['DOWN', 'UP'], 
                   yticklabels=['DOWN', 'UP'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("\nConfusion matrix plot saved to results/confusion_matrix.png")
        plt.close()
    
    def _plot_feature_importance(self, top_n=15):
        """Plot feature importance"""
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title('Feature Importance', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('results/feature_importance.png', dpi=150, bbox_inches='tight')
        print("Feature importance plot saved to results/feature_importance.png")
        plt.close()
        
        # Save feature importance to CSV
        feature_importance.to_csv('results/feature_importance.csv', index=False)
        print("Feature importance saved to results/feature_importance.csv")
    
    def _plot_training_progress(self, X_train, y_train, X_test, y_test, train_dates, test_dates, close_prices=None):
        """Plot ONLY the data split showing actual closing prices - ONE CHART ONLY"""
        
        # Split training into train(70%) and validation(10%)
        train_size = int(len(X_train) * 0.875)  # 70% of total
        
        # Get corresponding price data
        train_prices = close_prices[:train_size]
        val_prices = close_prices[train_size:len(X_train)]
        test_prices = close_prices[len(X_train):len(X_train)+len(X_test)]
        
        # Create SINGLE figure - ONLY the data separation chart
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Create time indices
        x_train = range(len(train_prices))
        x_val = range(len(train_prices), len(train_prices) + len(val_prices))
        x_test = range(len(train_prices) + len(val_prices), 
                      len(train_prices) + len(val_prices) + len(test_prices))
        
        # Plot ONLY closing prices - NO predictions
        ax.plot(x_train, train_prices, linewidth=1.5, color='#1f77b4', 
               label='Training data', alpha=0.9)
        ax.plot(x_val, val_prices, linewidth=1.5, color='#ff7f0e',
               label='Validation data', alpha=0.9)
        ax.plot(x_test, test_prices, linewidth=1.5, color='#2ca02c',
               label='Test data', alpha=0.9)
        
        # Add vertical lines at split points
        ax.axvline(x=len(train_prices), color='gray', linestyle='--', 
                  linewidth=2, alpha=0.7)
        ax.axvline(x=len(train_prices) + len(val_prices), color='gray', 
                  linestyle='--', linewidth=2, alpha=0.7)
        
        # Styling
        ax.set_xlabel('Time - Minutes From (UTC+8): 2021-01-01 02:11:00', fontsize=11)
        ax.set_ylabel('Closing Price [USD]', fontsize=11)
        ax.set_title('Data Separation\nClosing Price Through Time', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Format y-axis to show prices cleanly
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        plt.tight_layout()
        plt.savefig('results/training_progress.png', dpi=150, bbox_inches='tight')
        print("Training progress plot saved to results/training_progress.png")
        plt.close()
        
        # Calculate and print split info
        print(f"\nData Split Information:")
        print(f"  Training (70%): {len(train_prices):,} samples")
        print(f"  Validation (10%): {len(val_prices):,} samples")
        print(f"  Testing (20%): {len(test_prices):,} samples")
        print(f"  Total: {len(train_prices) + len(val_prices) + len(test_prices):,} samples")
    
    def save_model(self, filepath='models/btc_xgboost_model.joblib'):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'best_params': self.best_params,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='models/btc_xgboost_model.joblib'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.best_params = model_data['best_params']
        print(f"Model loaded from {filepath}")
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)


def trading_viability_analysis(metrics):
    """
    Analyze if the model would be usable in real trading
    
    Critical considerations:
    1. Accuracy and precision - Can we trust the predictions?
    2. Recall - Are we catching enough opportunities?
    3. Class balance - Is the model biased?
    4. Transaction costs - Would profits exceed trading fees?
    5. Slippage and latency - Real-world execution challenges
    6. Overfitting - Does it generalize to unseen data?
    """
    print("\n" + "="*60)
    print("TRADING VIABILITY ANALYSIS")
    print("="*60)
    
    print("\n1. PREDICTION QUALITY")
    print("-" * 40)
    if metrics['accuracy'] > 0.55:
        print(f"[OK] Accuracy ({metrics['accuracy']:.2%}) is above random (50%)")
    else:
        print(f"[X] Accuracy ({metrics['accuracy']:.2%}) is too close to random")
    
    print("\n2. PRECISION ANALYSIS")
    print("-" * 40)
    print(f"Precision UP: {metrics['precision_up']:.2%}")
    print(f"Precision DOWN: {metrics['precision_down']:.2%}")
    if metrics['precision_up'] > 0.52 and metrics['precision_down'] > 0.52:
        print("[OK] Precision is reasonable for both classes")
    else:
        print("[X] Precision is too low - too many false signals")
    
    print("\n3. RECALL ANALYSIS")
    print("-" * 40)
    print(f"Recall UP: {metrics['recall_up']:.2%}")
    print(f"Recall DOWN: {metrics['recall_down']:.2%}")
    if metrics['recall_up'] > 0.50 and metrics['recall_down'] > 0.50:
        print("[OK] Model catches reasonable portion of movements")
    else:
        print("[X] Model misses too many opportunities")
    
    print("\n4. TRANSACTION COSTS")
    print("-" * 40)
    print("Typical crypto exchange fees: 0.1% - 0.5% per trade")
    print("For profitable trading, edge must exceed ~0.2% per trade")
    print("[!] This model predicts direction only, not magnitude")
    print("[!] Small predicted moves may not cover transaction costs")
    
    print("\n5. REAL-WORLD CHALLENGES")
    print("-" * 40)
    print("[X] Slippage: Price moves between signal and execution")
    print("[X] Latency: Delays in data, prediction, and order placement")
    print("[X] Market impact: Large orders move the market")
    print("[X] Regime changes: Market dynamics shift over time")
    print("[X] Black swan events: Unpredictable market crashes")
    
    print("\n6. OVERFITTING RISK")
    print("-" * 40)
    print("[OK] Used time-based validation (no future data leakage)")
    print("[OK] Tested on completely unseen future data")
    print("[!] Past performance doesn't guarantee future results")
    
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    
    if metrics['accuracy'] > 0.55 and metrics['precision_up'] > 0.52:
        print("[!] MODEL SHOWS PROMISE BUT NOT READY FOR LIVE TRADING")
        print("\nReasons:")
        print("1. Edge is small and may not cover transaction costs")
        print("2. Model predicts direction, not profitable magnitude")
        print("3. No risk management or position sizing strategy")
        print("4. Needs extensive backtesting with realistic costs")
        print("5. Requires continuous monitoring and retraining")
        print("\nRecommendations:")
        print("- Paper trade for 3-6 months to validate performance")
        print("- Implement strict risk management (stop-loss, position sizing)")
        print("- Account for all transaction costs and slippage")
        print("- Monitor model performance and retrain regularly")
        print("- Start with very small positions if going live")
    else:
        print("[X] MODEL NOT SUITABLE FOR LIVE TRADING")
        print("\nReasons:")
        print("1. Accuracy too close to random guessing")
        print("2. Insufficient edge to overcome transaction costs")
        print("3. Would likely lose money after fees")
        print("\nRecommendations:")
        print("- Collect more diverse data")
        print("- Engineer better features")
        print("- Try different model architectures")
        print("- Consider ensemble methods")
        print("- Focus on specific market regimes")
    
    print("="*60)


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print(" "*20 + "BTC PRICE PREDICTION MODEL TRAINING")
    print("="*80)
    
    # Load processed data
    print("\nLoading processed data...")
    df = pd.read_csv('data/btc_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Data loaded: {df.shape}")
    
    # Initialize predictor
    predictor = BTCPricePredictor()
    
    # Prepare data with time-based split
    X_train, X_test, y_train, y_test, train_dates, test_dates = predictor.prepare_data(df, test_size=0.2)
    
    # Hyperparameter tuning
    best_params = predictor.tune_hyperparameters(X_train, y_train, n_splits=5)
    
    # Train model with best parameters
    predictor.train(X_train, y_train, params=best_params)
    
    # Evaluate model
    metrics = predictor.evaluate(X_test, y_test, save_plots=True)
    
    # Plot training progress (pass close prices)
    close_prices = df['close'].values
    predictor.plot_training_progress(X_train, y_train, X_test, y_test, train_dates, test_dates, close_prices)
    
    # Save metrics
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("\nMetrics saved to results/metrics.json")
    
    # Trading viability analysis
    trading_viability_analysis(metrics)
    
    # Save model
    predictor.save_model('models/btc_xgboost_model.joblib')
    
    print("\n" + "="*80)
    print(" "*25 + "TRAINING COMPLETE!")
    print("="*80)
    
    return predictor, metrics


if __name__ == '__main__':
    main()

