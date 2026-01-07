"""
Feature Engineering Module
Creates technical indicators for BTC price prediction

Technical Indicators (15 features):
1. RSI (14) - Momentum oscillator
2. MACD & Signal - Trend-following indicator
3. Bollinger Bands (High, Low, Position) - Volatility measure
4. Moving Averages (SMA 20, 50, EMA 12) - Trend indicators
5. ATR - Volatility measure
6. Volume SMA & Ratio - Volume analysis
7. ROC (1h, 24h) - Price momentum
8. Volatility - Price volatility measure
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange


class FeatureEngineer:
    """Create technical indicators and features for ML model"""
    
    def __init__(self):
        self.feature_names = []
        
    def create_features(self, df):
        """
        Create technical indicators - focused on most important features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with features
        """
        df = df.copy()
        
        print("Creating technical indicators...")
        
        # 1. RSI - Relative Strength Index (most popular momentum indicator)
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi_14'] = rsi.rsi()
        
        # 2. MACD - Moving Average Convergence Divergence
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # 3. Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # 4. Moving Averages (key trend indicators)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        
        # 5. ATR - Average True Range (Volatility)
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr'] = atr.average_true_range()
        
        # 6. Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # 7. Price momentum
        df['roc_1'] = df['close'].pct_change(periods=1) * 100
        df['roc_24'] = df['close'].pct_change(periods=24) * 100
        
        # 8. Price volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
        
        print(f"Total features created: 15")
        
        return df
    
    
    def create_target(self, df, horizon=24):
        """
        Create target variable: 1 if price goes up in next 24 hours, 0 otherwise
        
        Logic:
        - Look ahead 24 hours (24 1-hour candles)
        - Compare future close price with current close price
        - If future_close > current_close: label = 1 (UP)
        - If future_close <= current_close: label = 0 (DOWN)
        
        Args:
            df: DataFrame with features
            horizon: Number of hours to look ahead (default 24)
            
        Returns:
            DataFrame with target column
        """
        df = df.copy()
        
        # Future price after 'horizon' hours
        df['future_close'] = df['close'].shift(-horizon)
        
        # Calculate price change
        df['price_change_24h'] = (df['future_close'] - df['close']) / df['close'] * 100
        
        # Binary target: 1 if price goes up, 0 if down
        df['target'] = (df['future_close'] > df['close']).astype(int)
        
        # Remove rows where we don't have future data
        df = df[:-horizon]
        
        print(f"\nTarget Distribution:")
        print(df['target'].value_counts())
        print(f"\nTarget Distribution (%):")
        print(df['target'].value_counts(normalize=True) * 100)
        
        return df
    
    def get_feature_columns(self, df):
        """Get list of feature columns (exclude metadata and target)"""
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades',
                       'future_close', 'price_change_24h', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols


def main():
    """Test feature engineering"""
    from data_fetcher import BinanceDataFetcher
    
    # Load data
    fetcher = BinanceDataFetcher()
    df = fetcher.load_data('data/btc_hourly_5y.csv')
    
    print(f"Original data shape: {df.shape}")
    
    # Create features
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df)
    
    print(f"Data shape after features: {df_features.shape}")
    
    # Create target
    df_final = engineer.create_target(df_features, horizon=24)
    
    print(f"Final data shape: {df_final.shape}")
    
    # Get feature columns
    feature_cols = engineer.get_feature_columns(df_final)
    print(f"\nNumber of features: {len(feature_cols)}")
    print(f"\nFeature columns:")
    for i, col in enumerate(feature_cols, 1):
        print(f"{i}. {col}")
    
    # Check for missing values
    print(f"\nMissing values per column:")
    missing = df_final[feature_cols].isnull().sum()
    print(missing[missing > 0])
    
    # Save processed data
    output_path = 'data/btc_features.csv'
    df_final.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")
    
    return df_final


if __name__ == '__main__':
    main()

