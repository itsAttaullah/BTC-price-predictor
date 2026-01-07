"""
Data Fetcher Module
Fetches 5 years of BTC hourly OHLCV data from Binance API
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import os


class BinanceDataFetcher:
    """Fetch historical OHLCV data from Binance API"""
    
    def __init__(self, symbol='BTCUSDT', interval='1h'):
        self.symbol = symbol
        self.interval = interval
        self.base_url = 'https://api.binance.com/api/v3/klines'
        
    def fetch_data(self, start_date, end_date):
        """
        Fetch OHLCV data between start_date and end_date
        
        Args:
            start_date: datetime object or string 'YYYY-MM-DD'
            end_date: datetime object or string 'YYYY-MM-DD'
            
        Returns:
            pandas DataFrame with OHLCV data
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        all_data = []
        current_start = start_date
        
        # Binance API limit is 1000 candles per request
        # For 1h interval, 1000 hours = ~41.6 days
        chunk_size = timedelta(days=40)
        
        print(f"Fetching {self.symbol} data from {start_date} to {end_date}")
        
        with tqdm(total=(end_date - start_date).days) as pbar:
            while current_start < end_date:
                current_end = min(current_start + chunk_size, end_date)
                
                params = {
                    'symbol': self.symbol,
                    'interval': self.interval,
                    'startTime': int(current_start.timestamp() * 1000),
                    'endTime': int(current_end.timestamp() * 1000),
                    'limit': 1000
                }
                
                try:
                    response = requests.get(self.base_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    if data:
                        all_data.extend(data)
                    
                    # Update progress
                    pbar.update((current_end - current_start).days)
                    
                    # Rate limiting - be nice to Binance API
                    time.sleep(0.5)
                    
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching data: {e}")
                    time.sleep(5)  # Wait longer on error
                    continue
                
                current_start = current_end
        
        # Convert to DataFrame
        df = self._process_raw_data(all_data)
        return df
    
    def _process_raw_data(self, raw_data):
        """Convert raw Binance API data to pandas DataFrame"""
        df = pd.DataFrame(raw_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert price and volume columns to float
        price_cols = ['open', 'high', 'low', 'close']
        volume_cols = ['volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']
        
        for col in price_cols + volume_cols:
            df[col] = df[col].astype(float)
        
        df['trades'] = df['trades'].astype(int)
        
        # Keep only essential columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']]
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def save_data(self, df, filepath='data/btc_hourly_5y.csv'):
        """Save DataFrame to CSV"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"\nData saved to {filepath}")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Total hours: {len(df)}")
        
    def load_data(self, filepath='data/btc_hourly_5y.csv'):
        """Load data from CSV"""
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df


def main():
    """Main function to fetch 5 years of BTC data"""
    fetcher = BinanceDataFetcher(symbol='BTCUSDT', interval='1h')
    
    # Calculate date range - 5 years from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years
    
    print("="*60)
    print("BTC HOURLY DATA FETCHER")
    print("="*60)
    print(f"Symbol: BTCUSDT")
    print(f"Interval: 1 hour")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Expected data points: ~{5*365*24} hours")
    print("="*60)
    
    # Fetch data
    df = fetcher.fetch_data(start_date, end_date)
    
    # Save data
    fetcher.save_data(df)
    
    # Display summary statistics
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(df.describe())
    print("\n" + "="*60)
    print("FIRST 5 ROWS")
    print("="*60)
    print(df.head())
    print("\n" + "="*60)
    print("LAST 5 ROWS")
    print("="*60)
    print(df.tail())
    
    # Check for missing values
    print("\n" + "="*60)
    print("MISSING VALUES")
    print("="*60)
    print(df.isnull().sum())
    
    return df


if __name__ == '__main__':
    main()


