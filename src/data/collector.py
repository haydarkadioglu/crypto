# Cryptocurrency Data Collector
# Fetches historical OHLCV data from Binance public API

import requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceDataCollector:
    """
    Collects historical cryptocurrency data from Binance public API.
    No API key required for public klines endpoint.
    """
    
    BASE_URL = "https://api.binance.com"
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_klines(
        self, 
        symbol: str, 
        interval: str, 
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch klines/candlestick data from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 5m, 15m, 30m, 1h, etc.)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of candles per request (max 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        all_klines = []
        
        # If no start_time provided, get recent data
        if start_time is None:
            end_ts = int(time.time() * 1000) if end_time is None else end_time
            # Go back ~1000 candles from end
            if interval.endswith('m'):
                minutes = int(interval.replace('m', ''))
                lookback = minutes * limit * 1000
            elif interval.endswith('h'):
                hours = int(interval.replace('h', ''))
                lookback = hours * 60 * limit * 1000
            else:
                lookback = 86400000 * limit  # daily
            start_time = end_ts - lookback
        
        current_start = start_time
        
        while True:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'limit': limit
            }
            
            if end_time:
                params['endTime'] = end_time
            
            try:
                response = requests.get(
                    f"{self.BASE_URL}/api/v3/klines",
                    params=params,
                    timeout=30
                )
                
                if response.status_code != 200:
                    logger.warning(f"API error: {response.status_code}, {response.text}")
                    time.sleep(1)
                    continue
                    
                klines = response.json()
                
                if not klines:
                    break
                    
                all_klines.extend(klines)
                
                # Update start time for next iteration
                last_kline_end = klines[-1][6]  # Close time
                current_start = last_kline_end + 1
                
                # Stop if we've reached end_time or got fewer than limit
                if len(klines) < limit:
                    break
                    
                # Check if we've passed end_time
                if end_time and current_start > end_time:
                    break
                    
                # Rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                time.sleep(2)
                continue
        
        return self._process_klines(all_klines)
    
    def _process_klines(self, klines: List) -> pd.DataFrame:
        """Convert raw klines to DataFrame."""
        if not klines:
            return pd.DataFrame()
            
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'num_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'quote_volume', 'num_trades', 'taker_buy_base', 'taker_buy_quote']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df[['open', 'high', 'low', 'close', 'volume', 
                   'quote_volume', 'num_trades']]
    
    def fetch_multiple_intervals(
        self, 
        symbol: str, 
        intervals: List[str],
        days_back: int = 90
    ) -> dict:
        """Fetch data for multiple intervals."""
        end_time = int(time.time() * 1000)
        start_time = end_time - (days_back * 24 * 60 * 60 * 1000)
        
        data = {}
        for interval in intervals:
            logger.info(f"Fetching {symbol} {interval} data...")
            df = self.fetch_klines(symbol, interval, start_time, end_time)
            if len(df) > 0:
                data[interval] = df
                # Save to disk
                filepath = self.data_dir / f"{symbol}_{interval}.parquet"
                df.to_parquet(filepath)
                logger.info(f"Saved {len(df)} rows to {filepath}")
            time.sleep(0.5)
        
        return data
    
    def load_from_disk(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Load previously saved data."""
        filepath = self.data_dir / f"{symbol}_{interval}.parquet"
        if filepath.exists():
            return pd.read_parquet(filepath)
        return None


def main():
    """Main entry point for data collection."""
    collector = BinanceDataCollector()
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    intervals = ['1m', '5m', '15m', '30m', '1h']
    
    for symbol in symbols:
        logger.info(f"Collecting data for {symbol}")
        try:
            data = collector.fetch_multiple_intervals(symbol, intervals, days_back=180)
            for interval, df in data.items():
                logger.info(f"  {interval}: {len(df)} candles, "
                           f"from {df.index[0]} to {df.index[-1]}")
        except Exception as e:
            logger.error(f"Failed to collect {symbol}: {e}")
        time.sleep(1)
    
    logger.info("Data collection complete!")


if __name__ == "__main__":
    main()
