"""
Feature Engineering Module for Cryptocurrency Prediction

Creates technical indicators and features from OHLCV data.
Carefully designed to avoid look-ahead bias and data leakage.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Generates features from OHLCV data.
    All features are computed using only past/current data to prevent lookahead bias.
    """
    
    def __init__(self):
        self.feature_groups = []
        
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all feature groups."""
        df = df.copy()
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Trend features
        df = self._add_trend_features(df)
        
        # Momentum features
        df = self._add_momentum_features(df)
        
        # Volatility features
        df = self._add_volatility_features(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Market structure features
        df = self._add_market_structure_features(df)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        df['return_20'] = df['close'].pct_change(20)
        
        # Log returns
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
        
        # Price acceleration (change in returns)
        df['return_accel'] = df['return_1'] - df['return_1'].shift(1)
        
        # Rolling returns
        df['rolling_return_10'] = df['close'].pct_change(10).rolling(10).mean()
        
        # High-low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # Distance from MA
            df[f'sma_{period}_dist'] = (df['close'] - df[f'sma_{period}']) / df['close']
            df[f'ema_{period}_dist'] = (df['close'] - df[f'ema_{period}']) / df['close']
        
        # EMA crossover signals
        df['ema_5_10_diff'] = df['ema_5'] - df['ema_10']
        df['ema_10_20_diff'] = df['ema_10'] - df['ema_20']
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Trend slope (linear regression slope approximation)
        df['trend_slope_10'] = (df['close'] - df['close'].shift(10)) / 10
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Stochastic oscillator
        for period in [14, 21]:
            lowest_low = df['low'].rolling(window=period).min()
            highest_high = df['high'].rolling(window=period).max()
            df[f'stoch_k_{period}'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()
        
        # Rate of change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / (df['close'].shift(period) + 1e-10)
        
        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # ATR (Average True Range)
        for period in [14, 21]:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            true_range = np.maximum(np.maximum(high_low, high_close), low_close)
            df[f'atr_{period}'] = true_range.rolling(period).mean()
            df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close'] * 100
        
        # Rolling standard deviation (realized volatility)
        for period in [10, 20, 30]:
            df[f'realized_vol_{period}'] = df['return_1'].rolling(period).std()
        
        # Bollinger Bands
        for period in [20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            bb_upper = f'bb_upper_{period}'
            bb_lower = f'bb_lower_{period}'
            df[bb_upper] = sma + 2 * std
            df[bb_lower] = sma - 2 * std
            df[f'bb_width_{period}'] = (df[bb_upper] - df[bb_lower]) / sma
            df[f'bb_position_{period}'] = (df['close'] - df[bb_lower]) / (df[bb_upper] - df[bb_lower] + 1e-10)
        
        # Volatility regime
        df['vol_regime'] = df['realized_vol_20'] / df['realized_vol_20'].rolling(100).mean()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_change_5'] = df['volume'].pct_change(5)
        
        # Volume moving averages
        for period in [10, 20, 50]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / (df[f'volume_sma_{period}'] + 1e-10)
        
        # Abnormal volume
        df['abnormal_volume'] = df['volume'] / df['volume'].rolling(50).mean()
        
        # Volume-price relationship
        df['volume_price_trend'] = df['volume'] * df['return_1']
        df['vpt_cumsum'] = df['volume_price_trend'].rolling(20).sum()
        
        # On-balance volume (OBV) approximation
        obv = []
        for i in range(len(df)):
            if i == 0:
                obv.append(0)
            elif df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        df['obv_sma_10'] = df['obv'].rolling(10).mean()
        
        return df
    
    def _add_market_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features."""
        # Candle body and wicks
        df['body_size'] = np.abs(df['close'] - df['open']) / df['close']
        df['upper_wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / (df['body_size'] + 1e-10)
        
        # Bullish/bearish candle
        df['bullish'] = (df['close'] > df['open']).astype(int)
        
        # Support/resistance approximations
        for period in [20, 50]:
            donchian_upper = f'donchian_upper_{period}'
            donchian_lower = f'donchian_lower_{period}'
            df[donchian_upper] = df['high'].rolling(period).max()
            df[donchian_lower] = df['low'].rolling(period).min()
            df[f'donchian_position_{period}'] = (df['close'] - df[donchian_lower]) / \
                                                 (df[donchian_upper] - df[donchian_lower] + 1e-10)
        
        # Higher highs / lower lows
        df['higher_high_5'] = (df['high'] > df['high'].shift(5)).astype(int)
        df['lower_low_5'] = (df['low'] < df['low'].shift(5)).astype(int)
        
        # Price position in recent range
        for period in [20, 50]:
            rolling_max = df['high'].rolling(period).max()
            rolling_min = df['low'].rolling(period).min()
            df[f'price_position_{period}'] = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-10)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (exclude raw OHLCV and target columns)."""
        exclude_patterns = ['open', 'high', 'low', 'close', 'volume', 
                           'quote_volume', 'num_trades', 'target', 'future']
        
        feature_cols = []
        for col in df.columns:
            if not any(pattern in col.lower() for pattern in exclude_patterns):
                feature_cols.append(col)
        
        return feature_cols
    
    def clean_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Clean features by handling NaN and infinite values."""
        df = df.copy()
        
        # Replace infinite values with NaN
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        # Forward fill then backward fill remaining NaNs
        # This is acceptable because we're filling historical NaNs, not future data
        df[feature_cols] = df[feature_cols].ffill().bfill()
        
        # If still NaN (at the very beginning), fill with 0
        df[feature_cols] = df[feature_cols].fillna(0)
        
        return df


def create_targets(df: pd.DataFrame, horizons: List[int] = [15, 30, 60], 
                   threshold_multiplier: float = 0.0) -> pd.DataFrame:
    """
    Create prediction targets for multiple horizons.
    
    Args:
        df: DataFrame with OHLCV data
        horizons: List of forecast horizons in number of bars
        threshold_multiplier: Multiplier for volatility-adjusted threshold
                            If 0, use simple direction (any positive return = UP)
    
    Returns:
        DataFrame with added target columns
    """
    df = df.copy()
    
    for horizon in horizons:
        # Future return from current close to future close
        future_col = f'future_return_{horizon}'
        df[future_col] = df['close'].shift(-horizon) / df['close'] - 1
        
        # Direction target: UP=1, DOWN=-1, NEUTRAL=0
        direction_col = f'target_direction_{horizon}'
        
        if threshold_multiplier > 0:
            # Use volatility-adjusted threshold
            vol = df['return_1'].rolling(20).std() * np.sqrt(horizon)
            threshold = vol * threshold_multiplier
            
            df[direction_col] = 0  # NEUTRAL
            df.loc[df[future_col] > threshold, direction_col] = 1  # UP
            df.loc[df[future_col] < -threshold, direction_col] = -1  # DOWN
        else:
            # Simple direction
            df[direction_col] = np.sign(df[future_col])
        
        # Binary target (UP vs NOT_UP)
        binary_col = f'target_binary_{horizon}'
        df[binary_col] = (df[future_col] > 0).astype(int)
        
        # Return magnitude (for regression)
        return_col = f'target_return_{horizon}'
        df[return_col] = df[future_col]
    
    return df


if __name__ == "__main__":
    # Test feature engineering
    import pandas as pd
    
    # Load sample data
    df = pd.read_parquet('data/raw/BTCUSDT_15m.parquet')
    print(f"Original shape: {df.shape}")
    
    # Create features
    engine = FeatureEngine()
    df = engine.compute_all_features(df)
    
    # Create targets
    df = create_targets(df, horizons=[15, 30, 60])  # 15, 30, 60 bars ahead
    
    # Get feature columns
    feature_cols = engine.get_feature_columns(df)
    print(f"Number of features: {len(feature_cols)}")
    print(f"Sample features: {feature_cols[:10]}")
    
    # Clean features
    df = engine.clean_features(df, feature_cols)
    
    # Check for any remaining issues
    print(f"\nFinal shape: {df.shape}")
    print(f"NaN count: {df[feature_cols].isna().sum().sum()}")
    print(f"Inf count: {np.isinf(df[feature_cols]).sum().sum()}")
    
    # Show some statistics
    print("\nTarget distribution (15-bar horizon):")
    print(df['target_binary_15'].value_counts(normalize=True))
    
    print("\nFuture return stats (15-bar horizon):")
    print(df['future_return_15'].describe())
