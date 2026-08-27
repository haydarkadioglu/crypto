"""
Backtesting Module

Implements realistic walk-forward backtesting with:
- Trading fees
- Slippage
- Signal delay
- Position sizing
- Performance metrics (Sharpe, Sortino, Drawdown, etc.)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeConfig:
    """Configuration for backtesting."""
    fee_rate: float = 0.001  # 0.1% per trade (10 bps)
    slippage_rate: float = 0.0005  # 0.05% slippage
    position_size: float = 1.0  # Fraction of capital to use
    min_signal_strength: float = 0.55  # Minimum probability to trade
    stop_loss: Optional[float] = None  # Stop loss percentage
    take_profit: Optional[float] = None  # Take profit percentage
    holding_period: int = 15  # Number of bars to hold


@dataclass 
class BacktestResult:
    """Results from backtesting."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    expectancy: float
    calmar_ratio: float
    equity_curve: pd.Series
    trades: pd.DataFrame


class Backtester:
    """
    Realistic backtesting engine for cryptocurrency predictions.
    
    Accounts for:
    - Trading fees
    - Slippage
    - Signal delay
    - Realistic execution prices
    """
    
    def __init__(self, config: TradeConfig = None):
        self.config = config or TradeConfig()
        self.results = None
        
    def run(self, df: pd.DataFrame, 
            proba_col: str,
            horizon: int = 15,
            initial_capital: float = 100000.0) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV, features, predictions and actual outcomes
            proba_col: Column name for predicted probability of UP
            horizon: Prediction horizon in bars
            initial_capital: Starting capital
            
        Returns:
            BacktestResult with performance metrics
        """
        df = df.copy()
        
        # Remove rows with NaN in critical columns
        required_cols = ['close', proba_col, f'future_return_{horizon}']
        df = df.dropna(subset=required_cols)
        df = df.reset_index(drop=True)
        
        if len(df) == 0:
            logger.warning("No valid data for backtesting")
            return self._empty_result()
        
        # Generate signals
        df['signal'] = (df[proba_col] >= self.config.min_signal_strength).astype(int)
        df['signal'] = df['signal'].shift(1)  # Signal executed on next bar (realistic delay)
        
        # Track positions and PnL
        n_rows = len(df)
        position = np.zeros(n_rows)  # 1 = long, 0 = flat
        entry_price = np.zeros(n_rows)
        pnl = np.zeros(n_rows)
        trades = []
        
        in_position = False
        current_position = 0
        current_entry = 0
        
        for i in range(1, n_rows):
            prev_signal = df['signal'].iloc[i-1] if i > 0 else 0
            current_signal = df['signal'].iloc[i] if not pd.isna(df['signal'].iloc[i]) else 0
            
            close_price = df['close'].iloc[i]
            
            # Entry logic
            if not in_position and current_signal == 1:
                # Enter long with slippage
                exec_price = close_price * (1 + self.config.slippage_rate)
                in_position = True
                current_position = 1
                current_entry = exec_price
                
            # Exit logic
            elif in_position:
                # Check if we've reached holding period
                bars_held = i - np.where(position[:i] == 1)[0][0] if np.any(position[:i] == 1) else 0
                
                should_exit = False
                
                # Exit on signal reversal
                if current_signal == 0:
                    should_exit = True
                
                # Exit on stop loss
                if self.config.stop_loss and current_position == 1:
                    unrealized_pnl = (close_price - current_entry) / current_entry
                    if unrealized_pnl <= -self.config.stop_loss:
                        should_exit = True
                
                # Exit on take profit
                if self.config.take_profit and current_position == 1:
                    unrealized_pnl = (close_price - current_entry) / current_entry
                    if unrealized_pnl >= self.config.take_profit:
                        should_exit = True
                
                # Exit after holding period
                if bars_held >= self.config.holding_period:
                    should_exit = True
                
                if should_exit:
                    # Exit with slippage
                    exec_price = close_price * (1 - self.config.slippage_rate)
                    
                    # Calculate PnL
                    gross_return = (exec_price - current_entry) / current_entry
                    
                    # Subtract fees (entry + exit)
                    net_return = gross_return - 2 * self.config.fee_rate
                    
                    pnl[i] = net_return
                    
                    # Record trade
                    trades.append({
                        'entry_idx': np.where(position[:i] == 1)[0][0] if np.any(position[:i] == 1) else i-1,
                        'exit_idx': i,
                        'entry_price': current_entry,
                        'exit_price': exec_price,
                        'return': net_return,
                        'bars_held': bars_held
                    })
                    
                    in_position = False
                    current_position = 0
                    current_entry = 0
            
            position[i] = current_position
            entry_price[i] = current_entry if in_position else 0
        
        # Calculate equity curve
        returns = pd.Series(pnl, index=df.index)
        equity = (1 + returns).cumprod() * initial_capital
        
        # Handle any remaining open position at end
        if in_position and len(trades) > 0:
            last_exit = df['close'].iloc[-1] * (1 - self.config.slippage_rate)
            final_return = (last_exit - current_entry) / current_entry - 2 * self.config.fee_rate
            pnl[-1] = final_return
            equity.iloc[-1] = equity.iloc[-2] * (1 + final_return)
        
        # Build trades DataFrame
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=['entry_idx', 'exit_idx', 'entry_price', 'exit_price', 'return', 'bars_held']
        )
        
        # Calculate metrics
        result = self._calculate_metrics(equity, returns, trades_df, horizon)
        result.equity_curve = equity
        result.trades = trades_df
        
        self.results = result
        return result
    
    def _calculate_metrics(self, equity: pd.Series, returns: pd.Series, 
                          trades: pd.DataFrame, horizon: int) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        
        # Basic returns
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        
        # Annualized return (assuming 15-min bars, ~35040 bars/year)
        bars_per_year = 35040  # 24*4*365
        n_bars = len(equity)
        years = n_bars / bars_per_year
        if years > 0:
            annualized_return = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1
        else:
            annualized_return = 0.0
        
        # Risk metrics
        daily_returns = returns.resample('D').sum() if hasattr(returns.index, 'resample') else returns
        risk_free_rate = 0.02  # 2% annual
        
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() * bars_per_year - risk_free_rate) / (returns.std() * np.sqrt(bars_per_year))
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() * bars_per_year - risk_free_rate) / (downside_returns.std() * np.sqrt(bars_per_year))
        else:
            sortino_ratio = sharpe_ratio
        
        # Maximum drawdown
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Win rate
        if len(trades) > 0:
            winning_trades = (trades['return'] > 0).sum()
            win_rate = winning_trades / len(trades)
            
            # Profit factor
            gross_profits = trades[trades['return'] > 0]['return'].sum()
            gross_losses = abs(trades[trades['return'] <= 0]['return'].sum())
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
            
            avg_trade_return = trades['return'].mean()
            
            # Expectancy
            expectancy = (win_rate * avg_trade_return) - ((1-win_rate) * abs(avg_trade_return))
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_return = 0.0
            expectancy = 0.0
        
        # Calmar ratio
        if max_drawdown > 0:
            calmar_ratio = annualized_return / max_drawdown
        else:
            calmar_ratio = 0.0
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_return=avg_trade_return,
            expectancy=expectancy,
            calmar_ratio=calmar_ratio,
            equity_curve=equity,
            trades=trades
        )
    
    def _empty_result(self) -> BacktestResult:
        """Return empty result for edge cases."""
        return BacktestResult(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            avg_trade_return=0.0,
            expectancy=0.0,
            calmar_ratio=0.0,
            equity_curve=pd.Series([100000]),
            trades=pd.DataFrame()
        )


def compare_strategies(df: pd.DataFrame, 
                       predictions: Dict[str, pd.Series],
                       horizon: int = 15) -> pd.DataFrame:
    """
    Compare multiple prediction strategies.
    
    Args:
        df: DataFrame with OHLCV data
        predictions: Dict mapping strategy name to probability series
        horizon: Prediction horizon
        
    Returns:
        DataFrame with comparison metrics
    """
    backtester = Backtester()
    results = []
    
    for strategy_name, proba_series in predictions.items():
        df_temp = df.copy()
        df_temp['proba'] = proba_series
        
        try:
            result = backtester.run(df_temp, 'proba', horizon)
            results.append({
                'strategy': strategy_name,
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'total_trades': result.total_trades,
                'calmar_ratio': result.calmar_ratio
            })
        except Exception as e:
            logger.warning(f"Strategy {strategy_name} failed: {e}")
            results.append({
                'strategy': strategy_name,
                'total_return': 0.0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
                'calmar_ratio': 0.0
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Quick test
    from src.features.engine import FeatureEngine, create_targets
    from src.models.trainer import LightGBMModel, prepare_data
    
    print("Loading data...")
    df = pd.read_parquet('data/raw/BTCUSDT_15m.parquet').head(10000)
    
    print("Creating features and targets...")
    engine = FeatureEngine()
    df = engine.compute_all_features(df)
    df = create_targets(df, horizons=[15])
    
    feature_cols = engine.get_feature_columns(df)
    df = engine.clean_features(df, feature_cols)
    
    # Train model
    X, y = prepare_data(df, feature_cols, 'target_binary_15')
    split = int(len(X) * 0.8)
    
    model = LightGBMModel()
    model.fit(X[:split], y[:split])
    
    # Get predictions
    df['proba'] = 0.5
    proba_values = model.predict_proba(X[split:])[:, 1]
    df.iloc[split:, df.columns.get_loc('proba')] = proba_values
    
    # Run backtest
    backtester = Backtester(TradeConfig(
        fee_rate=0.001,
        slippage_rate=0.0005,
        min_signal_strength=0.55,
        holding_period=15
    ))
    
    result = backtester.run(df, 'proba', horizon=15)
    
    print("\n=== BACKTEST RESULTS ===")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Annualized Return: {result.annualized_return:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Calmar Ratio: {result.calmar_ratio:.2f}")
