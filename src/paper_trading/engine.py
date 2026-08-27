"""
Paper Trading Engine

Simulates real-time trading with:
- Virtual capital
- Realistic execution (fees, slippage)
- Position tracking
- PnL calculation
- Trade logging
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        if self.side == 'LONG':
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized PnL percentage."""
        if self.side == 'LONG':
            return (current_price / self.entry_price - 1) * 100
        else:
            return (1 - current_price / self.entry_price) * 100


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    fees: float
    exit_reason: str  # 'SIGNAL', 'STOP_LOSS', 'TAKE_PROFIT', 'TIME'


@dataclass
class PortfolioState:
    """Current portfolio state."""
    cash: float
    positions: Dict[str, Position]
    total_value: float
    unrealized_pnl: float
    realized_pnl: float


class PaperTradingEngine:
    """
    Simulates trading without real money.
    
    Tracks positions, calculates PnL, and logs all trades.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 fee_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 default_position_size: float = 0.1):
        """
        Args:
            initial_capital: Starting virtual capital
            fee_rate: Trading fee per transaction (0.1% = 0.001)
            slippage_rate: Expected slippage (0.05% = 0.0005)
            default_position_size: Fraction of capital per trade (10% = 0.1)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.position_size = default_position_size
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[datetime] = [datetime.now()]
        
        # Performance tracking
        self.total_realized_pnl = 0.0
        self.total_fees_paid = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def get_portfolio_state(self, current_prices: Dict[str, float]) -> PortfolioState:
        """Get current portfolio state."""
        unrealized_pnl = sum(
            pos.unrealized_pnl(current_prices.get(pos.symbol, pos.entry_price))
            for pos in self.positions.values()
        )
        
        position_value = sum(
            pos.quantity * current_prices.get(pos.symbol, pos.entry_price)
            for pos in self.positions.values()
        )
        
        total_value = self.cash + position_value
        
        return PortfolioState(
            cash=self.cash,
            positions=self.positions.copy(),
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.total_realized_pnl
        )
    
    def execute_signal(self, 
                       symbol: str,
                       signal: int,  # 1 = LONG, -1 = SHORT, 0 = CLOSE
                       price: float,
                       timestamp: datetime,
                       quantity: Optional[float] = None,
                       stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> Optional[Trade]:
        """
        Execute a trading signal.
        
        Args:
            symbol: Trading pair
            signal: 1=enter long, -1=enter short, 0=close position
            price: Current market price
            timestamp: Trade timestamp
            quantity: Override default position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Trade object if trade executed, None otherwise
        """
        exec_price = self._get_execution_price(price, signal)
        
        # Close existing position if signal is opposite or zero
        if symbol in self.positions:
            existing_pos = self.positions[symbol]
            
            # Close on opposite signal or explicit close signal
            if (signal == 0 or 
                (signal == 1 and existing_pos.side == 'SHORT') or
                (signal == -1 and existing_pos.side == 'LONG')):
                
                trade = self._close_position(symbol, exec_price, timestamp, 'SIGNAL')
                self.trades.append(trade)
                
                if signal == 0:
                    return trade
        
        # Open new position if signal indicates
        if signal != 0 and symbol not in self.positions:
            qty = quantity or self._calculate_quantity(price)
            
            cost = qty * exec_price
            fee = cost * self.fee_rate
            
            if cost + fee > self.cash:
                logger.warning(f"Insufficient cash for {symbol}")
                return None
            
            self.cash -= (cost + fee)
            self.total_fees_paid += fee
            
            side = 'LONG' if signal == 1 else 'SHORT'
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                entry_price=exec_price,
                quantity=qty,
                entry_time=timestamp,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            logger.info(f"Opened {side} {symbol} @ {exec_price:.2f}")
        
        return None
    
    def check_stop_loss_take_profit(self, symbol: str, current_price: float, 
                                    timestamp: datetime) -> Optional[Trade]:
        """Check and execute stop loss or take profit."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        exit_reason = None
        exit_price = self._get_execution_price(current_price, -1 if pos.side == 'LONG' else 1)
        
        # Check stop loss
        if pos.stop_loss:
            if pos.side == 'LONG' and exit_price <= pos.stop_loss:
                exit_reason = 'STOP_LOSS'
            elif pos.side == 'SHORT' and exit_price >= pos.stop_loss:
                exit_reason = 'STOP_LOSS'
        
        # Check take profit
        if pos.take_profit and not exit_reason:
            if pos.side == 'LONG' and exit_price >= pos.take_profit:
                exit_reason = 'TAKE_PROFIT'
            elif pos.side == 'SHORT' and exit_price <= pos.take_profit:
                exit_reason = 'TAKE_PROFIT'
        
        if exit_reason:
            trade = self._close_position(symbol, exit_price, timestamp, exit_reason)
            self.trades.append(trade)
            return trade
        
        return None
    
    def _close_position(self, symbol: str, exit_price: float, 
                        timestamp: datetime, reason: str) -> Trade:
        """Close an existing position."""
        pos = self.positions.pop(symbol)
        
        # Calculate PnL
        pnl = pos.unrealized_pnl(exit_price)
        pnl_pct = pos.unrealized_pnl_pct(exit_price)
        
        # Calculate fees
        exit_value = pos.quantity * exit_price
        exit_fee = exit_value * self.fee_rate
        self.total_fees_paid += exit_fee
        
        # Update cash
        self.cash += exit_value - exit_fee
        
        # Update performance tracking
        self.total_realized_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        trade = Trade(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fees=exit_fee,
            exit_reason=reason
        )
        
        logger.info(f"Closed {pos.side} {symbol}: PnL={pnl:.2f} ({pnl_pct:.2f}%), Reason={reason}")
        
        return trade
    
    def _get_execution_price(self, price: float, signal: int) -> float:
        """Calculate execution price with slippage."""
        if signal > 0:  # Buying
            return price * (1 + self.slippage_rate)
        elif signal < 0:  # Selling
            return price * (1 - self.slippage_rate)
        else:
            return price
    
    def _calculate_quantity(self, price: float) -> float:
        """Calculate position quantity based on position sizing."""
        position_value = self.cash * self.position_size
        return position_value / price
    
    def update_equity_curve(self, timestamp: datetime, current_prices: Dict[str, float]):
        """Update equity curve tracking."""
        state = self.get_portfolio_state(current_prices)
        self.equity_curve.append(state.total_value)
        self.timestamps.append(timestamp)
    
    def get_performance_summary(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {'message': 'No trades executed yet'}
        
        total_trades = len(self.trades)
        win_rate = self.winning_trades / total_trades if total_trades > 0 else 0
        
        winning_pnls = [t.pnl for t in self.trades if t.pnl > 0]
        losing_pnls = [t.pnl for t in self.trades if t.pnl <= 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        profit_factor = abs(sum(winning_pnls) / sum(losing_pnls)) if losing_pnls and sum(losing_pnls) != 0 else float('inf')
        
        max_drawdown = self._calculate_max_drawdown()
        
        # Return calculations
        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_realized_pnl': self.total_realized_pnl,
            'total_fees_paid': self.total_fees_paid,
            'total_trades': total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown / self.initial_capital * 100 if self.initial_capital > 0 else 0
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0
        
        peak = self.equity_curve[0]
        max_dd = 0.0
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get trade log as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for t in self.trades:
            trades_data.append({
                'symbol': t.symbol,
                'side': t.side,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'quantity': t.quantity,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'fees': t.fees,
                'exit_reason': t.exit_reason
            })
        
        return pd.DataFrame(trades_data)


if __name__ == "__main__":
    # Test paper trading engine
    print("=== PAPER TRADING TEST ===\n")
    
    engine = PaperTradingEngine(
        initial_capital=100000,
        fee_rate=0.001,
        slippage_rate=0.0005,
        default_position_size=0.1
    )
    
    # Simulate some trades
    base_price = 70000
    timestamps = [datetime.now()]
    
    # Trade 1: Long BTC
    print("Executing: LONG BTCUSDT")
    engine.execute_signal('BTCUSDT', signal=1, price=base_price, timestamp=datetime.now())
    
    # Price moves up
    base_price *= 1.02
    print(f"\nPrice moved to {base_price:.2f}")
    engine.check_stop_loss_take_profit('BTCUSDT', base_price, datetime.now())
    
    # Close position
    print("\nExecuting: CLOSE BTCUSDT")
    engine.execute_signal('BTCUSDT', signal=0, price=base_price, timestamp=datetime.now())
    
    # Trade 2: Short ETH
    eth_price = 3500
    print(f"\nExecuting: SHORT ETHUSDT @ {eth_price}")
    engine.execute_signal('ETHUSDT', signal=-1, price=eth_price, timestamp=datetime.now(),
                         stop_loss=eth_price * 1.03, take_profit=eth_price * 0.97)
    
    # Price moves against us (hits stop loss)
    eth_price *= 1.03
    print(f"\nPrice moved to {eth_price:.2f} (should hit stop loss)")
    engine.check_stop_loss_take_profit('ETHUSDT', eth_price, datetime.now())
    
    # Get performance summary
    print("\n=== PERFORMANCE SUMMARY ===")
    summary = engine.get_performance_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Get trade log
    print("\n=== TRADE LOG ===")
    trade_log = engine.get_trade_log()
    print(trade_log.to_string())
