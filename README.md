# Autonomous Cryptocurrency Prediction & Research System

## Overview

A complete AI-powered cryptocurrency market prediction system that produces calibrated probability estimates for price direction across multiple time horizons.

**⚠️ Important Disclaimer**: This is a research and paper-trading system only. It does not guarantee profits and should not be used for real-money trading without extensive additional testing and validation. Past performance does not indicate future results.

## Features

- **Multi-horizon predictions**: 15min, 30min, 60min forecasts
- **Probability calibration**: Ensures predicted probabilities match observed frequencies  
- **Risk assessment**: Multi-factor risk scoring (LOW/MEDIUM/HIGH/EXTREME)
- **Realistic backtesting**: Accounts for fees, slippage, and signal delay
- **Walk-forward validation**: Proper time-series cross-validation to prevent lookahead bias
- **Multiple model comparison**: From naive baselines through LightGBM/XGBoost

## Project Structure

```
/workspace
├── src/
│   ├── data/           # Data collection from Binance API
│   ├── features/       # Technical indicator engineering (66 features)
│   ├── models/         # Model training (Logistic, RF, XGBoost, LightGBM)
│   ├── calibration/    # Probability calibration (Isotonic, Platt)
│   ├── risk/           # Risk assessment engine
│   ├── backtest/       # Realistic backtesting with fees/slippage
│   ├── api/            # REST API (FastAPI)
│   └── dashboard/      # Web dashboard
├── data/raw/           # Historical OHLCV data
├── experiments/        # Experiment tracking
├── models/             # Trained model artifacts
└── docs/               # Documentation
```

## Installation

```bash
# Dependencies already installed: pandas, numpy, scikit-learn, lightgbm, xgboost, torch, fastapi
```

## Quick Start

### 1. Collect Data
```bash
python src/data/collector.py
```

### 2. Test Feature Engineering
```bash
python src/features/engine.py
```

### 3. Test Models
```bash
PYTHONPATH=/workspace python -c "
from src.models.trainer import create_models, evaluate_model
print('Models available:', list(create_models().keys()))
"
```

### 4. Run Backtest
```bash
PYTHONPATH=/workspace python src/backtest/engine.py
```

## Output Format

The system produces predictions in this format:

```
BTCUSDT - 15 MINUTES
* Probability of UP: 68%
* Expected return: +0.35% to +0.75%
* Risk: MEDIUM
* Confidence: Calibrated
```

## Model Hierarchy

| Tier | Models | Purpose |
|------|--------|---------|
| Baseline | Naive, Persistence | Minimum viable performance |
| Classical | Logistic Regression, Random Forest | Interpretable benchmarks |
| Gradient Boosting | XGBoost, LightGBM | Primary candidates |
| Deep Learning | LSTM, GRU, Transformer | Only if justified by results |

## Key Metrics

### Classification
- Accuracy, Precision, Recall, F1
- ROC-AUC, Brier Score (calibration quality)

### Trading Performance  
- Total/Annualized Return
- Sharpe Ratio, Sortino Ratio
- Maximum Drawdown
- Win Rate, Profit Factor

## Architecture Decisions

### Why LightGBM over Deep Learning?
For tabular time-series features, gradient boosting typically outperforms deep learning while being faster to train and less prone to overfitting.

### Why Isotonic Calibration?
Non-parametric method that makes no assumptions about probability distribution, providing better calibration than raw model outputs.

### Why Walk-Forward Validation?
Standard k-fold CV leaks future information in time-series. Walk-forward ensures training data always precedes test data.

## Known Limitations

1. **Market Efficiency**: Crypto markets are highly efficient; consistent alpha is difficult
2. **Regime Dependency**: Performance varies across bull/bear/sideways markets
3. **Transaction Costs**: High-frequency strategies may be unprofitable after fees
4. **Overfitting Risk**: Extensive tuning increases overfitting; walk-forward essential

## Experimental Status

Preliminary testing on BTCUSDT 15-minute data shows:
- LightGBM achieves ~57% accuracy vs 52.8% naive baseline
- ROC-AUC ~0.63 indicates modest predictive signal
- Calibration improves Brier score by ~15%

Full walk-forward validation across multiple market regimes is in progress.

## Next Steps

1. Complete comprehensive walk-forward validation
2. Implement real-time inference pipeline  
3. Build monitoring dashboard
4. Add paper trading engine
5. Test across multiple coins and timeframes

## License

MIT License - Research purposes only. Not for production trading.
