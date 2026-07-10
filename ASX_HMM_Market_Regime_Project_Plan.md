# ASX Market Regime Detection Project Plan (Hidden Markov Models)

## Objective
Develop a production-style market regime detection framework for the Australian equity market using Hidden Markov Models (HMMs). The system should identify hidden market regimes, estimate transition probabilities, and provide regime-aware insights for portfolio risk management and research.

---

# Project Scope

## Primary Market
- Australian equity market
- Primary benchmark: IOZ.AX or VAS.AX
- Daily historical data

## Data Sources

### Primary
- Yahoo Finance (via yfinance)
- Free, reliable, broad ASX coverage

### Secondary
- EODHD
- Used for validation and future expansion (fundamentals)

---

# Assumptions

- Daily frequency is sufficient.
- Regimes are latent and inferred statistically.
- Multiple market characteristics produce more robust regimes than returns alone.
- Historical behaviour provides useful information about future market states.

---

# High-Level Workflow

1. Define market universe and benchmark.
2. Acquire historical data.
3. Clean and align data.
4. Engineer market features.
5. Standardise features.
6. Train multiple HMMs with different numbers of hidden states.
7. Select the optimal model using statistical criteria and interpretability.
8. Label and interpret market regimes.
9. Analyse transition probabilities and regime persistence.
10. Visualise historical regimes.
11. Validate regime quality.
12. Backtest a regime-aware allocation strategy.
13. Automate data updates and reporting.
14. Extend with macroeconomic and cross-asset variables.

---

# Feature Categories

- Market returns
- Volatility
- Trend
- Drawdown
- Volume
- Momentum
- Range-based volatility
- Optional macro variables
- Optional cross-asset indicators

---

# Model Outputs

- Current market regime
- Regime probabilities
- Transition matrix
- Historical regime timeline
- Regime statistics
- Risk recommendations

---

# Validation

Evaluate:
- Statistical fit
- Regime stability
- Economic interpretability
- Out-of-sample performance
- Portfolio performance improvements

---

# Deliverables

- Data pipeline
- Feature engineering pipeline
- HMM model
- Regime visualisations
- Backtesting framework
- Dashboard
- Documentation

---

# Future Enhancements

- Macroeconomic indicators
- Cross-asset features
- Sector-level regimes
- Alternative HMM variants
- Walk-forward retraining
- Portfolio optimisation

---

# Proposed Timeline

## Phase 1 (Week 1)
- Select benchmark
- Build data pipeline
- Data validation

## Phase 2 (Week 2)
- Feature engineering
- Data preprocessing

## Phase 3 (Week 3)
- Train HMMs
- Compare model configurations

## Phase 4 (Week 4)
- Regime interpretation
- Visualisations
- Transition analysis

## Phase 5 (Week 5)
- Backtesting
- Performance evaluation

## Phase 6 (Week 6)
- Dashboard
- Documentation
- Final review

---

# Extensions:
## Layer 1 – Market (ETF)

These describe the behaviour of the ASX200 itself.

- Returns
- Volatility
- Drawdown
- ATR
- RSI
- MACD
- Volume

## Layer 2 – Market Breadth

These describe how healthy the market is underneath the index.

Examples:

- Percentage of ASX200 stocks above their 50-day moving average
- Percentage making 52-week highs
- Advance/Decline ratio
- Equal-weight vs market-cap-weight performance
- New highs minus new lows

These are much harder to obtain from free APIs but add a lot of value.

## Layer 3 – Australian Macro

Variables that influence the Australian market.

Examples include:

- AUD/USD exchange rate
- Iron ore price
- Gold price
- Brent crude oil
- RBA cash rate
- Australian 10-year government bond yield

## Layer 4 – Global Risk

Because the ASX is heavily influenced by overseas markets.

Examples:

- S&P 500 returns (SPY)
- Nasdaq returns (QQQ)
- VIX
- MSCI World ETF
- USD Index (DXY)

# Success Criteria

- Stable and interpretable market regimes
- Meaningful transition probabilities
- Improved understanding of market behaviour
- Practical use for risk management and investment research
- Modular architecture suitable for future expansion


# Dump:


## Time structure

- Compute features from daily data.

- Train the HMM on daily observations.

- Aggregate the regime signal to monthly frequency (e.g., the dominant regime in each month or the regime probability at month-end).


This gives you the best of both worlds:

### Daily data advantages

- More observations for training.

- Better estimation of volatility and drawdowns.

- Captures transitions more accurately.

### Monthly regime advantages

- Less noisy.

- More economically meaningful.

- More useful for asset allocation.

- Easier to interpret in reports.


## Validation FrameWork
Train 2010-2021
Test 2022-2026

### How to Evaluate:
1. Economic Interpretability
Ask:

Does the model identify COVID as a crisis regime?

Does it identify the 2022 selloff as a different regime?

Does it identify long bull markets?

This is often the most important validation.

2. Regime Statistics

For each state:

Metric       Question
Mean return  Is it positive or negative?
Volatility   Is it high or low?
Drawdown     Is it severe?
Duration     Does it persist?

A good HMM produces states that are clearly different.

3. Out-of-Sample Likelihood

Compute the log-likelihood of the test data under the trained model.

Compare:

2-state HMM

3-state HMM

4-state HMM

5-state HMM

Choose the model with the best trade-off between fit and simplicity.

4. Strategy Validation
Use the regimes in a simple allocation rule.

Regime   Allocation
Bull     100% equities
Recovery 70% equities
High Vol 40% equities 
Bear     0-20% equities

Then compare against buy-and-hold.

If the regime model reduces drawdowns while maintaining reasonable returns, that's strong evidence that the regimes are meaningful.
