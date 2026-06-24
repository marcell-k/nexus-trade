# NexusTrade

**Multiprocessing algorithmic trading framework for MetaTrader 5.**

Run multiple independent strategies in parallel — each isolated in its own process, all sharing a portfolio-level risk policy enforced in real time.

---

## Motivation

- Run multiple strategies simultaneously
- Mix timeframes
- Enforce risk at the portfolio level
- Trade multiple live accounts in parallel

---

## Architecture

```
nexus-trade/
├── src/nexus_trade/
│   ├── config/                  # Pydantic models: account, strategy, risk profile, timings
│   │   └── profiles/
│   │       └── example.toml     # Annotated risk profile — start here
│   ├── core/                    # MT5 connection, position cache, DataHandler, type definitions
│   ├── execution/               # OrderExecutor, request dataclasses, TradeIDSequenceManager
│   ├── filters/                 # NewsFilter, MarketCostCalculator, meta-labeling loader
│   ├── logging/                 # TradeLogger (SQLite, WAL) + AsyncTradeLogger (queue wrapper)
│   ├── risk/                    # RiskManager — layered validation pipeline
│   ├── utils/                   # Formatting, system sleep inhibitor, DB utilities
│   ├── orchestrator.py          # Multi-process coordinator, shared state, heartbeat + cache loop
│   └── runner.py                # Per-strategy event loop, position lifecycle, exit monitor
├── typings/MetaTrader5/         # Hand-written stub — complete type coverage for the MT5 C API
└── tests/unit/

```
---

## Features

### Portfolio Risk Management
- Global position cap
- Daily trade limits
- Daily drawdown ceiling
- Maximum drawdown 
- Adaptive position sizing

### Per-Strategy Risk
- **Fractional sizing**: `volume = (balance × risk_pct × multiplier) / (sl_distance / tick_size × tick_value)`
- **Fixed sizing**: `volume = risk_dollar / (sl_distance / tick_size × tick_value)` — risk a fixed dollar amount regardless of balance
- Per-strategy max positions and daily trade limits
- Spread and slippage point guards
- News filter: blocks entry within a configurable buffer around high-impact economic events (parses MT5 CSV calendar export)

### Order Types
market, limt, stop, bracket (OCO)

### Trade Logging
- SQLite per strategy, WAL mode, thread-local connections
- Startup reconciliation: on restart, open positions are matched to their last known `trade_id` from the DB

---

## Quick Start

**Prerequisites:** Python 3.13+, MetaTrader 5 terminal installed and logged in, Windows.

```bash
# 1. Clone
git clone https://github.com/marcell-k/nexus-trade.git
cd nexus-trade

# 2. Install
uv sync

# 3. Copy and update the risk profile template
cp src/nexus_trade/config/profiles/example.toml src/nexus_trade/config/profiles/live.toml

# 4. Create a .env file
LOGIN=12345678
PASSWORD=yourpassword
SERVER=YourBroker-Live
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
CALENDAR_PATH=path_to_calendar.txt
BROKER_TIMEZONE="Europe/Athens"
RISK_PROFILE=src\nexus_trade\config\profiles\live.toml

# 5. Run
uv run nexus-trade --env .env
```

### Running Multiple Accounts

Download and a second MT5 terminal to a separate directory.

```bash
uv run nexus-trade --env .env.account2
```

---

## Adding a Strategy

A strategy is two files. The framework discovers and wires everything else.

```
src/nexus_trade/strategies/
└── my_strategy/
    ├── config.py      # StrategyConfig + typed params
    └── strategy.py    # signal generation logic
```

```toml
# src/nexus_trade/config/profiles/live.toml

[strategies.my_strategy]
enabled                = true
position_sizing_method = "fractional"   # "fractional" | "fixed"
risk_value             = 1.0            # fractional: 1 % of balance per trade
                                        # fixed:      dollar amount per trade (e.g. 500)

[strategies.my_strategy.meta_labeling]
enabled         = false
use_calibration = false
min_confidence  = 0.0
```

---

## Disclaimer

NexusTrade is provided for educational and research purposes.
Algorithmic trading involves substantial risk of loss.
The authors accept no liability for financial losses incurred through use of this software.
Always test on a demo account before deploying real capital.
