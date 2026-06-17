# NexusTrade

**Production-grade multiprocessing algorithmic trading framework for MetaTrader 5.**

Run multiple independent strategies in parallel — each isolated in its own process, all sharing a portfolio-level risk policy enforced in real time.

> **Windows only.** The MetaTrader 5 Python API ships as a Windows DLL. NexusTrade wraps it without compromise.

---

## Motivation

No existing MT5 Python project did all of these at once:

- **Run multiple strategies simultaneously** — every open-source example assumed a single strategy running in a single loop
- **Mix timeframes freely** — M15, H1, D1 strategies coexist with independent bar-aligned schedules, each on its own process
- **Enforce risk at the portfolio level** — per-strategy position limits are easy; enforcing a global drawdown ceiling or a total position cap _atomically across processes_ is not
- **Trade multiple live accounts in parallel** — install a second MT5 terminal to a separate directory, point a second `.env` at it with a different `MT5_PATH` and `MT5_LOGIN`, and run a second `nexus-trade` process; each instance is fully isolated with its own log directory, trade database, and risk profile

NexusTrade was built to fill that gap. The design goal: adding a new strategy to a live portfolio should require writing two files and one config block — nothing else.

---

## Architecture

```
┌──────────────────────── Main Process ─────────────────────────────┐
│                                                                   │
│                           Orchestrator                            │
│                                                                   │
│   MT5Connection ──► PositionRepository ──► position cache (60 s)  │
│   threads: heartbeat_monitor  ·  drawdown_refresh                 │
│                                                                   │
└────────────────────────────┬──────────────────────────────────────┘
                             │  multiprocessing.Process × N
             ┌───────────────┼───────────────┐
             │               │               │
  ┌──────────▼──────┐ ┌──────▼───────┐ ┌─────▼───────────┐
  │ StrategyRunner  │ │StrategyRunner│ │ StrategyRunner  │
  │  EURUSD · M15   │ │ GBPUSD · H1  │ │  XAUUSD · H4    │
  │                 │ │              │ │                 │
  │  MT5Connection  │ │ MT5Connection│ │  MT5Connection  │
  │  DataHandler    │ │ DataHandler  │ │  DataHandler    │
  │  RiskManager    │ │ RiskManager  │ │  RiskManager    │
  │  OrderExecutor  │ │ OrderExecutor│ │  OrderExecutor  │
  │  AsyncLogger    │ │ AsyncLogger  │ │  AsyncLogger    │
  │   trades_a.db   │ │  trades_b.db │ │   trades_c.db   │
  └────────┬────────┘ └──────┬───────┘ └────────┬────────┘
           │                 │                  │
           └─────────────────┼──────────────────┘
                             │  read / write
  ┌──────────────────────────▼──────────────────────────────────┐
  │                       Shared IPC                            │
  │                                                             │
  │  Manager.dict  shared_state     Lock  position_cache_lock   │
  │  Value  global_position_count   Value  global_trade_count   │
  │  SQLite  trade_id_sequence.db                               │
  └─────────────────────────────────────────────────────────────┘
```

Orchestrator runs in the main process: refreshes the shared position cache from MT5 every 60 s, tracks drawdown on a background thread, and monitors process liveness. Each `StrategyRunner` is a fully independent `multiprocessing.Process` with its own MT5 connection, data pipeline, and SQLite trade log. Cross-process coordination uses only three primitives — a `Manager.dict` for cache and state, two `Value` atomic counters, and one `Lock` — keeping IPC surface minimal and contention-free.

---

## Features

### Portfolio Risk Management
- **Global position cap** — atomic counter blocks new entries when the limit is reached across all strategies
- **Daily trade quota** — resets at midnight; rejects entries once exhausted
- **Daily drawdown ceiling** — compares live equity against intraday peak; halts all strategies on breach
- **Maximum drawdown ceiling** — historical peak-to-trough guard computed incrementally from MT5 deal history
- **Adaptive position sizing** — configurable drawdown thresholds scale risk fraction down as the portfolio bleeds

### Per-Strategy Risk
- Fractional position sizing: `volume = (balance × risk_pct × multiplier) / (sl_distance / tick_size × tick_value)`
- Per-strategy max positions and daily trade limits
- Spread and slippage point guards before any order is sent
- News filter: blocks entry within a configurable buffer around high-impact economic events (parses MT5 CSV calendar export)

### Order Types

| Type | Description |
|------|-------------|
| `market` | Instant execution at ask/bid |
| `limit` | `BUY_LIMIT` / `SELL_LIMIT` pending |
| `stop` | `BUY_STOP` / `SELL_STOP` pending |
| `bracket` | OCO pair: buy-stop + sell-stop, auto-cancels the losing leg on fill |

Bracket orders degrade gracefully — entry inside the broker's stops level downgrades to a limit; inside `min_market_threshold_points`, executes at market.

### Execution
- Exponential-backoff retry on retryable MT5 error codes (`10006`, `10007`, `10010`, `10018`, `10019`)
- Symbol spec cache (300 s TTL) — one `symbol_info` round-trip per symbol per session
- Deal-ID recovery fallback when `order_send` returns `retcode=DONE` with `deal=0`
- Partial close support with proportional commission and swap attribution

### Meta-Labeling _(optional)_
Plug in an XGBoost classifier per strategy. Predicted class probability scales position volume. Calibration layer (Platt/Beta/Isotonic) optional. Trades below `min_confidence` are rejected before the position slot is reserved.

### Trade Logging
- SQLite per strategy, WAL mode, thread-local connections
- Single row per trade, updated on exit; partial closes append rows with an incremented `partial_sequence`
- Columns: entry/exit datetime, size, actual and expected prices, spread, slippage cost, commission, swap, gross/net PnL, RRR, fill latency (ms), volume multiplier, exit trigger
- Async write queue — logging never blocks the event loop
- Startup reconciliation: on restart, open positions are matched to their last known `trade_id` from the DB

---

## Quick Start

**Prerequisites:** Python 3.13+, MetaTrader 5 terminal installed and logged in, Windows.

```bash
# 1. Clone
git clone https://github.com/marcell-k/nexus-trade.git
cd nexus-trade

# 2. Install (uv recommended)
uv sync

# 3. Copy and fill the risk profile template
cp config/profiles/example.toml config/profiles/live.toml
# edit live.toml

# 4. Create a .env file
MT5_LOGIN=12345678
MT5_PASSWORD=yourpassword
MT5_SERVER=YourBroker-Live
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_CALENDAR_PATH=path_to_calendar.txt
BROKER_TIMEZONE="Europe/Athens"
RISK_PROFILE=src\nexus_trade\config\profiles/live.toml

# 5. Run
uv run nexus-trade --env .env
```

### Running Multiple Accounts

Download and install a second MT5 terminal to a separate directory (e.g. `C:\MT5_Account2\terminal64.exe`). Create a second `.env` with a different `MT5_PATH`, `MT5_LOGIN`, and `RISK_PROFILE`, then launch a second `nexus-trade` process in a new terminal window:

```bash
uv run nexus-trade --env .env.account2
```

Each process derives its log directory from the env filename, maintains its own `trade_id_sequence.db`, and enforces its own risk profile. The two instances share no state.

---

## Adding a Strategy

A strategy is two files. The framework discovers and wires everything else.

```
src/nexus_trade/strategies/
└── my_strategy/
    ├── config.py      # StrategyConfig + typed params
    └── strategy.py    # signal generation logic
```

See `src/nexus_trade/strategies/sma_crossover/` for a complete working example — read both files alongside the `[strategies.sma_crossover]` block in `config/profiles/example.toml` before writing your own.

Once both files are in place, enable the strategy in your risk profile and restart:

```toml
# config/profiles/live.toml
[strategies.my_strategy]
enabled  = true
risk_pct = 1.0    # 1% of account balance per trade
```

NexusTrade spawns a process for `my_strategy` alongside every other enabled strategy. All portfolio limits apply immediately.

---

## Configuration Reference

Fully annotated examples for every option live in `config/profiles/example.toml` — read that file directly. The table below is a quick field reference for `StrategyConfig`.

### `StrategyConfig` fields

| Field | Type | Description |
|-------|------|-------------|
| `order_type` | `market` \| `limit` \| `stop` \| `bracket` | Order routing mode |
| `execution.magic_number` | `int` | MT5 magic — **unique per strategy** |
| `execution.deviation` | `int` | Max slippage points for market orders |
| `risk.max_positions` | `int` | Max simultaneous positions for this strategy |
| `risk.max_trades` | `int` | Max daily trades for this strategy |
| `risk.max_spread_points` | `int` | Reject entry if spread exceeds this |
| `filters.news.enabled` | `bool` | Enable economic calendar filter |
| `filters.news.currencies` | `list[str]` | e.g. `["USD", "EUR"]` |
| `filters.news.buffer_minutes` | `int` | Block trading N minutes either side of an event |
| `trading_hours.enabled` | `bool` | Restrict entries to session windows |
| `trading_hours.sessions` | `list` | e.g. `[{start = "08:00", end = "17:00"}]` |
| `params.backcandles` | `int` | Bars fed to signal functions |
| `params.timeframe` | `str` | `M1` `M5` `M15` `M30` `H1` `H4` `D1` |

---

## Project Structure

```
nexus-trade/
├── src/nexus_trade/
│   ├── config/          # Pydantic models: account, strategy, risk profile, timings
│   ├── core/            # MT5 connection, position cache, DataHandler, type definitions
│   ├── execution/       # OrderExecutor, request dataclasses, TradeIDSequenceManager
│   ├── filters/         # NewsFilter, MarketCostCalculator, meta-labeling loader
│   ├── logging/         # TradeLogger (SQLite, WAL) + AsyncTradeLogger (queue wrapper)
│   ├── risk/            # RiskManager — layered validation pipeline
│   ├── utils/           # Formatting, system sleep inhibitor, DB utilities
│   ├── orchestrator.py  # Multi-process coordinator, shared state, heartbeat + cache loop
│   └── runner.py        # Per-strategy event loop, position lifecycle, exit monitor
├── typings/MetaTrader5/ # Hand-written stub — complete type coverage for the MT5 C API
├── tests/unit/
└── config/profiles/
    └── example.toml     # Annotated risk profile — start here
```

---

## License

MIT

## Disclaimer

NexusTrade is provided for educational and research purposes.
Algorithmic trading involves substantial risk of loss.
The authors accept no liability for financial losses incurred through use of this software.
Always test on a demo account before deploying real capital.
