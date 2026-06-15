# Contributing

## Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv)

## Setup

```bash
git clone https://github.com/marcell-k/nexus-trade.git
cd nexus-trade
uv sync --dev
```

## Before opening a PR

```bash
uv run ruff check src tests
uv run ruff format src tests
uv run basedpyright src
uv run pytest tests/unit -q
```

All four must pass. The CI gate runs the same commands.

## Writing a strategy

See `src/nexus_trade/strategies/sma_crossover/` for the canonical example.
Two files required — `config.py` and `strategy.py`. Read `README.md §Adding a Strategy` first.

## Commit style

`<type>(<scope>): <subject>` — e.g. `fix(executor): correct orders_get typo`.
Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.

## Reporting bugs

Open a GitHub issue with:
- Python version, broker name
- Minimal reproduction (anonymise credentials)
- Relevant log lines (set `logging.DEBUG` temporarily)
