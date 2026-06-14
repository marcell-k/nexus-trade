# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
NexusTrade uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

### Added
- Multi-process strategy orchestration via `multiprocessing.Process`
- Portfolio-level risk enforcement with atomic shared counters
- Per-strategy SQLite trade logging (WAL mode, async write queue)
- Bracket (OCO) order support with dual-fill protection
- Adaptive position sizing with drawdown thresholds
- Meta-labeling support (XGBoost + optional probability calibration)
- News filter via MT5 economic calendar CSV export
- Trading hours session filter with midnight-spanning support
- Exponential-backoff retry on retryable MT5 error codes
- Symbol spec cache (300 s TTL)
- Deal-ID recovery fallback for `retcode=DONE` with `deal=0`
- Partial close logging with proportional commission attribution
- Startup position reconciliation against SQLite trade history
- Multi-account support via separate `.env` files and MT5 installs
- Hand-written `MetaTrader5` type stubs for full static analysis coverage
- `WindowsInhibitor` to prevent system sleep during live sessions
