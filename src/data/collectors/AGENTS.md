# AGENTS.md - Data Collectors

**Generated:** 2025-01-10

---

## OVERVIEW

External API data collection layer for crypto exchanges, US stocks/ETFs via Polygon.io, and Interactive Brokers historical data.

---

## STRUCTURE

```
src/data/collectors/
├── base.py              # BaseDataCollector abstract class
├── exchange.py          # CCXT exchanges (Binance, OKX)
├── polygon.py           # Polygon.io API for US stocks/ETFs
└── ib_history.py        # IB TWS historical data
```

---

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add exchange | `exchange.py` → `ExchangeCollector` subclass |
| Polygon caching | `polygon.py` → `ParquetDataFrameStore` |
| IB historical | `ib_history.py` → `IBHistoryCollector` |
| Base interface | `base.py` → `BaseDataCollector` |

---

## CONVENTIONS & ANTI-PATTERNS

**Conventions**: All extend `BaseDataCollector`; use `async/await` for API calls; return DataFrame with `datetime` index (`ts_utc`); Polygon/IB use `ParquetDataFrameStore`; Polygon uses `certifi` for macOS cert issues; cache uses UTC-naive timestamps.
**Anti-patterns**: 11 broad `except Exception` handlers in `exchange.py` (silent failures); custom `Logger` class instead of Loguru from `src/utils/logger.py`; no rate limiting enforcement (relies on CCXT/Polygon); mixed sync/async (`IBHistoryCollector` has both `get_ohlcv()` and `get_ohlcv_async()`).

---

## COMMANDS

```bash
# Test Polygon (requires POLYGON_API_KEY): python3 -c "import asyncio; from src.data.collectors.polygon import PolygonCollector; c = PolygonCollector(); asyncio.run(c.get_ohlcv('SPY', '1d', limit=10))"
# Test IB (requires IB Gateway on 127.0.0.1:7497): python3 src/data/collectors/ib_history.py
```
