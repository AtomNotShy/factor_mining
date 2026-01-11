# API Routers

**Generated:** 2025-01-10

---

## OVERVIEW

FastAPI boundary layer with 7 routers providing REST endpoints for data, factors, evaluation, strategies, backtesting, monitoring, and visualization.

---

## STRUCTURE

```
src/api/routers/
├── data.py              # Data collection, exchange health, symbols, OHLCV
├── factors.py           # Factor registry, calculation, categories
├── evaluation.py        # Factor evaluation, IC analysis, backtest metrics
├── strategy.py          # Strategy generation, listing
├── strategy_backtest.py # Strategy backtesting, results (1680 lines)
├── monitoring.py        # Alerts, system health
└── visualization.py     # Chart data, plot rendering
```

---

## WHERE TO LOOK

| Task | Router | Pattern |
|------|--------|---------|
| **Add data endpoint** | `data.py` | `@router.post("/path", response_model=Response)` |
| **Factor calculation** | `factors.py` | `/api/v1/factors/calculate/{factor_name}` |
| **Strategy backtest** | `strategy_backtest.py` | POST `/api/v1/strategy-backtest/run` |
| **Pydantic schemas** | `../schemas/` | Request/response models |

---

## CONVENTIONS

- **Router prefix**: All routers use `/api/v1` prefix in `main.py`
- **Dependency injection**: `Depends(get_settings)` for config access
- **Response models**: `response_model=SchemaClass` for auto-validation
- **Error handling**: `raise HTTPException(status_code, detail="message")`
- **Global exceptions**: Handled in `main.py` with `JSONResponse`

---

## ANTI-PATTERNS

### CRITICAL
- **Embedded HTML/JS viewer**: `strategy_backtest.py` lines 908-1680 contains inline Plotly.js HTML. Move to `frontend/`.
- **Monolithic backtest router**: 1680 lines in `strategy_backtest.py`. Split into separate modules.
- **Global instances**: Module-level `collector`, `polygon_collector`, `backtest_store`. Use `Depends()`.
- **Bare exceptions**: Multiple `except Exception` blocks. Catch specific types.
