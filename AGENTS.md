# AGENTS.md - Factor Mining System Development Guide

**Generated:** 2025-01-10
**Commit:** Current working tree
**Branch:** N/A

---

## OVERVIEW

Factor mining and backtesting system with Python FastAPI backend + React/TypeScript frontend. ETF/stock strategy development with Interactive Brokers and Polygon.io data sources.

---

## STRUCTURE

```
factor_mining/
├── src/                    # Python backend (FastAPI)
│   ├── api/               # Routers (7 modules)
│   ├── config/            # Pydantic settings (nested with env_prefix)
│   ├── core/              # Domain types (Signal, Order, PortfolioState)
│   ├── data/              # Collectors (IB, Polygon, CCXT) + storage
│   ├── evaluation/        # Dual backtest engines + metrics
│   ├── execution/         # Broker implementations
│   ├── factors/           # 40+ technical factors
│   ├── strategies/        # Strategy implementations (v2)
│   └── utils/             # Loguru logger
├── frontend/               # React/TypeScript + Vite
│   └── src/
│       ├── components/     # Charts (Recharts + TradingView), pages
│       ├── pages/         # Dashboard, Backtest, History, Monitoring, Settings
│       ├── services/      # Axios API service
│       └── stores/       # Zustand state management
├── examples/              # 13 demo scripts
├── tests/                 # Empty (tests at root - see ANTI-PATTERNS)
└── data/                  # Local parquet cache, IB OHLCV
```

---

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| **Start backend** | `run.py` → `src/api/main.py` | Uvicorn on port 8000 |
| **Start frontend** | `frontend/src/main.tsx` | Vite on port 3000 |
| **Add strategy** | `src/strategies/` + register in `__init__.py` | Auto-registers on import |
| **Add factor** | `src/factors/technical/` | FactorRegistry |
| **API endpoints** | `src/api/routers/` | 7 routers with `/api/v1` prefix |
| **Settings** | `src/config/settings.py` | Pydantic with `env_prefix` per section |
| **Logger** | `src/utils/logger.py` | Loguru, colorize console |
| **Core types** | `src/core/types.py` | Dataclass enums (Signal, Order, OrderStatus) |
| **Test scripts** | `test_*.py` (ROOT) - See ANTI-PATTERNS | NOT in tests/ |

---

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `Signal` | dataclass | `src/core/types.py:50` | Trading signal (ts_utc, symbol, action, strength) |
| `OrderIntent` | dataclass | `src/core/types.py:64` | Order intent (side, qty, order_type) |
| `PortfolioState` | dataclass | `src/core/types.py:108` | Cash, positions, equity |
| `create_app()` | function | `src/api/main.py:68` | FastAPI factory with lifespan |
| `BacktestEngine` | class | `src/evaluation/backtesting/engine.py` | Primary backtest runner (v2) |
| `Strategy` | class | `src/strategies/base/strategy.py` | Strategy base class (v2) |
| `IBBroker` | class | `src/execution/ib_broker.py` | IB TWS integration |
| `FactorRegistry` | class | `src/factors/base/factor.py` | Factor discovery system |
| `StrategyRegistry` | class | `src/strategies/base/strategy.py` | Strategy discovery system |

---

## CONVENTIONS (Deviations from Standard)

### Python
- **Line length**: 100 chars (Black config in pyproject.toml)
- **Imports**: Standard → third-party → local, alphabetical within groups
- **Type annotations**: Required on all functions
- **Dataclasses**: Preferred for models (see `src/core/types.py`)
- **Enums**: Extend `str, Enum` for string-based enums
- **Config**: Pydantic `BaseSettings` with nested settings and `env_prefix`
- **Logging**: Loguru via `get_logger(__name__)`
- **Async**: `asynccontextmanager` for FastAPI lifespan

### Frontend
- **Imports**: React → third-party → local, use `@/*` alias
- **Components**: Functional with hooks, interface for props
- **State**: Zustand for global, local for component-level
- **Styling**: Tailwind CSS, dark mode via `'class'` strategy
- **Design**: Stripe Docs / Vercel / Linear / GOV.UK style (typography over icons)

---

## ANTI-PATTERNS (This Project)

### CRITICAL
- **`.env` file in repo**: Contains REAL API keys and credentials. Rotate immediately.
- **Tests at root**: `test_*.py` files in project root instead of `tests/`. Breaks pytest discovery.
- **Global variables**: `_ib_broker`, `_db_store`, `_global_registry` in multiple files. Use dependency injection.

### Code Quality
- **216 bare exception handlers**: `except:` and `except Exception` without specific types. Silent failures.
- **773 print() statements**: In tests and examples. Use `logger` instead.
- **Long files**: `strategy_backtest.py` (1681 lines). Break into modules.
- **API routes**: Some routes still use v1 patterns, need migration to v2.

### Organization
- **No CI/CD**: No GitHub Actions, pre-commit hooks, or Makefile.
- **Empty `tests/` directory**: Configured in pyproject.toml but contains only `__init__.py`.
- **Mixed docs**: Chinese markdown files at root. Move to `docs/` directory.

---

## COMMANDS

### Backend
```bash
# Install
pip install -r requirements.txt

# Run API server
python3 -m src.api.main
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Lint (line-length: 100)
black src/ && flake8 src/ && mypy src/

# Test (NOTE: tests at root, not in tests/)
pytest tests/ --cov=src  # Empty!
python test_momentum_20.py      # Manual test scripts
```

### Frontend
```bash
cd frontend
pnpm install

pnpm dev           # Port 3000
pnpm build         # tsc && vite build
pnpm lint          # ESLint
```

### Docker
```bash
docker-compose up --build    # 5 services (app, redis, postgres, influxdb, grafana)
docker-compose up -d
docker-compose down
```

---

## NOTES

- **Docker detection**: Settings auto-detect container runtime via `/.dockerenv` or `RUNNING_IN_DOCKER` env
- **Flat env vars supported**: `DB_HOST` maps to `database.host`, but `DATABASE__HOST` also works
- **Strategy auto-registration**: Import in `src/strategies/__init__.py` triggers registry
- **v2 migration**: v1 API removed. Use `examples/usage_example.py` as v2 reference.
- **No frontend tests**: Test structure exists only for Python
- **Coverage configured but unused**: `pytest-cov` installed but no coverage reports generated
- **USE IB for data source**
