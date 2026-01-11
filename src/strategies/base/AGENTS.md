# src/strategies/base/ - Strategy Development Guide

**Generated:** 2025-01-10
**Commit:** Current working tree
**Branch:** N/A

---

## OVERVIEW

Core strategy abstractions and registry system (v2 only).

---

## STRUCTURE

```
strategy.py      # Strategy (v2) + StrategyConfig + StrategyRegistry
```

---

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| **Strategy base** | `strategy.py` | Strategy with generate_signals/size_positions |
| **Registry** | `strategy.py` | StrategyRegistry singleton |
| **Auto-registration** | `../__init__.py:17` | Import triggers registry.register() |

---

## CONVENTIONS

- **Strategy interface**: `generate_signals()` → `size_positions()` using `Signal`/`OrderIntent` from core
- **Config**: StrategyConfig carries `strategy_id`, `timeframe`, and `params` (dict)
- **Registration**: Global `strategy_registry` singleton at module level

---

## ANTI-PATTERNS

- **Global registry**: `strategy_registry` is a singleton (no DI)
- **Large strategy files**: Keep strategy modules small and focused
