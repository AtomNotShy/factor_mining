#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.storage.backtest_store import BacktestStore
from src.evaluation.metrics.performance import PerformanceAnalyzer


@dataclass
class SeriesBundle:
    series: pd.Series
    has_timestamps: bool


def _parse_series(data: Any) -> Optional[SeriesBundle]:
    if data is None:
        return None

    if isinstance(data, dict):
        values = data.get("values")
        timestamps = data.get("timestamps")
        if values is None:
            # Fallback for dict-like series {index: value}
            try:
                items = list(data.items())
                if not items:
                    return None
                keys, vals = zip(*items)
                series = pd.Series(list(vals), index=list(keys))
                return SeriesBundle(series=series, has_timestamps=False)
            except Exception:
                return None

        if timestamps:
            try:
                index = pd.to_datetime(timestamps)
                return SeriesBundle(series=pd.Series(values, index=index), has_timestamps=True)
            except Exception:
                return SeriesBundle(series=pd.Series(values), has_timestamps=False)

        return SeriesBundle(series=pd.Series(values), has_timestamps=False)

    if isinstance(data, list):
        return SeriesBundle(series=pd.Series(data), has_timestamps=False)

    return None


def _periods_per_year(timeframe: Optional[str]) -> int:
    if not timeframe:
        return 252

    tf = timeframe.strip().lower()
    digits = "".join(ch for ch in tf if ch.isdigit())
    step = int(digits) if digits else 1

    if tf.endswith("m"):
        return int(252 * 390 / max(step, 1))
    if tf.endswith("h"):
        return int(252 * 6.5 / max(step, 1))
    if tf.endswith("d"):
        return int(252 / max(step, 1))
    if tf.endswith("w"):
        return int(52 / max(step, 1))

    return 252


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _max_drawdown_from_equity(equity: pd.Series) -> Tuple[float, int, Optional[int]]:
    if equity.empty:
        return float("nan"), 0, None

    peaks = equity.cummax()
    drawdown = equity / peaks - 1.0
    max_dd = float(drawdown.min())

    under = drawdown < 0
    max_duration = 0
    current = 0
    for is_under in under:
        if is_under:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0

    # Recovery duration in number of periods (positional index)
    recovery_duration = None
    try:
        trough_pos = int(np.nanargmin(drawdown.values))
        peak_value = peaks.iloc[trough_pos]
        post = equity.iloc[trough_pos:]
        recovered = post[post >= peak_value]
        if not recovered.empty:
            first_index = recovered.index[0]
            if isinstance(first_index, (int, np.integer)):
                recovery_duration = int(first_index) - trough_pos
            else:
                recovery_duration = int(equity.index.get_loc(first_index)) - trough_pos
    except Exception:
        recovery_duration = None

    return max_dd, max_duration, recovery_duration


def _rolling_sharpe(returns: pd.Series, window: int, periods_per_year: int) -> pd.Series:
    if len(returns) < window:
        return pd.Series(dtype=float)

    def _calc(r: pd.Series) -> float:
        vol = r.std() * np.sqrt(periods_per_year)
        if vol == 0:
            return np.nan
        return (r.mean() * periods_per_year) / vol

    return returns.rolling(window).apply(_calc, raw=False)


def _concentration_ratio(returns: pd.Series, top_n: int = 10) -> Optional[float]:
    gains = returns[returns > 0]
    if gains.empty:
        return None
    total = gains.sum()
    if total == 0:
        return None
    top = gains.sort_values(ascending=False).head(top_n).sum()
    return float(top / total)


def _rating_from_score(score: int) -> str:
    if score >= 6:
        return "A"
    if score == 5:
        return "A-"
    if score == 4:
        return "B+"
    if score == 3:
        return "B"
    if score == 2:
        return "C+"
    if score == 1:
        return "C"
    return "D"


def _build_llm_report(report: Dict[str, Any]) -> str:
    perf = report.get("performance", {})
    risk = report.get("risk", {})
    trade = report.get("trade", {})
    stability = report.get("stability", {})
    verdict = report.get("verdict", {})
    flags = report.get("risk_flags", [])

    lines = [
        "[BACKTEST_REPORT]",
        f"strategy: {report.get('strategy')}",
        f"symbol: {report.get('symbol')}",
        f"timeframe: {report.get('timeframe')}",
        f"period: {report.get('period_start')} -> {report.get('period_end')}",
        f"capital: {report.get('initial_capital')}",
        "",
        "performance:",
        f"  total_return: {perf.get('total_return')}",
        f"  cagr: {perf.get('cagr')}",
        f"  annual_return: {perf.get('annual_return')}",
        f"  volatility: {perf.get('volatility')}",
        f"  sharpe: {perf.get('sharpe')}",
        f"  sortino: {perf.get('sortino')}",
        f"  calmar: {perf.get('calmar')}",
        "",
        "risk:",
        f"  max_drawdown: {risk.get('max_drawdown')}",
        f"  max_drawdown_duration: {risk.get('max_drawdown_duration')}",
        f"  recovery_duration: {risk.get('recovery_duration')}",
        f"  skew: {risk.get('skew')}",
        f"  kurtosis: {risk.get('kurtosis')}",
        f"  var_95: {risk.get('var_95')}",
        f"  cvar_95: {risk.get('cvar_95')}",
        "",
        "trade_behavior:",
        f"  total_trades: {trade.get('total_trades')}",
        f"  win_rate: {trade.get('win_rate')}",
        f"  profit_loss_ratio: {trade.get('profit_loss_ratio')}",
        f"  concentration_top10_gains: {trade.get('gain_concentration_top10')}",
        "",
        "stability:",
        f"  rolling_sharpe_min_30: {stability.get('rolling_sharpe_min_30')}",
        f"  rolling_sharpe_median_30: {stability.get('rolling_sharpe_median_30')}",
        f"  rolling_return_min_30: {stability.get('rolling_return_min_30')}",
        "",
        "risk_flags:",
    ]
    for flag in flags:
        lines.append(f"  - {flag}")

    lines += [
        "",
        "verdict:",
        f"  rating: {verdict.get('rating')}",
        f"  suitable_for: {verdict.get('suitable_for')}",
        f"  primary_risks: {verdict.get('primary_risks')}",
        "",
        "next_actions:",
    ]
    for action in report.get("next_actions", []):
        lines.append(f"  - {action}")
    lines.append("[/BACKTEST_REPORT]")

    return "\n".join(lines)


def analyze_backtest(data: Dict[str, Any]) -> Dict[str, Any]:
    results = data.get("results", {})
    perf_stats = results.get("performance_stats", {})
    trade_stats = results.get("trade_stats", {})

    returns_bundle = _parse_series(results.get("returns"))
    equity_bundle = _parse_series(results.get("portfolio_value"))

    returns = returns_bundle.series if returns_bundle else pd.Series(dtype=float)
    equity = equity_bundle.series if equity_bundle else pd.Series(dtype=float)

    if returns.empty and not equity.empty:
        returns = equity.pct_change().fillna(0)

    periods_per_year = _periods_per_year(data.get("timeframe"))

    analyzer = PerformanceAnalyzer()
    comp = analyzer.comprehensive_analysis(returns, periods_per_year=periods_per_year)

    total_return = results.get("total_return")
    if total_return is None:
        total_return = perf_stats.get("total_return")
    if total_return is None and not equity.empty:
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)

    max_dd, max_dd_duration, recovery_duration = _max_drawdown_from_equity(equity)
    if np.isnan(max_dd):
        max_dd = comp.get("max_drawdown")

    rolling_30 = _rolling_sharpe(returns, 30, periods_per_year)
    rolling_return_30 = returns.rolling(30).sum() if len(returns) >= 30 else pd.Series(dtype=float)

    gain_concentration = _concentration_ratio(returns, top_n=10)

    win_rate = trade_stats.get("win_rate")
    if win_rate is None:
        win_rate = comp.get("win_rate")

    profit_loss_ratio = trade_stats.get("profit_loss_ratio")
    if profit_loss_ratio is None:
        profit_loss_ratio = comp.get("profit_loss_ratio")

    performance = {
        "total_return": total_return,
        "cagr": comp.get("cagr"),
        "annual_return": comp.get("annual_return"),
        "volatility": comp.get("volatility"),
        "sharpe": perf_stats.get("sharpe_ratio") or comp.get("sharpe_ratio"),
        "sortino": perf_stats.get("sortino_ratio") or comp.get("sortino_ratio"),
        "calmar": perf_stats.get("calmar_ratio") or comp.get("calmar_ratio"),
    }

    risk = {
        "max_drawdown": max_dd,
        "max_drawdown_duration": max_dd_duration,
        "recovery_duration": recovery_duration,
        "skew": comp.get("skewness"),
        "kurtosis": comp.get("kurtosis"),
        "var_95": comp.get("var_5pct"),
        "cvar_95": comp.get("cvar_5pct"),
        "max_daily_loss": float(returns.min()) if not returns.empty else None,
    }

    trade = {
        "total_trades": trade_stats.get("total_trades"),
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "buy_trades": trade_stats.get("buy_trades"),
        "sell_trades": trade_stats.get("sell_trades"),
        "gain_concentration_top10": gain_concentration,
    }

    stability = {
        "rolling_sharpe_min_30": float(rolling_30.min()) if not rolling_30.empty else None,
        "rolling_sharpe_median_30": float(rolling_30.median()) if not rolling_30.empty else None,
        "rolling_return_min_30": float(rolling_return_30.min()) if not rolling_return_30.empty else None,
    }

    flags: List[str] = []
    sharpe = _safe_float(performance.get("sharpe"))
    if sharpe is not None and sharpe < 0.8:
        flags.append("sharpe below 0.8")
    if max_dd is not None and max_dd < -0.2:
        flags.append("max drawdown worse than -20%")
    skew = _safe_float(risk.get("skew"))
    if skew is not None and skew < -0.5:
        flags.append("negative skew indicates tail risk")
    if profit_loss_ratio is not None and profit_loss_ratio < 1.0:
        flags.append("profit/loss ratio below 1.0")
    if win_rate is not None and win_rate < 0.45:
        flags.append("low win rate")

    score = 0
    if total_return is not None and total_return > 0:
        score += 1
    if sharpe is not None and sharpe > 1.0:
        score += 1
    if sharpe is not None and sharpe > 1.5:
        score += 1
    if max_dd is not None and max_dd > -0.1:
        score += 1
    if win_rate is not None and win_rate > 0.5:
        score += 1
    if profit_loss_ratio is not None and profit_loss_ratio > 1.2:
        score += 1

    verdict = {
        "rating": _rating_from_score(score),
        "suitable_for": "trend or momentum regimes" if sharpe and sharpe > 1.0 else "selective or defensive regimes",
        "primary_risks": ", ".join(flags) if flags else "none prominent",
    }

    report = {
        "backtest_id": data.get("backtest_id"),
        "strategy": data.get("strategy_name"),
        "symbol": data.get("symbol"),
        "timeframe": data.get("timeframe"),
        "period_start": data.get("backtest_period", {}).get("start_date"),
        "period_end": data.get("backtest_period", {}).get("end_date"),
        "initial_capital": data.get("config", {}).get("initial_capital"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_quality": {
            "has_returns": not returns.empty,
            "has_equity": not equity.empty,
            "has_trade_stats": bool(trade_stats),
        },
        "performance": performance,
        "risk": risk,
        "trade": trade,
        "stability": stability,
        "risk_flags": flags,
        "verdict": verdict,
        "next_actions": [
            "segment by regime to verify robustness",
            "stress test transaction cost and slippage",
            "validate out-of-sample period",
        ],
    }

    report["llm_report"] = _build_llm_report(report)
    return report


def _select_backtest(store: BacktestStore, backtest_id: Optional[str]) -> Dict[str, Any]:
    if backtest_id:
        data = store.get(backtest_id)
        if data is None:
            raise ValueError(f"Backtest id not found: {backtest_id}")
        data["backtest_id"] = backtest_id
        return data

    records = store.list(limit=1)
    if not records:
        raise ValueError("No backtest records found")
    record = records[0]
    data = store.get(record.id)
    if data is None:
        raise ValueError(f"Backtest id not found: {record.id}")
    data["backtest_id"] = record.id
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze backtest results for LLM-ready report")
    parser.add_argument("--id", dest="backtest_id", help="Backtest id to analyze")
    parser.add_argument("--format", choices=["json", "llm"], default="json")
    parser.add_argument("--out", help="Write output to file path")

    args = parser.parse_args()

    store = BacktestStore()
    data = _select_backtest(store, args.backtest_id)
    report = analyze_backtest(data)

    if args.format == "llm":
        output = report["llm_report"]
    else:
        output = json.dumps(report, indent=2, ensure_ascii=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
