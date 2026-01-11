#!/usr/bin/env python3
"""
命令行回测入口（v2）
示例：
  python backtest_cli.py --strategy us_etf_momentum --symbols SPY,QQQ,IWM --days 365
  python backtest_cli.py --strategy us_etf_momentum --params '{"target_positions":2}' --start 2023-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import numpy as np

from src.core.calendar import TradingCalendar
from src.core.context import RunContext, Environment
from src.evaluation.backtesting.engine import BacktestEngine
from src.evaluation.metrics.performance import PerformanceAnalyzer
from src.evaluation.metrics.benchmark import BenchmarkAnalyzer
from src.data.storage.backtest_store import BacktestStore
from src.strategies import strategy_registry  # 触发策略注册
from src.utils.logger import get_logger


logger = get_logger("backtest_cli")


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _coerce_value(raw: str) -> Any:
    text = raw.strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    if "," in text and not (text.startswith("\"") and text.endswith("\"")):
        return [item.strip() for item in text.split(",") if item.strip()]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_params(params_json: str, kv_params: List[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if params_json:
        params.update(json.loads(params_json))
    for item in kv_params:
        if "=" not in item:
            raise ValueError(f"参数格式应为 key=value: {item}")
        key, raw = item.split("=", 1)
        params[key.strip()] = _coerce_value(raw)
    return params


def _resolve_symbols(params: Dict[str, Any], cli_symbols: str) -> List[str]:
    if cli_symbols:
        return [s.strip().upper() for s in cli_symbols.split(",") if s.strip()]
    for key in ("etf_pool", "small_cap_pool", "universe", "symbols"):
        value = params.get(key)
        if isinstance(value, list) and value:
            return [str(s).upper() for s in value if str(s).strip()]
    return []


def _print_strategies() -> None:
    strategies = strategy_registry.list_strategies()
    if not strategies:
        print("未发现策略")
        return
    print("可用策略:")
    for name in strategies:
        print(f"- {name}")


def _print_table(title: str, rows: List[tuple[str, str]], header: tuple[str, str] = ("Metric", "Value")) -> None:
    if not rows:
        return
    key_width = max(len(header[0]), max(len(r[0]) for r in rows))
    val_width = max(len(header[1]), max(len(r[1]) for r in rows))
    total_width = key_width + val_width + 5
    title_line = title.center(total_width)

    top = f"┏{'━' * (key_width + 2)}┳{'━' * (val_width + 2)}┓"
    mid = f"┡{'━' * (key_width + 2)}╇{'━' * (val_width + 2)}┩"
    bottom = f"└{'─' * (key_width + 2)}┴{'─' * (val_width + 2)}┘"

    print(f"\n{title_line}")
    print(top)
    print(f"┃ {header[0].ljust(key_width)} ┃ {header[1].ljust(val_width)} ┃")
    print(mid)
    for key, value in rows:
        print(f"│ {key.ljust(key_width)} │ {value.ljust(val_width)} │")
    print(bottom)


def _print_matrix(title: str, headers: List[str], rows: List[List[str]]) -> None:
    if not rows:
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, col in enumerate(row):
            widths[idx] = max(widths[idx], len(col))

    total_width = sum(widths) + len(widths) * 3 + 1
    title_line = title.center(total_width)

    top = "┏" + "┳".join("━" * (w + 2) for w in widths) + "┓"
    mid = "┡" + "╇".join("━" * (w + 2) for w in widths) + "┩"
    bottom = "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"

    print(f"\n{title_line}")
    print(top)
    header_cells = [headers[i].ljust(widths[i]) for i in range(len(headers))]
    print("┃ " + " ┃ ".join(header_cells) + " ┃")
    print(mid)
    for row in rows:
        cells = [row[i].ljust(widths[i]) for i in range(len(headers))]
        print("│ " + " │ ".join(cells) + " │")
    print(bottom)


def _build_trade_rows(
    trades: List[Dict[str, Any]],
    initial_capital: float,
) -> List[List[str]]:
    grouped = defaultdict(list)
    for trade in trades:
        key = trade.get("group", "OTHER") or "OTHER"
        grouped[key].append(trade)

    rows: List[List[str]] = []
    total_trades = 0
    total_pnl = 0.0
    total_duration = 0.0
    total_win = 0
    total_draw = 0
    total_loss = 0
    total_ret_pct = 0.0

    for key in sorted(grouped.keys()):
        items = grouped[key]
        count = len(items)
        pnl = sum(t["pnl"] for t in items)
        avg_ret_pct = sum(t["ret_pct"] for t in items) / count if count else 0.0
        avg_duration = sum(t["duration"] or 0 for t in items) / count if count else 0.0
        wins = sum(1 for t in items if t["pnl"] > 0)
        draws = sum(1 for t in items if t["pnl"] == 0)
        losses = sum(1 for t in items if t["pnl"] < 0)
        win_rate = wins / count * 100 if count else 0.0

        total_trades += count
        total_pnl += pnl
        total_duration += avg_duration * count
        total_win += wins
        total_draw += draws
        total_loss += losses
        total_ret_pct += avg_ret_pct * count

        rows.append(
            [
                key,
                str(count),
                f"{avg_ret_pct * 100:.2f}",
                _fmt_currency(pnl),
                f"{(pnl / initial_capital) * 100:.2f}",
                _fmt_duration(avg_duration),
                str(wins),
                str(draws),
                str(losses),
                f"{win_rate:.1f}",
            ]
        )

    if total_trades > 0:
        rows.append(
            [
                "TOTAL",
                str(total_trades),
                f"{(total_ret_pct / total_trades) * 100:.2f}",
                _fmt_currency(total_pnl),
                f"{(total_pnl / initial_capital) * 100:.2f}",
                _fmt_duration(total_duration / total_trades),
                str(total_win),
                str(total_draw),
                str(total_loss),
                f"{total_win / total_trades * 100:.1f}",
            ]
        )

    return rows


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _fmt_pct(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "N/A"
    if math.isinf(val):
        return "Inf" if val > 0 else "-Inf"
    if math.isnan(val):
        return "N/A"
    return f"{val * 100:.2f}%"


def _fmt_num(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "N/A"
    if math.isinf(val):
        return "Inf" if val > 0 else "-Inf"
    if math.isnan(val):
        return "N/A"
    return f"{val:.2f}"


def _fmt_currency(value: Any) -> str:
    try:
        val = float(value)
    except Exception:
        return "N/A"
    if math.isinf(val):
        return "Inf" if val > 0 else "-Inf"
    if math.isnan(val):
        return "N/A"
    return f"${val:,.2f}"


def _fmt_duration(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "N/A"
    days = int(seconds // 86400)
    return f"{days}d"


def _to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


def _fmt_days(value: Any) -> str:
    if not _is_finite(value):
        return "N/A"
    return str(int(float(value)))


def _safe_date_str(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def _analyze_trades_simple(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
        }

    pnl_list: List[float] = []
    buy_trades: Dict[str, List[Dict[str, float]]] = {}

    for trade in trades:
        symbol = trade.get("symbol", "UNKNOWN")
        order_type = str(trade.get("order_type", "")).lower()
        size = abs(float(trade.get("size", 0) or 0))
        price = float(trade.get("price", 0) or 0)

        if symbol not in buy_trades:
            buy_trades[symbol] = []

        if order_type == "buy":
            buy_trades[symbol].append({"size": size, "price": price})
        elif order_type == "sell" and buy_trades[symbol]:
            buy = buy_trades[symbol].pop(0)
            pnl = (price - buy["price"]) * min(size, buy["size"])
            pnl_list.append(pnl)

    if not pnl_list:
        return {
            "total_trades": len(trades),
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
        }

    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p < 0]

    total_trades = len(pnl_list)
    winning_trades = len(wins)
    losing_trades = len(losses)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0

    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses)) / len(losses) if losses else 0.0

    profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0.0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    def _max_consecutive(values: List[float], positive: bool) -> int:
        max_run = 0
        current_run = 0
        for val in values:
            if (val > 0 and positive) or (val < 0 and not positive):
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "largest_win": max(wins) if wins else 0.0,
        "largest_loss": min(losses) if losses else 0.0,
        "consecutive_wins": _max_consecutive(pnl_list, positive=True),
        "consecutive_losses": _max_consecutive(pnl_list, positive=False),
    }


def _build_backtest_payload(
    result: Dict[str, Any],
    strategy_id: str,
    universe: List[str],
    start_date: date,
    end_date: date,
    initial_capital: float,
    commission_rate: float,
    slippage_rate: float,
    benchmark_symbol: str,
) -> Dict[str, Any]:
    fills = result.get("fills", [])
    requested_start_date = start_date
    effective_start_date = start_date
    first_fill_ts = None
    for fill in fills:
        ts_fill = getattr(fill, "ts_fill_utc", None)
        if ts_fill is None:
            continue
        if first_fill_ts is None or ts_fill < first_fill_ts:
            first_fill_ts = ts_fill
    if first_fill_ts is not None:
        first_fill_date = first_fill_ts.date()
        if first_fill_date > effective_start_date:
            effective_start_date = first_fill_date

    portfolio_df = result.get("portfolio_daily", pd.DataFrame())
    if not portfolio_df.empty:
        if "date" in portfolio_df.columns:
            trimmed = portfolio_df[portfolio_df["date"] >= effective_start_date]
        else:
            trimmed = portfolio_df[portfolio_df.index >= pd.Timestamp(effective_start_date)]
        if not trimmed.empty:
            portfolio_df = trimmed
    portfolio_values = portfolio_df["equity"].tolist() if not portfolio_df.empty else []
    total_return = result.get("total_return", 0.0)
    if isinstance(total_return, pd.Series):
        total_return = float(total_return.iloc[-1]) if len(total_return) > 0 else 0.0

    backtest_days = (end_date - effective_start_date).days
    annualized_return = (1 + total_return) ** (365.0 / max(backtest_days, 1)) - 1 if backtest_days > 0 else 0.0

    if not portfolio_df.empty:
        if "date" in portfolio_df.columns:
            equity_index = pd.to_datetime(portfolio_df["date"])
        else:
            equity_index = pd.to_datetime(portfolio_df.index)
        equity_series = pd.Series(portfolio_values, index=equity_index)
    else:
        equity_series = pd.Series([initial_capital])

    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    returns = equity_series.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if len(returns) > 1 and returns.std() > 0 else 0.0

    bars = result.get("bars", pd.DataFrame())
    price_data = {"timestamps": [], "close": [], "volume": []}
    if not bars.empty and "symbol" in bars.columns:
        first_symbol = universe[0] if universe else "SPY"
        symbol_bars = bars[bars["symbol"] == first_symbol]
        if not symbol_bars.empty:
            symbol_bars = symbol_bars.loc[symbol_bars.index >= pd.Timestamp(effective_start_date)]
            if not symbol_bars.empty:
                price_data["timestamps"] = [str(d)[:10] for d in symbol_bars.index.tolist()]
                price_data["close"] = symbol_bars["close"].tolist()
                price_data["volume"] = symbol_bars["volume"].tolist() if "volume" in symbol_bars.columns else []

    trades = []
    for i, fill in enumerate(fills):
        ts_fill = getattr(fill, "ts_fill_utc", None)
        side = getattr(fill, "side", None)
        side_value = side.value.lower() if side is not None else ""
        qty = float(getattr(fill, "qty", 0) or 0)
        price = float(getattr(fill, "price", 0) or 0)
        trades.append(
            {
                "id": str(i + 1),
                "timestamp": ts_fill.isoformat() if ts_fill else None,
                "symbol": getattr(fill, "symbol", ""),
                "order_type": side_value,
                "price": price,
                "size": qty,
                "amount": price * qty,
                "commission": float(getattr(fill, "fee", 0) or 0),
            }
        )
    trades.sort(key=lambda t: t.get("timestamp") or "")

    trade_stats = _analyze_trades_simple(trades)

    daily_returns = equity_series.pct_change().dropna()
    winning_days = int((daily_returns > 0).sum())
    losing_days = int((daily_returns < 0).sum())
    total_days = len(daily_returns)
    largest_daily_win = float(daily_returns.max()) if len(daily_returns) > 0 else 0.0
    largest_daily_loss = float(daily_returns.min()) if len(daily_returns) > 0 else 0.0

    drawdown_series = [float(x) for x in drawdown.tolist()]
    max_dd_value = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    max_dd_idx = int(np.argmin(drawdown.to_numpy())) if len(drawdown) > 0 else 0
    peak_idx = int(np.argmax(drawdown[: max_dd_idx + 1].to_numpy())) if max_dd_idx > 0 else 0

    if "date" in portfolio_df.columns:
        portfolio_dates = [_safe_date_str(d) for d in portfolio_df["date"].tolist()]
    else:
        portfolio_dates = [_safe_date_str(d) for d in portfolio_df.index.tolist()]

    if not portfolio_dates:
        portfolio_dates = [_safe_date_str(start_date)]

    max_dd_start = portfolio_dates[peak_idx] if peak_idx < len(portfolio_dates) else portfolio_dates[0]
    max_dd_end = portfolio_dates[max_dd_idx] if max_dd_idx < len(portfolio_dates) else portfolio_dates[-1]

    recovery_days = 0
    for i in range(max_dd_idx + 1, len(equity_series)):
        if equity_series.iloc[i] >= equity_series.iloc[peak_idx]:
            recovery_days = i - max_dd_idx
            break

    dd_pct = (drawdown * 100).abs()
    ulcer_index_val = (dd_pct.pow(2).mean() ** 0.5) if len(dd_pct) > 0 else 0.0

    avg_drawdown = float(drawdown.mean()) if len(drawdown) > 0 else 0.0

    portfolio_equity = [float(x) for x in portfolio_values]
    strategy_equity = portfolio_equity
    benchmark_equity = [
        initial_capital * (1 + i * total_return / max(len(portfolio_values) - 1, 1))
        for i in range(len(portfolio_values))
    ]
    excess_returns = [
        (portfolio_equity[i] - benchmark_equity[i]) / benchmark_equity[i] if benchmark_equity[i] > 0 else 0
        for i in range(len(portfolio_values))
    ]
    comparison_timestamps = portfolio_dates

    benchmark_return = total_return
    benchmark_volatility = float(daily_returns.std() * (252 ** 0.5)) if len(daily_returns) > 0 else 0.0
    excess_return = 0.0
    alpha = 0.0
    beta = 1.0
    tracking_error = 0.0
    information_ratio = 0.0
    r_squared = 1.0

    if benchmark_symbol and not returns.empty:
        try:
            analyzer = BenchmarkAnalyzer(benchmark_symbol)
            start_date_str = effective_start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            benchmark_df = analyzer.get_benchmark_data(
                start_date=start_date_str,
                end_date=end_date_str,
                data_source="ib",
            )
            if benchmark_df is None or benchmark_df.empty:
                benchmark_df = analyzer.get_benchmark_data(
                    start_date=start_date_str,
                    end_date=end_date_str,
                    data_source="local",
                )

            if benchmark_df is not None and not benchmark_df.empty:
                if "close" not in benchmark_df.columns and "c" in benchmark_df.columns:
                    benchmark_df["close"] = benchmark_df["c"]
                    benchmark_df = benchmark_df.drop(columns=["c"])
                if "close" in benchmark_df.columns:
                    benchmark_returns = analyzer.calculate_returns_from_prices(benchmark_df["close"])
                    benchmark_metrics = analyzer.calculate_benchmark_metrics(benchmark_returns)
                    benchmark_analysis = analyzer.comprehensive_benchmark_analysis(returns, benchmark_returns)
                    comparison = analyzer.get_equity_comparison(
                        returns, benchmark_returns, initial_value=initial_capital
                    )

                    benchmark_return = float(benchmark_metrics.total_return)
                    benchmark_volatility = float(benchmark_metrics.volatility)
                    excess_return = float(benchmark_analysis.get("excess_return", 0.0) or 0.0)
                    alpha = float(benchmark_analysis.get("alpha", 0.0) or 0.0)
                    beta = float(benchmark_analysis.get("beta", 1.0) or 1.0)
                    tracking_error = float(benchmark_analysis.get("tracking_error", 0.0) or 0.0)
                    information_ratio = float(benchmark_analysis.get("information_ratio", 0.0) or 0.0)
                    r_squared = float(benchmark_analysis.get("r_squared", 0.0) or 0.0)

                    strategy_equity = comparison.get("strategy_equity", strategy_equity)
                    benchmark_equity = comparison.get("benchmark_equity", benchmark_equity)
                    excess_returns = comparison.get("excess_returns", excess_returns)
                    comparison_timestamps = [str(ts) for ts in comparison.get("timestamps", comparison_timestamps)]
        except Exception as exc:
            logger.warning(f"基准曲线计算失败，回退线性基准: {exc}")

    trading_stats_data = {
        "total_signals": len(result.get("signals", [])),
        "total_orders": len(result.get("orders", [])),
        "total_fills": len(result.get("fills", [])),
        "total_trades": trade_stats.get("total_trades", 0),
        "winning_trades": trade_stats.get("winning_trades", 0),
        "losing_trades": trade_stats.get("losing_trades", 0),
        "win_rate": trade_stats.get("win_rate", 0.0),
        "daily_win_rate": winning_days / total_days if total_days > 0 else 0.0,
        "profit_loss_ratio": trade_stats.get("profit_loss_ratio", 0.0),
        "profit_factor": trade_stats.get("profit_factor", 0.0),
        "expectancy": trade_stats.get("expectancy", 0.0),
        "consecutive_wins": trade_stats.get("consecutive_wins", 0),
        "consecutive_losses": trade_stats.get("consecutive_losses", 0),
        "largest_win": trade_stats.get("largest_win", 0.0),
        "largest_loss": trade_stats.get("largest_loss", 0.0),
        "winning_days": winning_days,
        "losing_days": losing_days,
    }

    max_drawdown_window = {
        "start_date": max_dd_start,
        "end_date": max_dd_end,
        "drawdown_pct": max_dd_value,
        "duration_days": max_dd_idx - peak_idx if max_dd_idx >= peak_idx else 0,
        "recovery_days": recovery_days,
        "equity_at_peak": float(equity_series.iloc[peak_idx]) if peak_idx < len(equity_series) else initial_capital,
        "equity_at_trough": float(equity_series.iloc[max_dd_idx]) if max_dd_idx < len(equity_series) else initial_capital,
    }

    enhanced_metrics_data = {
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "annual_return": annualized_return,
        "excess_return": excess_return,
        "best_day": largest_daily_win,
        "worst_day": largest_daily_loss,
        "max_drawdown": max_dd_value,
        "annual_volatility": float(daily_returns.std() * (252 ** 0.5)) if len(daily_returns) > 0 else 0.0,
        "benchmark_volatility": benchmark_volatility,
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": (daily_returns.mean() / daily_returns[daily_returns < 0].std() * (252 ** 0.5))
        if len(daily_returns) > 1 and daily_returns[daily_returns < 0].std() > 0
        else 0.0,
        "calmar_ratio": annualized_return / abs(max_dd_value) if max_dd_value != 0 else 0.0,
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        "information_ratio": information_ratio,
        "tracking_error": tracking_error,
        "drawdown_series": drawdown_series,
        "max_drawdown_window": max_drawdown_window,
        "ulcer_index": float(ulcer_index_val) if ulcer_index_val else 0.0,
        "burke_ratio": 0.0,
        "time_in_market": 1.0,
        "avg_drawdown": avg_drawdown,
        "avg_drawdown_duration": 5,
        **trading_stats_data,
    }

    return {
        "run_id": result.get("run_id") or "",
        "strategy_name": strategy_id,
        "symbol": ",".join(universe) if universe else "SPY",
        "universe": universe,
        "timeframe": "1d",
        "backtest_period": {
            "start_date": _safe_date_str(effective_start_date),
            "end_date": _safe_date_str(end_date),
            "days": backtest_days,
            "requested_start_date": _safe_date_str(requested_start_date),
        },
        "config": {
            "initial_capital": initial_capital,
            "commission_rate": commission_rate,
            "slippage_rate": slippage_rate,
        },
        "price_data": price_data,
        "trades": trades,
        "results": {
            "final_value": result.get("final_equity", 0.0),
            "total_return": total_return,
            "annual_return": annualized_return,
            "best_day": largest_daily_win,
            "worst_day": largest_daily_loss,
            "daily_win_rate": winning_days / total_days if total_days > 0 else 0.0,
        },
        "enhanced_metrics": enhanced_metrics_data,
        "equity_comparison": {
            "strategy_equity": strategy_equity,
            "benchmark_equity": benchmark_equity,
            "excess_returns": excess_returns,
            "timestamps": comparison_timestamps,
        },
        "benchmark_data": {
            "symbol": benchmark_symbol,
            "alpha": alpha,
            "beta": beta,
            "r_squared": r_squared,
            "information_ratio": information_ratio,
            "tracking_error": tracking_error,
        },
        "performance": {
            "final_equity": result.get("final_equity", 0.0),
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_dd_value,
            "sharpe_ratio": float(sharpe_ratio),
        },
        "trading_stats": trading_stats_data,
        "portfolio": {
            "dates": portfolio_dates,
            "values": portfolio_equity,
        },
        "timestamp": datetime.now().isoformat(),
    }


async def main() -> int:
    class _HelpOnErrorParser(argparse.ArgumentParser):
        def error(self, message: str) -> None:
            self.print_help(sys.stderr)
            self.exit(2, f"\nerror: {message}\n")

    parser = _HelpOnErrorParser(description="策略回测 CLI (v2)")
    parser.add_argument("--list-strategies", action="store_true", help="列出可用策略")
    parser.add_argument("--strategy", default="us_etf_momentum", help="策略名称")
    parser.add_argument("--params", default="", help="策略参数 JSON 字符串")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="单个参数 key=value，可重复使用",
    )
    parser.add_argument("--symbols", default="", help="标的列表，逗号分隔")
    parser.add_argument("--start", default="", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=365, help="回测天数（仅在未指定 start 时生效）")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="初始资金")
    parser.add_argument("--commission", type=float, default=0.0005, help="手续费率")
    parser.add_argument("--slippage", type=float, default=0.0002, help="滑点率")
    parser.add_argument("--benchmark", default="", help="基准标的（默认使用首个标的）")
    parser.add_argument(
        "--auto-download",
        dest="auto_download",
        action="store_true",
        default=True,
        help="自动补齐数据（默认开启）",
    )
    parser.add_argument(
        "--no-auto-download",
        dest="auto_download",
        action="store_false",
        help="关闭自动补齐数据",
    )

    argv = sys.argv[1:]
    # 兼容常见误用：`-help`（argparse 会把它拆成 `-h` + `elp`）
    argv = ["--help" if arg == "-help" else arg for arg in argv]

    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.list_strategies:
        _print_strategies()
        return 0

    strategy = strategy_registry.get_strategy(args.strategy)
    if strategy is None:
        print(f"未找到策略: {args.strategy}")
        _print_strategies()
        return 1

    params = _parse_params(args.params, args.param)
    if params:
        strategy.config.params.update(params)

    symbols = _resolve_symbols(strategy.config.params, args.symbols)
    if symbols:
        if "small_cap_pool" in strategy.config.params:
            strategy.config.params["small_cap_pool"] = symbols
        else:
            strategy.config.params["etf_pool"] = symbols

    end_date = _parse_date(args.end) if args.end else date.today()
    if args.start:
        start_date = _parse_date(args.start)
    else:
        start_date = end_date - timedelta(days=args.days)

    universe = symbols or strategy.config.params.get("etf_pool") or ["SPY"]
    universe = [str(s).upper() for s in universe]
    benchmark_symbol = ""
    if args.benchmark:
        benchmark_symbol = args.benchmark.strip().upper()
    if not benchmark_symbol:
        benchmark_symbol = str(strategy.config.params.get("benchmark_symbol", "")).strip().upper()
    if not benchmark_symbol:
        benchmark_symbol = universe[0] if universe else ""

    engine = BacktestEngine(
        initial_capital=args.initial_capital,
        commission_rate=args.commission,
        slippage_rate=args.slippage,
    )
    ctx = RunContext.create(
        env=Environment.RESEARCH,
        config=strategy.config.params,
        trading_calendar=TradingCalendar(),
    )

    result = await engine.run(
        strategies=[strategy],
        universe=universe,
        start=start_date,
        end=end_date,
        ctx=ctx,
        auto_download=args.auto_download,
    )

    if "error" in result:
        print(f"❌ 回测失败: {result['error']}")
        return 1

    print("\n✅ 回测完成")
    summary_rows = [
        ("Strategy", strategy.strategy_id),
        ("Universe", ", ".join(universe)),
        ("Period", f"{start_date} ~ {end_date}"),
        ("Initial Capital", _fmt_currency(args.initial_capital)),
        ("Final Equity", _fmt_currency(result.get("final_equity"))),
        ("Total Return", _fmt_pct(result.get("total_return"))),
        ("Signals", str(len(result["signals"]))),
        ("Orders", str(len(result["orders"]))),
        ("Fills", str(len(result["fills"]))),
    ]
    _print_table("BACKTEST SUMMARY", summary_rows)

    portfolio_df = result.get("portfolio_daily")
    returns = None
    if isinstance(portfolio_df, pd.DataFrame) and not portfolio_df.empty:
        if "date" in portfolio_df.columns:
            dates = pd.to_datetime(portfolio_df["date"])
        else:
            dates = pd.RangeIndex(len(portfolio_df))
        if "daily_return" in portfolio_df.columns:
            returns = pd.Series(portfolio_df["daily_return"].values, index=dates)
        elif "equity" in portfolio_df.columns:
            returns = pd.Series(portfolio_df["equity"].pct_change().fillna(0).values, index=dates)

    if returns is not None and not returns.empty:
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.comprehensive_analysis(returns)
        best_day = returns.max()
        worst_day = returns.min()
        win_days = int((returns > 0).sum())
        total_days = int(len(returns))
        daily_win_rate = win_days / total_days if total_days > 0 else None

        trade_records = []
        for fill in result.get("fills", []):
            side = getattr(fill.side, "value", str(fill.side)).lower()
            trade_records.append(
                {
                    "order_type": side,
                    "size": float(getattr(fill, "qty", 0.0)),
                    "price": float(getattr(fill, "price", 0.0)),
                    "commission": float(getattr(fill, "fee", 0.0)),
                    "timestamp": getattr(fill, "ts_fill_utc", None),
                    "symbol": getattr(fill, "symbol", ""),
                }
            )

        trade_stats = analyzer.calculate_trade_analysis(trade_records, returns)
        expectancy_info = analyzer.calculate_expectancy(trade_records)
        profit_factor = analyzer.calculate_profit_factor(trade_records)

        daily_pnl = None
        if isinstance(portfolio_df, pd.DataFrame) and "daily_pnl" in portfolio_df.columns:
            daily_pnl = float(pd.Series(portfolio_df["daily_pnl"]).mean())

        alpha = None
        beta = None
        benchmark_returns = None
        bars = result.get("bars")
        if benchmark_symbol and isinstance(bars, pd.DataFrame) and not bars.empty:
            if "symbol" in bars.columns:
                bench_bars = bars[bars["symbol"] == benchmark_symbol]
            else:
                bench_bars = bars
            if not bench_bars.empty and "close" in bench_bars.columns:
                benchmark_returns = BenchmarkAnalyzer(benchmark_symbol).calculate_returns_from_prices(
                    bench_bars["close"]
                )
                if benchmark_returns is not None and not benchmark_returns.empty:
                    benchmark_analysis = BenchmarkAnalyzer(benchmark_symbol).comprehensive_benchmark_analysis(
                        returns, benchmark_returns
                    )
                    alpha = benchmark_analysis.get("alpha")
                    beta = benchmark_analysis.get("beta")

        perf_rows = [
            ("Trading Days", str(total_days)),
            ("CAGR", _fmt_pct(metrics.get("cagr"))),
            ("Annual Return", _fmt_pct(metrics.get("annual_return"))),
            ("Volatility", _fmt_pct(metrics.get("volatility"))),
            ("Sharpe Ratio", _fmt_num(metrics.get("sharpe_ratio"))),
            ("Sortino Ratio", _fmt_num(metrics.get("sortino_ratio"))),
            ("Calmar Ratio", _fmt_num(metrics.get("calmar_ratio"))),
            ("Max Drawdown", _fmt_pct(metrics.get("max_drawdown"))),
            ("Max DD Duration (days)", _fmt_days(metrics.get("max_drawdown_duration"))),
            ("Best Day", _fmt_pct(best_day)),
            ("Worst Day", _fmt_pct(worst_day)),
            ("Daily Win Rate", _fmt_pct(daily_win_rate)),
            ("Skew / Kurtosis", f"{_fmt_num(metrics.get('skewness'))} / {_fmt_num(metrics.get('kurtosis'))}"),
            ("VaR 5%", _fmt_pct(metrics.get("var_5pct"))),
            ("CVaR 5%", _fmt_pct(metrics.get("cvar_5pct"))),
            ("Benchmark", benchmark_symbol or "N/A"),
            ("Alpha", _fmt_pct(alpha)),
            ("Beta", _fmt_num(beta)),
        ]
        _print_table("PERFORMANCE SUMMARY", perf_rows)

        trade_rows = [
            ("Initial Capital", _fmt_currency(args.initial_capital)),
            ("Total Trades", str(trade_stats.get("total_trades", 0))),
            ("Buy / Sell", f"{trade_stats.get('buy_trades', 0)} / {trade_stats.get('sell_trades', 0)}"),
            ("Trades / Day", _fmt_num(trade_stats.get("trades_per_day"))),
            ("Avg Buy Amount", _fmt_currency(trade_stats.get("avg_buy_amount"))),
            ("Avg Sell Amount", _fmt_currency(trade_stats.get("avg_sell_amount"))),
            ("Win Rate (Trades)", _fmt_pct(expectancy_info.get("win_rate"))),
            ("Profit Factor", _fmt_num(profit_factor)),
            ("Expectancy", _fmt_currency(expectancy_info.get("expectancy"))),
        ]
        if daily_pnl is not None:
            trade_rows.append(("Avg Daily PnL", _fmt_currency(daily_pnl)))
        _print_table("TRADE SUMMARY", trade_rows, header=("Trade Metric", "Value"))

        closed_trades: List[Dict[str, Any]] = []
        open_lots: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for fill in result.get("fills", []):
            side = getattr(fill.side, "value", str(fill.side)).lower()
            symbol = getattr(fill, "symbol", "UNKNOWN")
            qty = float(getattr(fill, "qty", 0.0))
            price = float(getattr(fill, "price", 0.0))
            ts = _to_datetime(getattr(fill, "ts_fill_utc", None))
            metadata = getattr(fill, "metadata", {}) or {}
            entry_tag = metadata.get("entry_tag") or "OTHER"
            exit_reason = metadata.get("exit_reason") or "OTHER"

            if side == "buy":
                if qty > 0:
                    open_lots[symbol].append(
                        {
                            "qty": qty,
                            "price": price,
                            "ts": ts,
                            "entry_tag": entry_tag,
                        }
                    )
                continue
            if side != "sell" or qty <= 0:
                continue

            remaining = qty
            lots = open_lots.get(symbol, [])
            while remaining > 0 and lots:
                lot = lots[0]
                lot_qty = float(lot["qty"])
                close_qty = min(lot_qty, remaining)
                entry_price = float(lot["price"])
                pnl = (price - entry_price) * close_qty
                ret_pct = (price - entry_price) / entry_price if entry_price else 0.0
                entry_ts = lot.get("ts")
                duration = None
                if entry_ts and ts:
                    duration = (ts - entry_ts).total_seconds()
                closed_trades.append(
                    {
                        "symbol": symbol,
                        "qty": close_qty,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl": pnl,
                        "ret_pct": ret_pct,
                        "duration": duration,
                        "entry_tag": lot.get("entry_tag") or "OTHER",
                        "exit_reason": exit_reason,
                    }
                )
                remaining -= close_qty
                lot_qty -= close_qty
                if lot_qty <= 0:
                    lots.pop(0)
                else:
                    lot["qty"] = lot_qty

        if closed_trades:
            report_rows = _build_trade_rows(
                [
                    {**trade, "group": trade.get("symbol", "UNKNOWN")}
                    for trade in closed_trades
                ],
                args.initial_capital,
            )
            headers = [
                "Pair",
                "Trades",
                "Avg Profit %",
                "Tot Profit USD",
                "Tot Profit %",
                "Avg Duration",
                "Win",
                "Draw",
                "Loss",
                "Win%",
            ]
            _print_matrix("BACKTESTING REPORT", headers, report_rows)

            entry_rows = _build_trade_rows(
                [
                    {**trade, "group": trade.get("entry_tag", "OTHER")}
                    for trade in closed_trades
                ],
                args.initial_capital,
            )
            entry_headers = [
                "Enter Tag",
                "Entries",
                "Avg Profit %",
                "Tot Profit USD",
                "Tot Profit %",
                "Avg Duration",
                "Win",
                "Draw",
                "Loss",
                "Win%",
            ]
            _print_matrix(
                "ENTER TAG STATS",
                entry_headers,
                entry_rows,
            )

            exit_rows = _build_trade_rows(
                [
                    {**trade, "group": trade.get("exit_reason", "OTHER")}
                    for trade in closed_trades
                ],
                args.initial_capital,
            )
            exit_headers = [
                "Exit Reason",
                "Exits",
                "Avg Profit %",
                "Tot Profit USD",
                "Tot Profit %",
                "Avg Duration",
                "Win",
                "Draw",
                "Loss",
                "Win%",
            ]
            _print_matrix(
                "EXIT REASON STATS",
                exit_headers,
                exit_rows,
            )

            mixed_rows = _build_trade_rows(
                [
                    {
                        **trade,
                        "group": f"{trade.get('entry_tag', 'OTHER')} / {trade.get('exit_reason', 'OTHER')}",
                    }
                    for trade in closed_trades
                ],
                args.initial_capital,
            )
            mixed_headers = [
                "Enter Tag",
                "Exit Reason",
                "Trades",
                "Avg Profit %",
                "Tot Profit USD",
                "Tot Profit %",
                "Avg Duration",
                "Win",
                "Draw",
                "Loss",
                "Win%",
            ]
            mixed_display_rows = []
            for row in mixed_rows:
                if row[0] == "TOTAL":
                    mixed_display_rows.append(
                        ["TOTAL", "", row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]]
                    )
                    continue
                if " / " in row[0]:
                    enter_tag, exit_reason = row[0].split(" / ", 1)
                else:
                    enter_tag, exit_reason = row[0], "OTHER"
                mixed_display_rows.append(
                    [enter_tag, exit_reason, row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]]
                )
            _print_matrix(
                "MIXED TAG STATS",
                mixed_headers,
                mixed_display_rows,
            )

        if open_lots:
            last_prices = {}
            bars = result.get("bars")
            if isinstance(bars, pd.DataFrame) and not bars.empty and "symbol" in bars.columns:
                latest = bars.sort_index().groupby("symbol").tail(1)
                last_prices = {
                    row["symbol"]: float(row["close"]) for _, row in latest.iterrows()
                }

            open_rows: List[List[str]] = []
            total_open_trades = 0
            total_open_pnl = 0.0
            total_open_duration = 0.0
            total_open_win = 0
            total_open_draw = 0
            total_open_loss = 0
            total_open_ret_pct = 0.0

            end_dt = datetime.combine(end_date, datetime.min.time())

            for symbol in sorted(open_lots.keys()):
                lots = open_lots[symbol]
                if not lots:
                    continue
                last_price = last_prices.get(symbol)
                if last_price is None:
                    continue

                trade_pnls = []
                durations = []
                ret_pcts = []
                for lot in lots:
                    qty = float(lot["qty"])
                    entry_price = float(lot["price"])
                    pnl = (last_price - entry_price) * qty
                    ret_pct = (last_price - entry_price) / entry_price if entry_price else 0.0
                    entry_ts = lot.get("ts")
                    duration = (end_dt - entry_ts).total_seconds() if entry_ts else None
                    trade_pnls.append(pnl)
                    ret_pcts.append(ret_pct)
                    durations.append(duration or 0)

                count = len(lots)
                pnl_total = sum(trade_pnls)
                avg_ret_pct = sum(ret_pcts) / count if count else 0.0
                avg_duration = sum(durations) / count if count else 0.0
                wins = sum(1 for pnl in trade_pnls if pnl > 0)
                draws = sum(1 for pnl in trade_pnls if pnl == 0)
                losses = sum(1 for pnl in trade_pnls if pnl < 0)
                win_rate = wins / count * 100 if count else 0.0

                total_open_trades += count
                total_open_pnl += pnl_total
                total_open_duration += avg_duration * count
                total_open_win += wins
                total_open_draw += draws
                total_open_loss += losses
                total_open_ret_pct += avg_ret_pct * count

                open_rows.append(
                    [
                        symbol,
                        str(count),
                        f"{avg_ret_pct * 100:.2f}",
                        _fmt_currency(pnl_total),
                        f"{(pnl_total / args.initial_capital) * 100:.2f}",
                        _fmt_duration(avg_duration),
                        str(wins),
                        str(draws),
                        str(losses),
                        f"{win_rate:.1f}",
                    ]
                )

            if total_open_trades > 0:
                open_rows.append(
                    [
                        "TOTAL",
                        str(total_open_trades),
                        f"{(total_open_ret_pct / total_open_trades) * 100:.2f}",
                        _fmt_currency(total_open_pnl),
                        f"{(total_open_pnl / args.initial_capital) * 100:.2f}",
                        _fmt_duration(total_open_duration / total_open_trades),
                        str(total_open_win),
                        str(total_open_draw),
                        str(total_open_loss),
                        f"{total_open_win / total_open_trades * 100:.1f}",
                    ]
                )

                headers = [
                    "Pair",
                    "Trades",
                    "Avg Profit %",
                    "Tot Profit USD",
                    "Tot Profit %",
                    "Avg Duration",
                    "Win",
                    "Draw",
                    "Loss",
                    "Win%",
                ]
                _print_matrix("LEFT OPEN TRADES REPORT", headers, open_rows)

    backtest_payload = _build_backtest_payload(
        result=result,
        strategy_id=strategy.strategy_id,
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.initial_capital,
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        benchmark_symbol=benchmark_symbol,
    )
    store = BacktestStore()
    backtest_id = backtest_payload.get("run_id") or store.generate_id()
    backtest_payload["run_id"] = backtest_id
    try:
        store.save(backtest_id, backtest_payload)
    except Exception as exc:
        logger.warning(f"保存回测结果失败: {exc}")
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
