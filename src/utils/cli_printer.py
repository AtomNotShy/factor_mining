import math
import json
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

class CLIPrinter:
    """CLI 输出格式化工具"""

    @staticmethod
    def _is_finite(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except Exception:
            return False

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if seconds is None or seconds < 0:
            return "N/A"
        days = int(seconds // 86400)
        return f"{days}d"
    
    @staticmethod
    def _print_table(title: str, rows: List[Tuple[str, str]], header: Tuple[str, str] = ("Metric", "Value")) -> None:
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

    @staticmethod
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

    @classmethod
    def match_trades(cls, fills: List[Any]) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Matching buy/sell fills to generate closed trades and identify open positions.
        Returns (closed_trades, open_lots)
        """
        closed_trades: List[Dict[str, Any]] = []
        open_lots: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Sort fills by timestamp
        fills_sorted = sorted(fills, key=lambda f: getattr(f, "ts_fill_utc", datetime.min))

        for fill in fills_sorted:
            # Handle both Fill objects (with enum side) and dicts (serialized)
            side_raw = fill.side if isinstance(fill, dict) else getattr(fill, "side", None)
            if isinstance(side_raw, dict):
                side = side_raw.get("value", str(side_raw)).lower()
            else:
                side = getattr(side_raw, "value", str(side_raw)).lower()
            symbol = getattr(fill, "symbol", "UNKNOWN")
            qty = float(getattr(fill, "qty", 0.0))
            price = float(getattr(fill, "price", 0.0))
            ts = getattr(fill, "ts_fill_utc", None)
            if ts and isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            
            metadata = getattr(fill, "metadata", {}) or {}
            entry_tag = metadata.get("entry_tag") or "OTHER"
            exit_reason = metadata.get("exit_reason") or "OTHER"

            # Buying adds to open lots
            if side == "buy":
                if qty > 0:
                    open_lots[symbol].append({
                        "qty": qty,
                        "price": price,
                        "ts": ts,
                        "entry_tag": entry_tag,
                    })
                continue
            
            # Selling reduces open lots (FIFO)
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
                
                closed_trades.append({
                    "symbol": symbol,
                    "qty": close_qty,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": pnl,
                    "ret_pct": ret_pct,
                    "duration": duration,
                    "entry_tag": lot.get("entry_tag") or "OTHER",
                    "exit_reason": exit_reason,
                    "exit_ts": ts
                })
                
                remaining -= close_qty
                lot_qty -= close_qty
                
                if lot_qty <= 1e-9: # Float tolerance
                    lots.pop(0)
                else:
                    lot["qty"] = lot_qty

        return closed_trades, open_lots

    @classmethod
    def _build_trade_rows(cls, trades: List[Dict[str, Any]], initial_capital: float) -> List[List[str]]:
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

            rows.append([
                str(key),
                str(count),
                f"{avg_ret_pct * 100:.2f}",
                cls._fmt_currency(pnl),
                f"{(pnl / initial_capital) * 100:.2f}",
                cls._fmt_duration(avg_duration),
                str(wins),
                str(draws),
                str(losses),
                f"{win_rate:.1f}",
            ])

        if total_trades > 0:
            rows.append([
                "TOTAL",
                str(total_trades),
                f"{(total_ret_pct / total_trades) * 100:.2f}",
                cls._fmt_currency(total_pnl),
                f"{(total_pnl / initial_capital) * 100:.2f}",
                cls._fmt_duration(total_duration / total_trades),
                str(total_win),
                str(total_draw),
                str(total_loss),
                f"{total_win / total_trades * 100:.1f}",
            ])

        return rows

    @classmethod
    def print_report(cls, summary: Dict[str, Any], engine_result: Dict[str, Any]):
        """
        Main entry point to print the full backtest report.
        Args:
            summary: Processed summary from BacktestResultProcessor
            engine_result: Raw result from BacktestEngine (contains fills)
        """
        # 1. Backtest Summary
        m = summary.get('enhanced_metrics', {})
        results = summary.get('results', {})
        config = summary.get('config', {})
        backtest_period = summary.get('backtest_period', {})
        
        initial_capital = config.get('initial_capital', 100000.0)
        
        summary_rows = [
            ("Strategy", summary.get("strategy_name", "unknown")),
            ("Universe", summary.get("symbol", "")),
            ("Period", f"{backtest_period.get('start_date')} ~ {backtest_period.get('end_date')}"),
            ("Duration", f"{backtest_period.get('days')} days"),
            ("Initial Capital", cls._fmt_currency(initial_capital)),
            ("Final Equity", cls._fmt_currency(results.get("final_value"))),
            ("Total Return", cls._fmt_pct(results.get("total_return"))),
            ("Signals", str(len(engine_result.get("signals", [])))),
            ("Orders", str(len(engine_result.get("orders", [])))),
            ("Fills", str(len(engine_result.get("fills", [])))),
        ]
        cls._print_table("BACKTEST SUMMARY", summary_rows)

        # 2. Performance Summary
        perf_rows = [
            ("CAGR", cls._fmt_pct(m.get("annual_return"))), # Approximation if cagr not explicitly calc
            ("Annual Return", cls._fmt_pct(m.get("annual_return"))),
            ("Volatility", cls._fmt_pct(m.get("annual_volatility"))),
            ("Sharpe Ratio", cls._fmt_num(m.get("sharpe_ratio"))),
            ("Sortino Ratio", cls._fmt_num(m.get("sortino_ratio"))),
            ("Calmar Ratio", cls._fmt_num(m.get("calmar_ratio"))),
            ("Max Drawdown", cls._fmt_pct(m.get("max_drawdown"))),
            ("Best Day", cls._fmt_pct(m.get("best_day"))),
            ("Worst Day", cls._fmt_pct(m.get("worst_day"))),
            ("Daily Win Rate", cls._fmt_pct(m.get("daily_win_rate"))),
            ("Benchmark", summary.get("benchmark_data", {}).get("symbol", "N/A")),
            ("Alpha", cls._fmt_num(m.get("alpha"))),
            ("Beta", cls._fmt_num(m.get("beta"))),
        ]
        cls._print_table("PERFORMANCE SUMMARY", perf_rows)

        # 3. Trade Analysis
        fills = engine_result.get("fills", [])
        if not fills:
            return

        closed_trades, open_lots = cls.match_trades(fills)
        
        if closed_trades:
            # Group by Symbol
            report_rows = cls._build_trade_rows(
                [{**t, "group": t.get("symbol", "UNKNOWN")} for t in closed_trades],
                initial_capital
            )
            headers = ["Pair", "Trades", "AvgRet%", "TotPnl$", "TotPnl%", "AvgDur", "Win", "Draw", "Loss", "Win%"]
            cls._print_matrix("BACKTESTING REPORT (BY SYMBOL)", headers, report_rows)

            # Group by Entry Tag
            entry_rows = cls._build_trade_rows(
                [{**t, "group": t.get("entry_tag", "OTHER")} for t in closed_trades],
                initial_capital
            )
            cls._print_matrix("ENTER TAG STATS", ["EnterTag", "Trades", "AvgRet%", "TotPnl$", "TotPnl%", "AvgDur", "Win", "Draw", "Loss", "Win%"], entry_rows)

            # Group by Exit Reason
            exit_rows = cls._build_trade_rows(
                [{**t, "group": t.get("exit_reason", "OTHER")} for t in closed_trades],
                initial_capital
            )
            cls._print_matrix("EXIT REASON STATS", ["ExitReason", "Trades", "AvgRet%", "TotPnl$", "TotPnl%", "AvgDur", "Win", "Draw", "Loss", "Win%"], exit_rows)

