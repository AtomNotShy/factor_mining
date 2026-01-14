"""
增强的回测可视化模块

新增图表：
1. 月度收益热力图
2. 收益分布直方图
3. 持仓时间 vs 收益散点图
4. 滚动的收益曲线
5. 信号延迟分析图
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from src.evaluation.backtesting.report import EnhancedBacktestReport, TradeDetail
from src.utils.logger import get_logger


logger = get_logger("visualizer")


class EnhancedBacktestVisualizer:
    """增强的回测可视化工具"""
    
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = output_dir
    
    def plot_monthly_returns_heatmap(
        self,
        monthly_returns: Dict[str, float],
        strategy_name: str = "Strategy",
    ) -> go.Figure:
        """
        绘制月度收益热力图
        
        Args:
            monthly_returns: 月度收益字典 {"2024-01": 0.02, ...}
            strategy_name: 策略名称
            
        Returns:
            Plotly Figure
        """
        if not monthly_returns:
            return go.Figure()
        
        # 转换为 DataFrame
        df = pd.DataFrame([
            {'month': k, 'return': v * 100}
            for k, v in monthly_returns.items()
        ])
        
        if df.empty:
            return go.Figure()
        
        # 提取年份和月份
        df['year'] = df['month'].str[:4]
        df['month_num'] = df['month'].str[5:7].astype(int)
        
        # 创建透视表
        pivot = df.pivot(index='year', columns='month_num', values='return')
        pivot = pivot.fillna(0)
        
        # 月份名称
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[month_names[i-1] for i in pivot.columns],
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=[[f'{v:.1f}%' for v in row] for row in pivot.values],
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{y}-%{x}: %{z:.2f}%<extra></extra>',
        ))
        
        fig.update_layout(
            title=f'{strategy_name} - Monthly Returns Heatmap',
            xaxis_title='Month',
            yaxis_title='Year',
            height=400,
        )
        
        return fig
    
    def plot_returns_distribution(
        self,
        returns: List[float],
        title: str = "Returns Distribution",
    ) -> go.Figure:
        """
        绘制收益分布直方图
        
        Args:
            returns: 收益率列表
            title: 图表标题
            
        Returns:
            Plotly Figure
        """
        if not returns:
            return go.Figure()
        
        returns_pct = [r * 100 for r in returns]
        
        fig = go.Figure()
        
        # 直方图
        fig.add_trace(go.Histogram(
            x=returns_pct,
            nbinsx=50,
            name='Returns',
            marker_color='royalblue',
            opacity=0.75,
        ))
        
        # 添加统计线
        mean_ret = np.mean(returns_pct)
        std_ret = np.std(returns_pct)
        
        fig.add_vline(x=mean_ret, line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {mean_ret:.2f}%")
        fig.add_vline(x=mean_ret - std_ret, line_dash="dot", line_color="orange")
        fig.add_vline(x=mean_ret + std_ret, line_dash="dot", line_color="orange")
        
        fig.update_layout(
            title=title,
            xaxis_title='Return (%)',
            yaxis_title='Frequency',
            height=400,
            bargap=0.1,
        )
        
        return fig
    
    def plot_duration_vs_pnl(
        self,
        trades: List[TradeDetail],
        strategy_name: str = "Strategy",
    ) -> go.Figure:
        """
        绘制持仓时间 vs 盈亏散点图
        
        Args:
            trades: 交易详情列表
            strategy_name: 策略名称
            
        Returns:
            Plotly Figure
        """
        if not trades:
            return go.Figure()
        
        # 准备数据
        durations = [t.duration_days for t in trades]
        pnls = [t.pnl_pct * 100 for t in trades]
        symbols = [t.symbol for t in trades]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        
        fig = go.Figure()
        
        # 散点图
        fig.add_trace(go.Scatter(
            x=durations,
            y=pnls,
            mode='markers',
            marker=dict(
                size=12,
                color=colors,
                opacity=0.7,
            ),
            text=[f"{s}<br>PnL: {p:.2f}%<br>Days: {d:.0f}"
                  for s, p, d in zip(symbols, pnls, durations)],
            hovertemplate='%{text}<extra></extra>',
        ))
        
        # 添加零线
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        # 添加趋势线
        if len(durations) > 1:
            z = np.polyfit(durations, pnls, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(durations), max(durations), 100)
            fig.add_trace(go.Scatter(
                x=x_line,
                y=p(x_line),
                mode='lines',
                name='Trend',
                line=dict(color='blue', dash='dot'),
            ))
        
        fig.update_layout(
            title=f'{strategy_name} - Duration vs PnL',
            xaxis_title='Duration (Days)',
            yaxis_title='PnL (%)',
            height=400,
        )
        
        return fig
    
    def plot_rolling_returns(
        self,
        equity_curve: List[Dict],
        initial_capital: float,
        strategy_name: str = "Strategy",
    ) -> go.Figure:
        """
        绘制滚动收益曲线
        
        Args:
            equity_curve: 净值曲线
            initial_capital: 初始资金
            strategy_name: 策略名称
            
        Returns:
            Plotly Figure
        """
        if not equity_curve:
            return go.Figure()
        
        df = pd.DataFrame(equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # 计算累积收益
        df['cumulative_return'] = (df['equity'] / initial_capital - 1) * 100
        
        # 计算滚动收益
        df['return_1m'] = df['equity'].pct_change(periods=21).fillna(0) * 100
        df['return_3m'] = df['equity'].pct_change(periods=63).fillna(0) * 100
        df['return_6m'] = df['equity'].pct_change(periods=126).fillna(0) * 100
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1,
                           subplot_titles=('Cumulative Return (%)', 'Rolling Returns (%)'))
        
        # 累积收益
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['cumulative_return'],
            name='Cumulative Return',
            line=dict(color='royalblue', width=2),
        ), row=1, col=1)
        
        # 滚动收益
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['return_1m'],
            name='1-Month',
            line=dict(color='green', width=1),
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['return_3m'],
            name='3-Month',
            line=dict(color='orange', width=1),
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['return_6m'],
            name='6-Month',
            line=dict(color='red', width=1),
        ), row=2, col=1)
        
        fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title=f'{strategy_name} - Rolling Returns Analysis',
            height=600,
            showlegend=True,
        )
        
        return fig
    
    def plot_equity_with_drawdown(
        self,
        equity_curve: List[Dict],
        initial_capital: float,
        strategy_name: str = "Strategy",
    ) -> go.Figure:
        """
        绘制净值曲线与回撤图
        
        Args:
            equity_curve: 净值曲线
            initial_capital: 初始资金
            strategy_name: 策略名称
            
        Returns:
            Plotly Figure
        """
        if not equity_curve:
            return go.Figure()
        
        df = pd.DataFrame(equity_curve)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # 计算回撤
        df['peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['peak']) / df['peak'] * 100
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=('Equity Curve', 'Drawdown'))
        
        # 净值曲线
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['equity'],
            name='Equity',
            fill='tozeroy',
            line=dict(color='royalblue', width=2),
        ), row=1, col=1)
        
        # 回撤曲线
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['drawdown'],
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=1),
        ), row=2, col=1)
        
        fig.update_layout(
            title=f'{strategy_name} - Equity & Drawdown',
            height=600,
            showlegend=True,
        )
        
        return fig
    
    def plot_trade_pnl_distribution(
        self,
        trades: List[TradeDetail],
        strategy_name: str = "Strategy",
    ) -> go.Figure:
        """
        绘制交易盈亏分布图
        
        Args:
            trades: 交易详情列表
            strategy_name: 策略名称
            
        Returns:
            Plotly Figure
        """
        if not trades:
            return go.Figure()
        
        pnls = [t.pnl_pct * 100 for t in trades]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        
        fig = go.Figure()
        
        # 柱状图
        fig.add_trace(go.Bar(
            x=list(range(len(pnls))),
            y=pnls,
            marker_color=colors,
            text=[f'{p:.2f}%' for p in pnls],
            textposition='outside',
        ))
        
        # 添加盈亏分界线
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title=f'{strategy_name} - Trade PnL Distribution',
            xaxis_title='Trade #',
            yaxis_title='PnL (%)',
            height=400,
        )
        
        return fig
    
    def plot_exit_reason_pie(
        self,
        trades: List[TradeDetail],
        strategy_name: str = "Strategy",
    ) -> go.Figure:
        """
        绘制离场原因饼图
        
        Args:
            trades: 交易详情列表
            strategy_name: 策略名称
            
        Returns:
            Plotly Figure
        """
        if not trades:
            return go.Figure()
        
        # 统计离场原因
        exit_reasons = {}
        for t in trades:
            reason = t.exit_reason or 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        if not exit_reasons:
            return go.Figure()
        
        fig = go.Figure(data=[go.Pie(
            labels=list(exit_reasons.keys()),
            values=list(exit_reasons.values()),
            hole=0.4,
            textinfo='label+percent',
        )])
        
        fig.update_layout(
            title=f'{strategy_name} - Exit Reason Distribution',
            height=400,
        )
        
        return fig
    
    def generate_full_report(
        self,
        report: EnhancedBacktestReport,
        equity_curve: List[Dict],
        output_filename: str = "enhanced_backtest_report.html",
    ) -> str:
        """
        生成完整的 HTML 报告
        
        Args:
            report: 增强回测报告
            equity_curve: 净值曲线
            output_filename: 输出文件名
            
        Returns:
            输出文件路径
        """
        import os
        from datetime import datetime
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建图表
        equity_fig = self.plot_equity_with_drawdown(equity_curve, report.initial_capital, report.strategy_name)
        rolling_fig = self.plot_rolling_returns(equity_curve, report.initial_capital, report.strategy_name)
        monthly_fig = self.plot_monthly_returns_heatmap(report.monthly_returns, report.strategy_name)
        duration_fig = self.plot_duration_vs_pnl(report.trades, report.strategy_name)
        trade_fig = self.plot_trade_pnl_distribution(report.trades, report.strategy_name)
        exit_fig = self.plot_exit_reason_pie(report.trades, report.strategy_name)
        
        # 生成 HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Backtest Report - {report.strategy_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-card.green {{ background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); }}
        .metric-card.red {{ background: linear-gradient(135deg, #f44336 0%, #da190b 100%); }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; opacity: 0.9; }}
        .chart-section {{ margin: 30px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Enhanced Backtest Report: {report.strategy_name}</h1>
        <p>Timeframe: {report.timeframe} | Timerange: {report.timerange}</p>
        
        <h2>💰 Performance Summary</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">${report.final_equity:,.2f}</div>
                <div class="metric-label">Final Equity</div>
            </div>
            <div class="metric-card {'green' if report.total_return_pct > 0 else 'red'}">
                <div class="metric-value">{report.total_return_pct*100:.2f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card {'green' if report.annualized_return > 0 else 'red'}">
                <div class="metric-value">{report.annualized_return*100:.2f}%</div>
                <div class="metric-label">Annualized Return</div>
            </div>
            <div class="metric-card red">
                <div class="metric-value">{report.max_drawdown_pct*100:.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report.win_rate*100:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
        </div>
        
        <h2>📈 Rolling Returns</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{report.rolling_returns_1m*100:.2f}%</div>
                <div class="metric-label">1-Month</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report.rolling_returns_3m*100:.2f}%</div>
                <div class="metric-label">3-Month</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report.rolling_returns_6m*100:.2f}%</div>
                <div class="metric-label">6-Month</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report.rolling_returns_12m*100:.2f}%</div>
                <div class="metric-label">12-Month</div>
            </div>
        </div>
        
        <h2>📉 Equity & Drawdown</h2>
        <div id="equity-chart" class="chart-section"></div>
        
        <h2>📊 Rolling Returns Analysis</h2>
        <div id="rolling-chart" class="chart-section"></div>
        
        <h2>🗓️ Monthly Returns Heatmap</h2>
        <div id="monthly-chart" class="chart-section"></div>
        
        <h2>⏱️ Duration vs PnL</h2>
        <div id="duration-chart" class="chart-section"></div>
        
        <h2>📊 Trade PnL Distribution</h2>
        <div id="trade-chart" class="chart-section"></div>
        
        <h2>🍕 Exit Reason Distribution</h2>
        <div id="exit-chart" class="chart-section"></div>
        
        <h2>📋 Trade Details</h2>
        <table>
            <tr>
                <th>#</th>
                <th>Symbol</th>
                <th>Entry Time</th>
                <th>Exit Time</th>
                <th>Duration</th>
                <th>PnL</th>
                <th>Entry Reason</th>
                <th>Exit Reason</th>
            </tr>
"""
        
        # 添加交易详情
        for i, t in enumerate(report.trades[:20], 1):  # 只显示前20笔
            pnl_class = 'positive' if t.pnl > 0 else 'negative'
            html_content += f"""
            <tr>
                <td>{i}</td>
                <td>{t.symbol}</td>
                <td>{t.entry_time.strftime('%Y-%m-%d')}</td>
                <td>{t.exit_time.strftime('%Y-%m-%d')}</td>
                <td>{t.duration_days:.0f} days</td>
                <td class="{pnl_class}">{t.pnl_pct*100:.2f}%</td>
                <td>{t.entry_reason}</td>
                <td>{t.exit_reason}</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <p style="color: #666; font-size: 12px; margin-top: 30px;">
            Generated at """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
        </p>
    </div>
    
    <script>
        Plotly.newPlot('equity-chart", """ + equity_fig.to_html(full_html=False, include_plotlyjs=False) + """);
        Plotly.newPlot('rolling-chart", """ + rolling_fig.to_html(full_html=False, include_plotlyjs=False) + """);
        Plotly.newPlot('monthly-chart", """ + monthly_fig.to_html(full_html=False, include_plotlyjs=False) + """);
        Plotly.newPlot('duration-chart", """ + duration_fig.to_html(full_html=False, include_plotlyjs=False) + """);
        Plotly.newPlot('trade-chart", """ + trade_fig.to_html(full_html=False, include_plotlyjs=False) + """);
        Plotly.newPlot('exit-chart", """ + exit_fig.to_html(full_html=False, include_plotlyjs=False) + """);
    </script>
</body>
</html>
"""
        
        output_path = os.path.join(self.output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Enhanced report saved to: {output_path}")
        
        return output_path


if __name__ == "__main__":
    # 测试可视化
    from src.evaluation.backtesting.report import EnhancedReportGenerator, EnhancedBacktestReport
    import pandas as pd
    from datetime import timedelta
    
    # 生成测试数据
    np.random.seed(42)
    start_date = pd.Timestamp('2024-01-01')
    dates = pd.date_range(start=start_date, periods=252, freq='B')
    
    equity = 100000
    equity_curve = []
    for ts in dates:
        daily_return = np.random.normal(0.0005, 0.015)
        equity = equity * (1 + daily_return)
        equity_curve.append({
            'timestamp': ts,
            'equity': equity,
        })
    
    trades = []
    for i in range(15):
        entry_time = start_date + timedelta(days=i * 15)
        exit_time = entry_time + timedelta(days=np.random.randint(5, 30))
        pnl = np.random.uniform(-1000, 2000)
        trades.append({
            'symbol': 'SPY',
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': 450 + np.random.uniform(-10, 10),
            'exit_price': 450 + np.random.uniform(-10, 10),
            'pnl': pnl,
            'pnl_pct': pnl / 100000,
            'quantity': 100,
            'entry_reason': 'momentum_crossover',
            'exit_reason': np.random.choice(['stoploss', 'roi', 'exit_signal', 'trailing']),
        })
    
    # 生成报告
    generator = EnhancedReportGenerator(
        strategy_name="ETF Momentum Strategy",
        timeframe="1d",
        timerange="2024-01-01 to 2024-12-31",
    )
    
    report = generator.generate_report(
        initial_capital=100000,
        final_equity=equity,
        equity_curve=equity_curve,
        trades=trades,
    )
    
    # 生成可视化报告
    visualizer = EnhancedBacktestVisualizer("./reports")
    output_path = visualizer.generate_full_report(report, equity_curve, "test_enhanced_report.html")
    
    print(f"\\nEnhanced visual report saved to: {output_path}")
