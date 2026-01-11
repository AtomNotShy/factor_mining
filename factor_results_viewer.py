#!/usr/bin/env python3
"""
因子测试结果查看器
提供便捷的界面查看和分析因子测试结果
"""

import sys
import os
import pandas as pd
import json
from pathlib import Path
import warnings
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class FactorResultsViewer:
    """因子测试结果查看器"""
    
    def __init__(self, results_dir: str = "factor_test_results"):
        self.results_dir = Path(results_dir)
        self.summary_df = pd.DataFrame()
        self.detailed_results = []
        
        print(f"🔍 因子结果查看器初始化")
        print(f"📁 结果目录: {self.results_dir.absolute()}")
        
        # 自动加载最新结果
        self.load_latest_results()
    
    def load_latest_results(self) -> bool:
        """加载最新的测试结果"""
        try:
            # 加载汇总结果
            summary_file = self.results_dir / "latest_factor_test_summary.csv"
            if summary_file.exists():
                self.summary_df = pd.read_csv(summary_file)
                print(f"✅ 已加载汇总结果: {len(self.summary_df)} 个因子")
            else:
                print("⚠️ 未找到最新的汇总结果文件")
                return False
            
            # 加载详细结果
            detailed_file = self.results_dir / "latest_factor_test_detailed.json"
            if detailed_file.exists():
                with open(detailed_file, 'r', encoding='utf-8') as f:
                    self.detailed_results = json.load(f)
                print(f"✅ 已加载详细结果: {len(self.detailed_results)} 个因子")
            else:
                print("⚠️ 未找到最新的详细结果文件")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载结果失败: {e}")
            return False
    
    def show_summary_stats(self):
        """显示汇总统计"""
        if self.summary_df.empty:
            print("❌ 无可用数据")
            return
        
        print("\n📊 因子测试汇总统计")
        print("=" * 50)
        
        total_factors = len(self.summary_df)
        print(f"总因子数量: {total_factors}")
        
        # 按评级分布
        rating_counts = self.summary_df['overall_rating'].value_counts()
        print(f"\n📈 评级分布:")
        for rating, count in rating_counts.items():
            percentage = count / total_factors * 100
            print(f"   {rating}: {count} 个 ({percentage:.1f}%)")
        
        # 按类别分布
        category_counts = self.summary_df['category'].value_counts()
        print(f"\n🏷️ 类别分布:")
        for category, count in category_counts.items():
            percentage = count / total_factors * 100
            print(f"   {category}: {count} 个 ({percentage:.1f}%)")
        
        # 评分统计
        print(f"\n🏆 评分统计:")
        print(f"   最高评分: {self.summary_df['final_score'].max():.3f}")
        print(f"   最低评分: {self.summary_df['final_score'].min():.3f}")
        print(f"   平均评分: {self.summary_df['final_score'].mean():.3f}")
        print(f"   中位数评分: {self.summary_df['final_score'].median():.3f}")
    
    def show_top_factors(self, n: int = 10, min_score: float = 0.4):
        """显示最佳因子"""
        if self.summary_df.empty:
            print("❌ 无可用数据")
            return
        
        # 过滤和排序
        filtered_df = self.summary_df[
            self.summary_df['final_score'] >= min_score
        ].head(n)
        
        print(f"\n🏆 表现最好的 {len(filtered_df)} 个因子 (评分 ≥ {min_score})")
        print("=" * 100)
        print(f"{'排名':<4} {'因子名称':<25} {'评分':<6} {'评级':<6} {'1日IC':<8} {'5日IC':<8} {'多空收益':<10} {'类别':<12}")
        print("-" * 100)
        
        for i, (idx, row) in enumerate(filtered_df.iterrows(), 1):
            print(f"{i:<4} {row['factor_name']:<25} {row['final_score']:<6.3f} "
                  f"{row['overall_rating']:<6} {row['ic_1d']:<8.4f} {row['ic_5d']:<8.4f} "
                  f"{row['long_short_return']:<10.4f} {row['sub_category']:<12}")
    
    def show_factors_by_category(self, category: str = None):
        """按类别显示因子"""
        if self.summary_df.empty:
            print("❌ 无可用数据")
            return
        
        if category:
            filtered_df = self.summary_df[
                self.summary_df['category'].str.contains(category, case=False, na=False) |
                self.summary_df['sub_category'].str.contains(category, case=False, na=False)
            ]
            print(f"\n🏷️ {category} 类别因子 ({len(filtered_df)} 个)")
        else:
            filtered_df = self.summary_df
            print(f"\n📋 所有因子 ({len(filtered_df)} 个)")
        
        if filtered_df.empty:
            print(f"未找到 {category} 类别的因子")
            return
        
        # 按评分排序
        filtered_df = filtered_df.sort_values('final_score', ascending=False)
        
        print("=" * 100)
        print(f"{'因子名称':<25} {'评分':<6} {'评级':<6} {'1日IC':<8} {'IC_IR':<8} {'多空收益':<10} {'子类别':<12}")
        print("-" * 100)
        
        for idx, row in filtered_df.iterrows():
            print(f"{row['factor_name']:<25} {row['final_score']:<6.3f} "
                  f"{row['overall_rating']:<6} {row['ic_1d']:<8.4f} {row['ic_ir_1d']:<8.4f} "
                  f"{row['long_short_return']:<10.4f} {row['sub_category']:<12}")
    
    def show_factor_details(self, factor_name: str):
        """显示因子详细信息"""
        # 从详细结果中查找
        factor_details = None
        for result in self.detailed_results:
            if result.get('factor_name') == factor_name:
                factor_details = result
                break
        
        if not factor_details:
            print(f"❌ 未找到因子 {factor_name} 的详细信息")
            return
        
        if "error" in factor_details:
            print(f"❌ 因子 {factor_name} 测试失败: {factor_details['error']}")
            return
        
        print(f"\n📄 {factor_name} 详细报告")
        print("=" * 80)
        
        # 基本信息
        print(f"📋 基本信息:")
        print(f"   名称: {factor_details['factor_name']}")
        print(f"   描述: {factor_details['factor_description']}")
        print(f"   类别: {factor_details['factor_category']} - {factor_details['factor_sub_category']}")
        print(f"   计算窗口: {factor_details['calculation_window']}")
        
        # 数据统计
        basic_stats = factor_details['basic_stats']
        print(f"\n📊 数据统计:")
        print(f"   总数据点: {basic_stats['total_count']}")
        print(f"   有效数据点: {basic_stats['valid_count']}")
        print(f"   有效率: {basic_stats['valid_rate']:.2%}")
        print(f"   均值: {basic_stats['mean']:.6f}")
        print(f"   标准差: {basic_stats['std']:.6f}")
        
        # 评分结果
        score_result = factor_details['score_result']
        print(f"\n🏆 评分结果:")
        print(f"   综合评分: {score_result['final_score']:.3f}")
        print(f"   总体评级: {score_result['overall_rating']}")
        print(f"   使用建议: {score_result['recommendation']}")
        
        # 详细评分
        print(f"\n📈 详细评分:")
        for key, detail in score_result['details'].items():
            if isinstance(detail, dict):
                print(f"   {key}: {detail['rating']} (值: {detail['value']:.4f})")
        
        # IC分析
        ic_results = factor_details['ic_results']
        print(f"\n📊 IC分析:")
        for period, ic_data in ic_results.items():
            period_num = period.split('_')[1]
            ic = ic_data['ic']
            ic_ir = ic_data['ic_ir']
            win_rate = ic_data['ic_win_rate']
            print(f"   {period_num}期: IC={ic:.4f}, IC_IR={ic_ir:.4f}, 胜率={win_rate:.2%}")
        
        # 分层回测
        backtest = factor_details['backtest_result']
        if 'quantile_stats' in backtest:
            print(f"\n🔍 分层回测:")
            print(f"   多空收益: {backtest['long_short_return']:.4f}")
            print(f"   样本数量: {backtest['total_samples']}")
            
            print(f"\n   各分位数表现:")
            for q, stats in backtest['quantile_stats'].items():
                print(f"     {q}: 平均收益={stats['avg_return']:.4f}, "
                      f"夏普={stats['sharpe']:.3f}, 胜率={stats['win_rate']:.2%}")
    
    def search_factors(self, keyword: str):
        """搜索因子"""
        if self.summary_df.empty:
            print("❌ 无可用数据")
            return
        
        # 在因子名称和描述中搜索
        mask = (
            self.summary_df['factor_name'].str.contains(keyword, case=False, na=False) |
            self.summary_df['description'].str.contains(keyword, case=False, na=False)
        )
        
        results = self.summary_df[mask].sort_values('final_score', ascending=False)
        
        print(f"\n🔍 搜索结果: '{keyword}' ({len(results)} 个)")
        print("=" * 100)
        
        if results.empty:
            print("未找到匹配的因子")
            return
        
        print(f"{'因子名称':<25} {'评分':<6} {'评级':<6} {'1日IC':<8} {'描述':<30}")
        print("-" * 100)
        
        for idx, row in results.iterrows():
            desc = row['description'][:27] + "..." if len(row['description']) > 30 else row['description']
            print(f"{row['factor_name']:<25} {row['final_score']:<6.3f} "
                  f"{row['overall_rating']:<6} {row['ic_1d']:<8.4f} {desc:<30}")
    
    def export_results(self, output_file: str = None, top_n: int = None):
        """导出结果到文件"""
        if self.summary_df.empty:
            print("❌ 无可用数据")
            return
        
        if output_file is None:
            output_file = f"factor_analysis_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # 选择要导出的数据
        export_df = self.summary_df.copy()
        if top_n:
            export_df = export_df.head(top_n)
        
        # 添加一些计算字段
        export_df['abs_ic_1d'] = export_df['ic_1d'].abs()
        export_df['abs_ic_5d'] = export_df['ic_5d'].abs()
        
        # 保存文件
        export_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ 结果已导出到: {output_file}")
        print(f"   导出因子数量: {len(export_df)}")
    
    def generate_charts(self, output_dir: str = None):
        """生成图表"""
        if self.summary_df.empty:
            print("❌ 无可用数据")
            return
        
        if output_dir is None:
            output_dir = self.results_dir / "charts"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 生成图表...")
        
        # 1. 评分分布直方图
        self._plot_score_distribution(output_dir)
        
        # 2. IC分布图
        self._plot_ic_distribution(output_dir)
        
        # 3. 评分vs IC散点图
        self._plot_score_vs_ic(output_dir)
        
        # 4. 类别对比图
        self._plot_category_comparison(output_dir)
        
        print(f"✅ 图表已保存到: {output_dir}")
    
    def _plot_score_distribution(self, output_dir: Path):
        """绘制评分分布图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(self.summary_df['final_score'], bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Score')
            ax.set_ylabel('Number of Factors')
            ax.set_title('Factor Score Distribution')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'score_distribution.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"⚠️ 生成评分分布图失败: {e}")
    
    def _plot_ic_distribution(self, output_dir: Path):
        """绘制IC分布图"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # 1日IC分布
            ax1.hist(self.summary_df['ic_1d'], bins=30, edgecolor='black', alpha=0.7, color='blue')
            ax1.set_xlabel('IC (1-day)')
            ax1.set_ylabel('Number of Factors')
            ax1.set_title('1-day IC Distribution')
            ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax1.grid(True, alpha=0.3)
            
            # 5日IC分布
            ax2.hist(self.summary_df['ic_5d'], bins=30, edgecolor='black', alpha=0.7, color='green')
            ax2.set_xlabel('IC (5-day)')
            ax2.set_ylabel('Number of Factors')
            ax2.set_title('5-day IC Distribution')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'ic_distribution.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"⚠️ 生成IC分布图失败: {e}")
    
    def _plot_score_vs_ic(self, output_dir: Path):
        """绘制评分vs IC散点图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(self.summary_df['ic_1d'], self.summary_df['final_score'], 
                      alpha=0.6, s=50, c=self.summary_df['final_score'], cmap='viridis')
            ax.set_xlabel('IC (1-day)')
            ax.set_ylabel('Score')
            ax.set_title('Score vs IC (1-day)')
            ax.grid(True, alpha=0.3)
            plt.colorbar(ax.collections[0], ax=ax, label='Score')
            plt.tight_layout()
            plt.savefig(output_dir / 'score_vs_ic.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"⚠️ 生成评分vs IC图失败: {e}")
    
    def _plot_category_comparison(self, output_dir: Path):
        """绘制类别对比图"""
        try:
            if 'category' not in self.summary_df.columns:
                return
            
            category_stats = self.summary_df.groupby('category').agg({
                'final_score': 'mean',
                'ic_1d': 'mean',
                'long_short_return': 'mean'
            }).reset_index()
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 平均评分
            axes[0].bar(category_stats['category'], category_stats['final_score'])
            axes[0].set_ylabel('Average Score')
            axes[0].set_title('Average Score by Category')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # 平均IC
            axes[1].bar(category_stats['category'], category_stats['ic_1d'])
            axes[1].set_ylabel('Average IC (1-day)')
            axes[1].set_title('Average IC by Category')
            axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # 平均多空收益
            axes[2].bar(category_stats['category'], category_stats['long_short_return'])
            axes[2].set_ylabel('Average Long-Short Return')
            axes[2].set_title('Average Long-Short Return by Category')
            axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[2].tick_params(axis='x', rotation=45)
            axes[2].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'category_comparison.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"⚠️ 生成类别对比图失败: {e}")
    
    def _translate_rating(self, rating: str) -> str:
        """翻译评级为英文"""
        rating_map = {
            '优秀': 'Excellent',
            '良好': 'Good',
            '一般': 'Fair',
            '较差': 'Poor'
        }
        return rating_map.get(rating, rating)
    
    def generate_html_report(self, output_file: str = None, top_n: int = 20):
        """生成HTML报告"""
        if self.summary_df.empty:
            print("❌ 无可用数据")
            return
        
        if output_file is None:
            output_file = self.results_dir / f"factor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        else:
            output_file = Path(output_file)
        
        # 生成图表
        chart_dir = output_file.parent / "charts"
        self.generate_charts(chart_dir)
        
        # 准备数据
        top_factors = self.summary_df.head(top_n)
        
        # 计算相对路径（HTML文件在factor_test_results目录下，charts也在同一目录下）
        chart_relative_path = "charts"
        
        # 生成HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Factor Test Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        h3 {{
            color: #666;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 18px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .stat-box {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .chart-section {{
            margin-bottom: 40px;
        }}
        .chart-container {{
            background: #fafafa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            margin-bottom: 20px;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            color: #999;
            font-size: 14px;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Factor Test Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 Summary Statistics</h2>
        <div>
            <div class="stat-box">
                <div class="stat-label">Total Factors</div>
                <div class="stat-value">{len(self.summary_df)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Average Score</div>
                <div class="stat-value">{self.summary_df['final_score'].mean():.3f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Max Score</div>
                <div class="stat-value">{self.summary_df['final_score'].max():.3f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Avg IC (1-day)</div>
                <div class="stat-value">{self.summary_df['ic_1d'].mean():.4f}</div>
            </div>
        </div>
        
        <h2>📈 Chart Analysis</h2>
        <div class="chart-section">
            <div class="chart-container">
                <h3>Score Distribution</h3>
                <img src="{chart_relative_path}/score_distribution.png" alt="Score Distribution" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                <div style="display:none; padding:20px; text-align:center; color:#999;">Chart failed to load. Please check file path.</div>
            </div>
            
            <div class="chart-container">
                <h3>IC Distribution</h3>
                <img src="{chart_relative_path}/ic_distribution.png" alt="IC Distribution" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                <div style="display:none; padding:20px; text-align:center; color:#999;">Chart failed to load. Please check file path.</div>
            </div>
            
            <div class="chart-container">
                <h3>Score vs IC</h3>
                <img src="{chart_relative_path}/score_vs_ic.png" alt="Score vs IC" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                <div style="display:none; padding:20px; text-align:center; color:#999;">Chart failed to load. Please check file path.</div>
            </div>
            
            <div class="chart-container">
                <h3>Category Comparison</h3>
                <img src="{chart_relative_path}/category_comparison.png" alt="Category Comparison" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                <div style="display:none; padding:20px; text-align:center; color:#999;">Chart failed to load. Please check file path.</div>
            </div>
        </div>
        
        <h2>🏆 Top {top_n} Factors</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Factor Name</th>
                    <th>Score</th>
                    <th>Rating</th>
                    <th>1-day IC</th>
                    <th>5-day IC</th>
                    <th>Long-Short Return</th>
                    <th>Category</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for i, (idx, row) in enumerate(top_factors.iterrows(), 1):
            rating_en = self._translate_rating(row['overall_rating'])
            html_content += f"""
                <tr>
                    <td>{i}</td>
                    <td>{row['factor_name']}</td>
                    <td>{row['final_score']:.3f}</td>
                    <td>{rating_en}</td>
                    <td>{row['ic_1d']:.4f}</td>
                    <td>{row['ic_5d']:.4f}</td>
                    <td>{row['long_short_return']:.4f}</td>
                    <td>{row.get('category', 'N/A')}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        
        # 保存HTML文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML报告已生成: {output_file}")
    
    def interactive_menu(self):
        """交互式菜单"""
        while True:
            print("\n" + "="*50)
            print("🔍 因子测试结果查看器")
            print("="*50)
            print("1. 显示汇总统计")
            print("2. 显示最佳因子")
            print("3. 按类别查看因子")
            print("4. 查看因子详情")
            print("5. 搜索因子")
            print("6. 导出结果")
            print("7. 重新加载结果")
            print("8. 生成图表")
            print("9. 生成HTML报告")
            print("0. 退出")
            
            choice = input("\n请选择操作 (0-9): ").strip()
            
            if choice == "0":
                print("👋 再见!")
                break
            elif choice == "1":
                self.show_summary_stats()
            elif choice == "2":
                n = input("显示前几名 (默认10): ").strip()
                n = int(n) if n else 10
                min_score = input("最低评分 (默认0.4): ").strip()
                min_score = float(min_score) if min_score else 0.4
                self.show_top_factors(n=n, min_score=min_score)
            elif choice == "3":
                category = input("输入类别名称 (如momentum, volatility, 空白显示所有): ").strip()
                category = category if category else None
                self.show_factors_by_category(category)
            elif choice == "4":
                factor_name = input("输入因子名称: ").strip()
                if factor_name:
                    self.show_factor_details(factor_name)
            elif choice == "5":
                keyword = input("输入搜索关键词: ").strip()
                if keyword:
                    self.search_factors(keyword)
            elif choice == "6":
                output_file = input("输出文件名 (空白自动生成): ").strip()
                output_file = output_file if output_file else None
                top_n = input("导出前几名 (空白导出全部): ").strip()
                top_n = int(top_n) if top_n else None
                self.export_results(output_file, top_n)
            elif choice == "7":
                self.load_latest_results()
            elif choice == "8":
                self.generate_charts()
            elif choice == "9":
                top_n = input("显示前几名 (默认20): ").strip()
                top_n = int(top_n) if top_n else 20
                self.generate_html_report(top_n=top_n)
            else:
                print("⚠️ 无效选项，请重新选择")


def main():
    """主函数"""
    print("🔍 因子测试结果查看器")
    print("用途: 查看和分析因子测试结果")
    
    viewer = FactorResultsViewer()
    
    # 如果有数据，启动交互式菜单
    if not viewer.summary_df.empty:
        viewer.interactive_menu()
    else:
        print("\n❌ 未找到测试结果数据")
        print("💡 请先运行 batch_factor_test.py 进行因子测试")


if __name__ == "__main__":
    main() 