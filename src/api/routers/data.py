"""
数据管理API路由
提供数据采集、查询、管理等功能
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from src.data.collectors.exchange import MultiExchangeCollector
from src.data.collectors.polygon import PolygonCollector
from src.data.collectors.ib_history import IBHistoryCollector
from src.config.settings import get_settings
from src.utils.logger import get_logger
from ..schemas.data import (
    OHLCVRequest, OHLCVResponse, 
    SymbolsResponse, HealthResponse
)

router = APIRouter()
logger = get_logger(__name__)

# 全局数据采集器实例
collector = MultiExchangeCollector()
polygon_collector = PolygonCollector()

# 初始化 IB 采集器（使用配置信息）
settings = get_settings()
ib_collector = IBHistoryCollector(
    host=settings.ib.host,
    port=settings.ib.port,
    client_id=settings.ib.collector_client_id
)


@router.get("/exchanges/health", response_model=Dict[str, HealthResponse])
async def get_exchanges_health():
    """获取所有交易所健康状态"""
    try:
        health_results = await collector.health_check_all()
        return health_results
    except Exception as e:
        logger.error(f"获取交易所健康状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols", response_model=SymbolsResponse)
async def get_symbols(exchange: str = Query("binance", description="交易所名称")):
    """获取指定交易所的可用交易对"""
    try:
        if exchange not in collector.collectors:
            raise HTTPException(status_code=400, detail=f"不支持的交易所: {exchange}")
        
        symbols = await collector.collectors[exchange].get_symbols()
        
        return SymbolsResponse(
            exchange=exchange,
            symbols=symbols,
            count=len(symbols)
        )
    
    except Exception as e:
        logger.error(f"获取交易对失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ohlcv", response_model=OHLCVResponse)
async def get_ohlcv_data(request: OHLCVRequest):
    """获取OHLCV数据"""
    try:
        # 参数验证
        if not request.symbol or "/" not in request.symbol:
            raise HTTPException(status_code=400, detail="无效的交易对格式")
        
        if request.timeframe not in ["1m", "5m", "15m", "1h", "4h", "1d"]:
            raise HTTPException(status_code=400, detail="不支持的时间周期")
        
        # 获取数据
        df = await collector.get_ohlcv_from_best_source(
            symbol=request.symbol,
            timeframe=request.timeframe,
            since=request.since,
            limit=request.limit
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail="未找到数据")
        
        # 转换为响应格式
        data_records = []
        for timestamp, row in df.iterrows():
            data_records.append({
                "timestamp": timestamp.isoformat(),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": float(row['volume'])
            })
        
        return OHLCVResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            exchange=df['exchange'].iloc[0] if 'exchange' in df.columns else "unknown",
            data=data_records,
            count=len(data_records)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取OHLCV数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/polygon/ohlcv", response_model=OHLCVResponse)
async def get_polygon_ohlcv_data(request: OHLCVRequest):
    """获取美股/ETF OHLCV（Polygon，带本地缓存）"""
    try:
        if not request.symbol:
            raise HTTPException(status_code=400, detail="无效的标的代码")

        if request.timeframe not in ["1m", "5m", "15m", "30m", "1h", "1d"]:
            raise HTTPException(status_code=400, detail="不支持的时间周期")

        df = await polygon_collector.get_ohlcv(
            symbol=request.symbol,
            timeframe=request.timeframe,
            since=request.since,
            limit=request.limit,
        )

        if df.empty:
            raise HTTPException(status_code=404, detail="未找到数据（或未配置POLYGON_API_KEY）")

        data_records = []
        for timestamp, row in df.iterrows():
            data_records.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0.0)),
                }
            )

        return OHLCVResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            exchange="polygon",
            data=data_records,
            count=len(data_records),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Polygon 获取OHLCV失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ib/ohlcv", response_model=OHLCVResponse)
async def get_ib_ohlcv_data(request: OHLCVRequest):
    """获取美股/ETF OHLCV（IB，带本地缓存）"""
    try:
        if not request.symbol:
            raise HTTPException(status_code=400, detail="无效的标的代码")

        # IB 数据一般需要指定开始时间，如果未指定则取最近 500 条
        df = await ib_collector.get_ohlcv_async(
            symbol=request.symbol,
            timeframe=request.timeframe,
            since=request.since,
            use_cache=True
        )

        if df.empty:
            raise HTTPException(
                status_code=404, 
                detail="未找到数据（请确保 TWS/IB Gateway 已启动并连接）"
            )

        # 转换为响应格式
        data_records = []
        for timestamp, row in df.iterrows():
            data_records.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0.0)),
                }
            )

        return OHLCVResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            exchange="ib",
            data=data_records,
            count=len(data_records),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"IB 获取OHLCV失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ticker/{symbol}")
async def get_ticker(
    symbol: str,
    exchange: str = Query("binance", description="交易所名称")
):
    """获取实时行情数据"""
    try:
        if exchange not in collector.collectors:
            raise HTTPException(status_code=400, detail=f"不支持的交易所: {exchange}")
        
        ticker = await collector.collectors[exchange].get_ticker(symbol)
        
        if not ticker:
            raise HTTPException(status_code=404, detail="未找到行情数据")
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "ticker": ticker
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取行情数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{symbol}")
async def get_24h_stats(
    symbol: str,
    exchange: str = Query("binance", description="交易所名称")
):
    """获取24小时统计数据"""
    try:
        if exchange not in collector.collectors:
            raise HTTPException(status_code=400, detail=f"不支持的交易所: {exchange}")
        
        stats = await collector.collectors[exchange].get_24h_stats(symbol)
        
        if not stats:
            raise HTTPException(status_code=404, detail="未找到统计数据")
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "stats": stats
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取统计数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orderbook/{symbol}")
async def get_orderbook(
    symbol: str,
    exchange: str = Query("binance", description="交易所名称"),
    limit: int = Query(50, description="订单簿深度")
):
    """获取订单簿数据"""
    try:
        if exchange not in collector.collectors:
            raise HTTPException(status_code=400, detail=f"不支持的交易所: {exchange}")
        
        orderbook = await collector.collectors[exchange].get_orderbook(symbol, limit)
        
        if not orderbook:
            raise HTTPException(status_code=404, detail="未找到订单簿数据")
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "orderbook": orderbook
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取订单簿失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collect/start")
async def start_data_collection(
    symbols: List[str],
    timeframes: List[str],
    exchanges: List[str] = None
):
    """启动数据采集任务"""
    try:
        if exchanges is None:
            exchanges = list(collector.collectors.keys())
        
        # 验证参数
        for exchange in exchanges:
            if exchange not in collector.collectors:
                raise HTTPException(status_code=400, detail=f"不支持的交易所: {exchange}")
        
        # 这里应该启动后台任务进行数据采集
        # 暂时返回成功响应
        logger.info(f"启动数据采集: symbols={symbols}, timeframes={timeframes}, exchanges={exchanges}")
        
        return {
            "status": "started",
            "symbols": symbols,
            "timeframes": timeframes,
            "exchanges": exchanges,
            "message": "数据采集任务已启动"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动数据采集失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collect/stop")
async def stop_data_collection():
    """停止数据采集任务"""
    try:
        # 这里应该停止后台采集任务
        logger.info("停止数据采集任务")
        
        return {
            "status": "stopped",
            "message": "数据采集任务已停止"
        }
    
    except Exception as e:
        logger.error(f"停止数据采集失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/ib")
async def sync_ib_data(symbols: Optional[List[str]] = None):
    """
    触发 IB 数据增量同步
    
    Args:
        symbols: 要同步的标的列表，如果为 None 则从 tickers.txt 读取
    """
    try:
        from ib_data_maintenance import load_tickers, run_maintenance
        
        # 如果提供了 symbols 则临时覆盖
        if symbols:
            # 这里我们不直接运行 run_maintenance，因为那个函数会连接/断分。
            # 我们直接使用已经初始化的 ib_collector。
            results = {}
            for symbol in symbols:
                symbol = symbol.upper()
                df = await ib_collector.get_ohlcv_incremental_async(symbol, timeframe="1d")
                results[symbol] = len(df) if not df.empty else 0
            
            return {
                "status": "completed",
                "results": results,
                "message": f"同步完成: {len(symbols)} 个标的"
            }
        else:
            # 运行完整的维护逻辑（扫描 tickers.txt）
            # 注意：这可能会阻塞当前请求，在生产环境中建议使用 TaskManager 异步运行。
            # 这里我们复用系统现有的 task_manager。
            from src.core.task_manager import task_manager
            
            task_id = task_manager.submit(
                name="IB 数据自动维护同步",
                func=run_maintenance
            )
            
            return {
                "status": "started",
                "task_id": task_id,
                "message": "IB 数据维护任务已在后台启动"
            }
            
    except Exception as e:
        logger.error(f"同步 IB 数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/local-symbols")
async def get_local_symbols(timeframe: str = Query("1d", description="时间周期")):
    """
    获取本地存在的标的列表（扫描本地Parquet文件）
    
    Args:
        timeframe: 时间周期，如 "1d", "1h" 等
        
    Returns:
        本地存在的标的列表
    """
    try:
        settings = get_settings()
        data_dir = Path(settings.storage.data_dir)
        
        # 扫描不同可能的数据目录
        possible_dirs = [
            data_dir / "daily",  # 下载脚本保存的位置
            data_dir / "ib" / "ohlcv" / timeframe,  # IBCollector的缓存位置
            data_dir / "polygon" / "ohlcv" / "adjusted" / "utc" / timeframe,  # PolygonCollector的缓存位置
            data_dir / "polygon" / "ohlcv" / "raw" / "utc" / timeframe,
        ]
        
        symbols = set()
        
        for dir_path in possible_dirs:
            if not dir_path.exists():
                continue
            
            # 扫描parquet文件
            for parquet_file in dir_path.glob("*.parquet"):
                # 提取标的代码
                # 格式可能是: AAPL_1d.parquet 或 AAPL.parquet
                symbol = parquet_file.stem
                # 移除时间周期后缀（如果有）
                if f"_{timeframe}" in symbol:
                    symbol = symbol.replace(f"_{timeframe}", "")
                symbols.add(symbol.upper())
        
        # 排序并返回
        symbols_list = sorted(list(symbols))
        
        logger.info(f"找到 {len(symbols_list)} 个本地标的 (timeframe={timeframe})")
        
        return {
            "symbols": symbols_list,
            "count": len(symbols_list),
            "timeframe": timeframe,
            "data_dir": str(data_dir)
        }
    
    except Exception as e:
        logger.error(f"获取本地标的列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 
