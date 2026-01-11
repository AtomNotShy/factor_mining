"""
因子计算API路由
提供因子计算、管理等功能
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import pandas as pd

from src.factors.base.factor import factor_registry
import src.factors  # noqa: F401 触发因子自动注册（technical/momentum/volatility/reversal 等）
from src.data.collectors.exchange import MultiExchangeCollector
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# 数据采集器
collector = MultiExchangeCollector()


@router.get("/list")
async def list_factors(category: Optional[str] = Query(None, description="因子分类")):
    """获取因子列表"""
    try:
        factors = factor_registry.list_factors(category)
        
        factor_info = []
        for factor_name in factors:
            factor = factor_registry.get_factor(factor_name)
            if factor:
                factor_info.append(factor.get_factor_info())
        
        return {
            "factors": factor_info,
            "count": len(factor_info),
            "category": category
        }
    
    except Exception as e:
        logger.error(f"获取因子列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_factor_categories():
    """获取因子分类"""
    try:
        all_factors = factor_registry.list_factors()
        categories = set()
        
        for factor_name in all_factors:
            factor = factor_registry.get_factor(factor_name)
            if factor:
                categories.add(factor.category)
        
        return {
            "categories": list(categories),
            "count": len(categories)
        }
    
    except Exception as e:
        logger.error(f"获取因子分类失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info/{factor_name}")
async def get_factor_info(factor_name: str):
    """获取特定因子的详细信息"""
    try:
        factor = factor_registry.get_factor(factor_name)
        
        if not factor:
            raise HTTPException(status_code=404, detail=f"未找到因子: {factor_name}")
        
        return factor.get_factor_info()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取因子信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate/{factor_name}")
async def calculate_factor(
    factor_name: str,
    symbol: str = Query(..., description="交易对符号"),
    timeframe: str = Query("1h", description="时间周期"),
    limit: int = Query(1000, description="数据条数")
):
    """计算指定因子的值"""
    try:
        # 获取因子
        factor = factor_registry.get_factor(factor_name)
        if not factor:
            raise HTTPException(status_code=404, detail=f"未找到因子: {factor_name}")
        
        # 获取市场数据
        df = await collector.get_ohlcv_from_best_source(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail="未找到市场数据")
        
        # 计算因子值
        factor_values = factor.calculate_with_validation(df)
        
        if factor_values.empty:
            raise HTTPException(status_code=500, detail="因子计算失败")
        
        # 转换为响应格式
        result_data = []
        for timestamp, value in factor_values.items():
            if pd.notna(value):  # 过滤NaN值
                result_data.append({
                    "timestamp": timestamp.isoformat(),
                    "value": float(value)
                })
        
        return {
            "factor_name": factor_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "data": result_data,
            "count": len(result_data),
            "statistics": {
                "mean": float(factor_values.mean()) if not factor_values.empty else None,
                "std": float(factor_values.std()) if not factor_values.empty else None,
                "min": float(factor_values.min()) if not factor_values.empty else None,
                "max": float(factor_values.max()) if not factor_values.empty else None
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"计算因子失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate/batch")
async def calculate_factors_batch(
    factor_names: List[str],
    symbol: str = Query(..., description="交易对符号"),
    timeframe: str = Query("1h", description="时间周期"),
    limit: int = Query(1000, description="数据条数")
):
    """批量计算多个因子的值"""
    try:
        # 获取市场数据（只获取一次）
        df = await collector.get_ohlcv_from_best_source(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail="未找到市场数据")
        
        results = {}
        
        for factor_name in factor_names:
            try:
                # 获取因子
                factor = factor_registry.get_factor(factor_name)
                if not factor:
                    results[factor_name] = {
                        "error": f"未找到因子: {factor_name}"
                    }
                    continue
                
                # 计算因子值
                factor_values = factor.calculate_with_validation(df)
                
                if factor_values.empty:
                    results[factor_name] = {
                        "error": "因子计算失败"
                    }
                    continue
                
                # 转换为响应格式
                result_data = []
                for timestamp, value in factor_values.items():
                    if pd.notna(value):
                        result_data.append({
                            "timestamp": timestamp.isoformat(),
                            "value": float(value)
                        })
                
                results[factor_name] = {
                    "data": result_data,
                    "count": len(result_data),
                    "statistics": {
                        "mean": float(factor_values.mean()) if not factor_values.empty else None,
                        "std": float(factor_values.std()) if not factor_values.empty else None,
                        "min": float(factor_values.min()) if not factor_values.empty else None,
                        "max": float(factor_values.max()) if not factor_values.empty else None
                    }
                }
                
            except Exception as e:
                logger.error(f"计算因子 {factor_name} 失败: {e}")
                results[factor_name] = {
                    "error": str(e)
                }
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "factors": results,
            "total_factors": len(factor_names),
            "successful": len([r for r in results.values() if "error" not in r])
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量计算因子失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 
