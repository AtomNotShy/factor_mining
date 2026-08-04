"""
Freqtrade 风格参数系统
支持整数、小数、分类参数的动态调整，便于超参数优化
"""

from typing import Union, Any, Optional
from dataclasses import dataclass, field
import json


@dataclass
class Parameter:
    """参数基类"""
    name: str
    min_val: Union[int, float]
    max_val: Union[int, float]
    default: Union[int, float]
    space: str = "buy"  # buy/sell/protect
    optimize: bool = True

    def __post_init__(self) -> None:
        if not (self.min_val <= self.default <= self.max_val):
            raise ValueError(
                f"参数 {self.name} 默认值 {self.default} 不在范围 "
                f"[{self.min_val}, {self.max_val}] 内"
            )
        self._current_value: Union[int, float] = self.default

    @property
    def value(self) -> Union[int, float]:
        """获取当前值（支持优化时动态变更）"""
        return self._current_value

    @value.setter
    def value(self, val: Union[int, float]) -> None:
        """设置当前值（供优化器使用）"""
        if not (self.min_val <= val <= self.max_val):
            raise ValueError(
                f"参数 {self.name} 值 {val} 不在范围 "
                f"[{self.min_val}, {self.max_val}] 内"
            )
        self._current_value = val

    def to_dict(self) -> dict:
        """导出为字典"""
        return {
            "name": self.name,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "default": self.default,
            "current_value": self._current_value,
            "space": self.space,
            "optimize": self.optimize,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Parameter":
        """从字典创建"""
        param = cls(
            name=data["name"],
            min_val=data["min_val"],
            max_val=data["max_val"],
            default=data["default"],
            space=data.get("space", "buy"),
        )
        param._current_value = data.get("current_value", data["default"])
        return param


@dataclass
class IntParameter(Parameter):
    """整数参数"""

    min_val: int
    max_val: int
    default: int

    @Parameter.value.setter
    def value(self, val: Union[int, float]) -> None:
        if not isinstance(val, int):
            val = int(val)
        if not (self.min_val <= val <= self.max_val):
            raise ValueError(
                f"参数 {self.name} 值 {val} 不在范围 "
                f"[{self.min_val}, {self.max_val}] 内"
            )
        self._current_value = val


@dataclass
class DecimalParameter(Parameter):
    """小数参数"""

    min_val: float
    max_val: float
    default: float


class CategoricalParameter(Parameter):
    """分类参数"""

    options: list
    default: Any

    def __init__(
        self,
        name: str,
        options: list,
        default: Any,
        space: str = "buy",
        optimize: bool = True,
    ):
        super().__init__(
            name=name, min_val=0, max_val=len(options) - 1, default=default, space=space
        )
        self.options = options
        self.optimize = optimize

    @property
    def value(self) -> Any:
        if hasattr(self, "_current_index"):
            return self.options[int(self._current_index)]
        return self.default

    @value.setter
    def value(self, val: Any) -> None:
        if val not in self.options:
            raise ValueError(
                f"参数 {self.name} 值 {val} 不在选项 {self.options} 中"
            )
        self._current_index = self.options.index(val)

    @property
    def index(self) -> int:
        """获取当前索引"""
        if hasattr(self, "_current_index"):
            return int(self._current_index)
        return self.options.index(self.default) if self.default in self.options else 0

    @index.setter
    def index(self, idx: int) -> None:
        if 0 <= idx < len(self.options):
            self._current_index = idx

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["options"] = self.options
        base["current_index"] = getattr(self, "_current_index", self.index)
        return base


class BooleanParameter(CategoricalParameter):
    """布尔参数（CategoricalParameter 的简化版本）"""

    def __init__(
        self, name: str, default: bool = False, space: str = "buy", optimize: bool = True
    ):
        super().__init__(name=name, options=[False, True], default=default, space=space)
        self.optimize = optimize

    @property
    def value(self) -> bool:
        return bool(super().value)

    @value.setter
    def value(self, val: Any) -> None:
        # 直接操作父类的私有属性
        if val not in [False, True]:
            val = bool(val)
        if val not in self.options:
            raise ValueError(
                f"参数 {self.name} 值 {val} 不在选项 {self.options} 中"
            )
        self._current_index = self.options.index(val)


class ParameterSpace:
    """参数空间管理"""

    def __init__(self) -> None:
        self._params: dict[str, Parameter] = {}

    def add(self, param: Parameter) -> "ParameterSpace":
        """添加参数"""
        self._params[param.name] = param
        return self

    def get(self, name: str) -> Optional[Parameter]:
        """获取参数"""
        return self._params.get(name)

    def __getitem__(self, name: str) -> Parameter:
        """字典式访问"""
        return self._params[name]

    def __contains__(self, name: str) -> bool:
        """in 操作符支持"""
        return name in self._params

    def values(self) -> dict[str, Any]:
        """获取所有当前值"""
        return {name: param.value for name, param in self._params.items()}

    def defaults(self) -> dict[str, Any]:
        """获取所有默认值"""
        return {name: param.default for name, param in self._params.items()}

    def to_json(self) -> str:
        """导出为 JSON"""
        return json.dumps({name: p.to_dict() for name, p in self._params.items()})

    @classmethod
    def from_json(cls, data: str) -> "ParameterSpace":
        """从 JSON 创建"""
        space = cls()
        raw = json.loads(data)
        for name, param_data in raw.items():
            if "options" in param_data:
                space.add(CategoricalParameter.from_dict(param_data))
            else:
                space.add(Parameter.from_dict(param_data))
        return space


def create_int_parameter(
    name: str, min_val: int, max_val: int, default: int, space: str = "buy"
) -> IntParameter:
    """创建整数参数的工厂函数"""
    return IntParameter(name=name, min_val=min_val, max_val=max_val, default=default, space=space)


def create_decimal_parameter(
    name: str, min_val: float, max_val: float, default: float, space: str = "buy"
) -> DecimalParameter:
    """创建小数参数的工厂函数"""
    return DecimalParameter(name=name, min_val=min_val, max_val=max_val, default=default, space=space)


def create_boolean_parameter(
    name: str, default: bool = False, space: str = "buy"
) -> BooleanParameter:
    """创建布尔参数的工厂函数"""
    return BooleanParameter(name=name, default=default, space=space)


def time_to_decimal(hour: int, minute: int) -> float:
    """将时间转换为小数格式

    用于方便的时间比较和数值计算。

    Args:
        hour: 小时 (0-23)
        minute: 分钟 (0-59)

    Returns:
        小数时间，如 9:50 -> 9.83

    Examples:
        >>> time_to_decimal(9, 30)
        9.5
        >>> time_to_decimal(9, 50)
        9.83
        >>> time_to_decimal(14, 30)
        14.5
    """
    return hour + minute / 60


def decimal_to_time(decimal_time: float) -> tuple[int, int]:
    """将小数时间转换为 (hour, minute)

    Args:
        decimal_time: 小数时间，如 9.83

    Returns:
        (小时, 分钟) 元组

    Examples:
        >>> decimal_to_time(9.5)
        (9, 30)
        >>> decimal_to_time(9.83)
        (9, 49)  # 注意：9.83 是近似值，实际是 49.xx 分钟
    """
    hour = int(decimal_time)
    minute = int(round((decimal_time - hour) * 60))
    # 处理四舍五入导致的 60 分钟情况
    if minute == 60:
        hour += 1
        minute = 0
    return hour, minute


def create_time_parameter(
    name: str,
    min_hour: int,
    min_minute: int,
    max_hour: int,
    max_minute: int,
    default_hour: int,
    default_minute: int,
    space: str = "buy",
) -> DecimalParameter:
    """创建时间参数（小数格式）的工厂函数

    Args:
        name: 参数名
        min_hour, min_minute: 最小时间
        max_hour, max_minute: 最大时间
        default_hour, default_minute: 默认时间
        space: 参数空间 (buy/sell/protect)

    Returns:
        DecimalParameter，可直接用 .value 获取小数时间

    Examples:
        # 创建 9:30 - 16:00 范围的时间参数，默认 9:50
        time_param = create_time_parameter(
            name="intraday_time",
            min_hour=9, min_minute=30,
            max_hour=16, max_minute=0,
            default_hour=9, default_minute=50,
        )
        if current_time > time_param.value:  # 直接比较
            ...
    """
    return DecimalParameter(
        name=name,
        min_val=time_to_decimal(min_hour, min_minute),
        max_val=time_to_decimal(max_hour, max_minute),
        default=time_to_decimal(default_hour, default_minute),
        space=space,
    )


def parse_time(time_str: str) -> float:
    """解析字符串时间为小数格式

    Args:
        time_str: 时间字符串，支持 "9:50", "9:5", "14:30", "16:00" 等格式

    Returns:
        小数时间，如 9:50 -> 9.83

    Examples:
        >>> parse_time("9:50")
        9.83
        >>> parse_time("14:30")
        14.5
        >>> parse_time("16:00")
        16.0
    """
    parts = time_str.split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    return time_to_decimal(hour, minute)


def create_time_parameter_str(
    name: str,
    start_time: str,
    end_time: str,
    default_time: str,
    space: str = "buy",
) -> DecimalParameter:
    """创建时间参数（小数格式）的工厂函数 - 字符串版本

    Args:
        name: 参数名
        start_time: 最小时间，如 "9:30"
        end_time: 最大时间，如 "16:00"
        default_time: 默认时间，如 "9:50"
        space: 参数空间 (buy/sell/protect)

    Returns:
        DecimalParameter，可直接用 .value 获取小数时间

    Examples:
        # 创建 9:30 - 16:00 范围的时间参数，默认 9:50
        time_param = create_time_parameter_str(
            name="intraday_time",
            start_time="9:30",
            end_time="16:00",
            default_time="9:50",
        )
        if current_time > time_param.value:  # 直接比较，值为 9.83
            ...
    """
    return DecimalParameter(
        name=name,
        min_val=parse_time(start_time),
        max_val=parse_time(end_time),
        default=parse_time(default_time),
        space=space,
    )
