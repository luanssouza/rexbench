"""Range validation for metric outputs.

AUDIT.md section 5 found the predecessor pipeline's gini() silently returning -1.0 (outside its documented
[0,1] range) straight into a results CSV and a LaTeX table. @validated_range makes that
class of bug impossible to ship silently: an out-of-range, non-NaN return raises
MetricRangeError, which core/runner.py catches and records as a first-class `degenerate`
failure row instead of a results row.
"""
from __future__ import annotations

import functools
import math
from typing import Callable


class MetricRangeError(ValueError):
    def __init__(self, metric_name: str, value: float, lo: float, hi: float):
        self.metric_name = metric_name
        self.value = value
        self.lo = lo
        self.hi = hi
        super().__init__(
            f"{metric_name} returned {value!r}, outside declared valid range [{lo}, {hi}]"
        )


def validated_range(lo: float, hi: float, allow_nan: bool = False) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            value = fn(*args, **kwargs)
            if isinstance(value, float) and math.isnan(value):
                if not allow_nan:
                    raise MetricRangeError(fn.__name__, value, lo, hi)
                return value
            if not (lo <= value <= hi):
                raise MetricRangeError(fn.__name__, value, lo, hi)
            return value

        wrapper.valid_range = (lo, hi)
        wrapper.allow_nan = allow_nan
        return wrapper

    return decorator
