"""Radius validation shared by discovery, saved selections and CLI stages."""

import math

MIN_RADIUS_M = 10


def read_radius_m(value):
    if isinstance(value, bool):
        raise ValueError(f'반경은 {MIN_RADIUS_M}m 이상의 유한한 값으로 지정해 주세요.')
    number = float(value)
    if not math.isfinite(number) or number < MIN_RADIUS_M:
        raise ValueError(f'반경은 {MIN_RADIUS_M}m 이상의 유한한 값으로 지정해 주세요.')
    return number
