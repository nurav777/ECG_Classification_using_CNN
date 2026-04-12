import math
import random
import time


class ECGStreamGenerator:
    def __init__(self) -> None:
        self._t = 0.0
        self._dt = 0.04

    def next_value(self) -> float:
        p_wave = 0.1 * math.sin(2 * math.pi * 1.2 * self._t)
        qrs_like = 0.7 * math.sin(2 * math.pi * 5.0 * self._t) ** 5
        t_wave = 0.2 * math.sin(2 * math.pi * 0.5 * self._t + 0.5)
        noise = random.uniform(-0.03, 0.03)
        value = p_wave + qrs_like + t_wave + noise
        self._t += self._dt
        return round(value, 4)

    @staticmethod
    def now_ms() -> int:
        return int(time.time() * 1000)
