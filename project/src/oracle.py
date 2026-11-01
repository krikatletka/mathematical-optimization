

from collections.abc import Callable


class Oracle:
    def __init__(self, function: Callable[[float], float]):
        self.function = function
        self.evaluations = 0
        self.history: list[tuple[float, float]] = []

    def value(self, x: float) -> float:
        """Обчислює функцію та збільшує Nf."""

        result = float(self.function(x))

        self.evaluations += 1
        self.history.append((x, result))

        return result

    def peek(self, x: float) -> float:
        """Обчислює функцію без зміни Nf."""

        return float(self.function(x))

    @property
    def calls(self) -> int:
        return self.evaluations

    def reset(self) -> None:
        self.evaluations = 0
        self.history.clear()