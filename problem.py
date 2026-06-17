"""
problem.py — дані та базові операції для задачі TOPTW.
"""
from __future__ import annotations
import json
from typing import Iterable


class ProblemInstance:
    """Зберігає всі параметри конкретної задачі TOPTW."""

    def __init__(self, n: int, k: int, alpha: float,
                 a: list, t: list, o: list, c: list,
                 T: list, s: list, m: list,
                 d: list):
        """
        n       — кількість туристичних об'єктів без депо;
        k       — кількість маршрутів;
        alpha   — коефіцієнт зменшення привабливості повторних включень;
        a[i-1]  — привабливість об'єкта i, i = 1..n;
        t[i-1]  — час огляду об'єкта i, i = 1..n;
        o[i], c[i] — початок та кінець часового вікна пункту i, i = 0..n;
        T[r]    — бюджет часу маршруту r;
        s[r]    — абсолютний час старту маршруту r;
        m[r]    — максимальна кількість туристичних об'єктів у маршруті r;
        d[i][j] — час переміщення між пунктами i та j, i,j = 0..n.
        """
        self.n = int(n)
        self.k = int(k)
        self.alpha = float(alpha)
        self.a = [float(x) for x in a]
        self.t = [float(x) for x in t]
        self.o = [float(x) for x in o]
        self.c = [float(x) for x in c]
        self.T = [float(x) for x in T]
        self.s = [float(x) for x in s]
        self.m = [int(x) for x in m]
        self.d = [[float(x) for x in row] for row in d]
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        if self.n < 0 or self.k < 0:
            raise ValueError("n та k мають бути невід'ємними")
        if not (0 < self.alpha <= 1):
            raise ValueError("alpha має належати інтервалу (0; 1]")
        if len(self.a) != self.n or len(self.t) != self.n:
            raise ValueError("Списки a та t повинні мати довжину n")
        if len(self.o) != self.n + 1 or len(self.c) != self.n + 1:
            raise ValueError("Списки o та c повинні мати довжину n+1")
        if len(self.T) != self.k or len(self.s) != self.k or len(self.m) != self.k:
            raise ValueError("Списки T, s та m повинні мати довжину k")
        if len(self.d) != self.n + 1 or any(len(row) != self.n + 1 for row in self.d):
            raise ValueError("Матриця d повинна мати розмір (n+1)x(n+1)")

    # ------------------------------------------------------------------ #
    #  Перевірка допустимості маршруту                                    #
    # ------------------------------------------------------------------ #

    def route_feasible(self, route: list[int], r: int) -> bool:
        """
        route — список індексів туристичних об'єктів (1..n, без депо 0).
        r     — номер маршруту (0-based).
        Повертає True, якщо маршрут допустимий за всіма обмеженнями.
        """
        if r < 0 or r >= self.k:
            return False
        if len(route) > self.m[r]:
            return False
        if len(route) != len(set(route)):
            return False
        if any((obj < 1 or obj > self.n) for obj in route):
            return False

        cur_time = self.s[r]
        prev = 0
        for obj in route:
            arr = cur_time + self.d[prev][obj]
            start = max(arr, self.o[obj])
            end = start + self.t[obj - 1]
            # Прибути раніше відкриття можна, але огляд має завершитися
            # не пізніше часу закриття.
            if end > self.c[obj] + 1e-9:
                return False
            cur_time = end
            prev = obj

        # T[r] — це тривалість маршруту, тому порівнюємо з s[r] + T[r].
        return cur_time + self.d[prev][0] <= self.s[r] + self.T[r] + 1e-9

    def route_end_time(self, route: list[int], r: int) -> float:
        """Абсолютний час повернення в депо для заданого маршруту."""
        cur = self.s[r]
        prev = 0
        for obj in route:
            arr = cur + self.d[prev][obj]
            cur = max(arr, self.o[obj]) + self.t[obj - 1]
            prev = obj
        return cur + self.d[prev][0]

    def route_duration(self, route: list[int], r: int) -> float:
        """Тривалість маршруту від старту до повернення в депо."""
        return self.route_end_time(route, r) - self.s[r]

    def route_waypoint_times(self, route: list[int], r: int) -> list[tuple[int, float]]:
        """
        Повертає список (пункт, час_прибуття/початку_огляду) для депо + об'єктів + депо.
        Для депо на старті — час відправлення s[r].
        Для кожного об'єкта — фактичний початок огляду (з урахуванням очікування).
        Для депо в кінці — час повернення.
        """
        waypoints = [(0, self.s[r])]
        cur = self.s[r]
        prev = 0
        for obj in route:
            arr = cur + self.d[prev][obj]
            start = max(arr, self.o[obj])
            waypoints.append((obj, start))
            cur = start + self.t[obj - 1]
            prev = obj
        waypoints.append((0, cur + self.d[prev][0]))
        return waypoints

    # ------------------------------------------------------------------ #
    #  Цільова функція                                                     #
    # ------------------------------------------------------------------ #

    def objective(self, portfolio: list[list[int]]) -> float:
        """
        Обчислює сумарну привабливість портфеля маршрутів.

        Якщо об'єкт i включено до q_i різних маршрутів, його внесок дорівнює
        a_i + a_i*alpha + ... + a_i*alpha^(q_i-1).
        Повтори всередині одного маршруту не враховуються як коректні, оскільки
        такі маршрути є недопустимими.
        """
        counts = [0] * (self.n + 1)
        for route in portfolio:
            for obj in set(route):
                if 1 <= obj <= self.n:
                    counts[obj] += 1

        total = 0.0
        for obj in range(1, self.n + 1):
            for q in range(counts[obj]):
                total += self.a[obj - 1] * (self.alpha ** q)
        return round(total, 6)

    def portfolio_feasible(self, portfolio: list[list[int]]) -> bool:
        """Перевіряє допустимість усіх маршрутів портфеля."""
        return len(portfolio) == self.k and all(
            self.route_feasible(route, r) for r, route in enumerate(portfolio)
        )

    # ------------------------------------------------------------------ #
    #  Серіалізація                                                        #
    # ------------------------------------------------------------------ #

    def to_json(self, path: str) -> None:
        data = {
            "n": self.n, "k": self.k, "alpha": self.alpha,
            "a": self.a, "t": self.t, "o": self.o, "c": self.c,
            "T": self.T, "s": self.s, "m": self.m, "d": self.d,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "ProblemInstance":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def display(self) -> None:
        print(f"n={self.n}, k={self.k}, alpha={self.alpha}")
        print(f"Привабливості a: {self.a}")
        print(f"Часи огляду t: {self.t}")
        print(f"Часові вікна o: {self.o}")
        print(f"             c: {self.c}")
        for r in range(self.k):
            print(f"Маршрут {r + 1}: T={self.T[r]}, s={self.s[r]}, m={self.m[r]}")
        print("Матриця часу переміщення d:")
        for row in self.d:
            print(" ", row)
