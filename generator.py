"""
generator.py — генератор індивідуальних задач TOPTW.
"""
from __future__ import annotations
import random
import math
from problem import ProblemInstance


def generate_problem(n: int, k: int, alpha: float = 0.7,
                     a_mean: float = 10.0, a_half: float = 4.0,
                     d_mean: float = 15.0, d_half: float = 5.0,
                     t_mean: float = 20.0, t_half: float = 8.0,
                     window_width_factor: float = 2.0,
                     T_budget_factor: float = 1.5,
                     m_per_route: int | None = None,
                     seed: int | None = None) -> ProblemInstance:
    """
    Генерує випадкову задачу TOPTW.

    Вхід:
        n                  — кількість об'єктів
        k                  — кількість маршрутів
        alpha              — коефіцієнт зменшення привабливості
        a_mean, a_half     — середнє та напівінтервал привабливостей
        d_mean, d_half     — середнє та напівінтервал відстаней
        t_mean, t_half     — середнє та напівінтервал часів огляду
        window_width_factor — ширина вікна = factor * (t[i] + d_mean)
        T_budget_factor    — бюджет = factor * n * d_mean
        m_per_route        — макс. об'єктів у маршруті (за замовч. n)
        seed               — seed для відтворюваності

    Вихід:
        ProblemInstance
    """
    if seed is not None:
        random.seed(seed)

    def uniform(mean, half):
        return round(random.uniform(mean - half, mean + half), 1)

    # Привабливості об'єктів (1..n)
    a = [max(1.0, uniform(a_mean, a_half)) for _ in range(n)]

    # Часи огляду (1..n)
    t = [max(5.0, uniform(t_mean, t_half)) for _ in range(n)]

    # Матриця відстаней (0..n), 0 — депо
    size = n + 1
    d = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            dist = max(1.0, uniform(d_mean, d_half))
            d[i][j] = dist
            d[j][i] = dist

    # Часові вікна (o[0]=0, c[0]=big для депо)
    horizon = int(T_budget_factor * n * d_mean * k * 2)
    o = [0.0] * (n + 1)  # індекси 0..n, 0 — депо
    c = [float(horizon)] * (n + 1)

    for i in range(1, n + 1):
        # Орієнтовний час прибуття = d[0][i]
        arr = d[0][i]
        width = window_width_factor * (t[i - 1] + d_mean)
        oi = round(max(0.0, arr - width / 2), 1)
        ci = round(oi + width + t[i - 1], 1)
        o[i] = oi
        c[i] = ci

    # Параметри маршрутів
    T_budget = round(T_budget_factor * n * d_mean, 1)
    s_time = 0.0
    m_max = m_per_route if m_per_route is not None else n
    T_list = [T_budget] * k
    s_list = [s_time] * k
    m_list = [m_max] * k

    return ProblemInstance(
        n=n, k=k, alpha=alpha,
        a=a, t=t, o=o, c=c,
        T=T_list, s=s_list, m=m_list,
        d=d
    )


if __name__ == "__main__":
    p = generate_problem(n=5, k=2, seed=42)
    p.display()
