"""
greedy.py — жадібний алгоритм розв'язання задачі TOPTW.
"""
from __future__ import annotations
import time
from problem import ProblemInstance


class SolveResult:
    """Результат роботи алгоритму."""

    def __init__(self, algorithm: str, routes: list[list[int]], objective: float,
                 runtime: float, extra: dict | None = None,
                 problem: "ProblemInstance | None" = None):
        self.algorithm = algorithm
        self.routes = routes
        self.objective = objective
        self.runtime = runtime
        self.extra = extra or {}
        self.problem = problem

    def display(self) -> None:
        print(f"\n{'=' * 50}")
        print(f"Алгоритм: {self.algorithm}")
        print(f"Z* = {self.objective:.4f}")
        print(f"Час виконання: {self.runtime:.4f} сек")
        for r, route in enumerate(self.routes):
            if self.problem is not None:
                waypoints = self.problem.route_waypoint_times(route, r)
                route_str = " → ".join(f"{pt}({t:.0f}хв)" for pt, t in waypoints)
                dur = self.problem.route_duration(route, r)
                print(f"  Маршрут {r + 1}: {route_str}  (тривалість={dur:.1f}хв)")
            else:
                print(f"  Маршрут {r + 1}: {[0] + route + [0]}")
        if self.extra.get("history"):
            print(f"  Ітерацій: {self.extra.get('iterations', len(self.extra['history']) - 1)}")


def _candidate_extra_time(p: ProblemInstance, route: list[int], r: int,
                          cur_time: float, cur_pos: int, candidate: int) -> float:
    """Оцінює додаткові часові витрати вставки кандидата в кінець маршруту."""
    arr = cur_time + p.d[cur_pos][candidate]
    wait = max(0.0, p.o[candidate] - arr)
    return (p.d[cur_pos][candidate] + wait + p.t[candidate - 1] +
            p.d[candidate][0] - p.d[cur_pos][0])


def greedy_solve(problem: ProblemInstance) -> SolveResult:
    """
    Жадібний алгоритм розв'язання TOPTW.

    Для кожного маршруту послідовно додається допустимий об'єкт з найбільшим
    відношенням маржинальної привабливості до додаткових часових витрат.
    Маржинальна привабливість дорівнює a_i * alpha^q_i, де q_i — кількість
    маршрутів, до яких об'єкт уже був включений раніше.
    """
    t0 = time.perf_counter()
    p = problem
    routes: list[list[int]] = [[] for _ in range(p.k)]
    visit_count = [0] * (p.n + 1)

    for r in range(p.k):
        route: list[int] = []
        cur_time = p.s[r]
        cur_pos = 0

        while len(route) < p.m[r]:
            best_score = -1.0
            best_i = -1
            best_end_time = cur_time

            for i in range(1, p.n + 1):
                if i in route:  # повтори всередині маршруту заборонені
                    continue

                new_route = route + [i]
                if not p.route_feasible(new_route, r):
                    continue

                arr_i = cur_time + p.d[cur_pos][i]
                start_i = max(arr_i, p.o[i])
                end_i = start_i + p.t[i - 1]
                extra_time = _candidate_extra_time(p, route, r, cur_time, cur_pos, i)
                if extra_time <= 0:
                    extra_time = 1e-9

                marginal_value = p.a[i - 1] * (p.alpha ** visit_count[i])
                score = marginal_value / extra_time

                if score > best_score:
                    best_score = score
                    best_i = i
                    best_end_time = end_i

            if best_i == -1:
                break

            route.append(best_i)
            visit_count[best_i] += 1
            cur_time = best_end_time
            cur_pos = best_i

        routes[r] = route

    obj = _compute_objective(p, routes)
    runtime = time.perf_counter() - t0
    return SolveResult("Жадібний алгоритм", routes, obj, runtime, problem=problem)


def _compute_objective(p: ProblemInstance, routes: list[list[int]]) -> float:
    """Єдина точка обчислення ЦФ для всіх алгоритмів."""
    return p.objective(routes)


if __name__ == "__main__":
    from generator import generate_problem
    prob = generate_problem(n=5, k=2, seed=42)
    result = greedy_solve(prob)
    result.display()
