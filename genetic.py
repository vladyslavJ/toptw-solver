"""
genetic.py — steady-state генетичний алгоритм розв'язання задачі TOPTW.
"""
from __future__ import annotations
import copy
import math
import random
import time
from problem import ProblemInstance
from greedy import SolveResult, _compute_objective


# ======================================================================= #
#  Допоміжні функції                                                        #
# ======================================================================= #

def _candidate_extra_time(p: ProblemInstance, cur_time: float, cur_pos: int,
                          candidate: int) -> float:
    arr = cur_time + p.d[cur_pos][candidate]
    wait = max(0.0, p.o[candidate] - arr)
    return p.d[cur_pos][candidate] + wait + p.t[candidate - 1] + p.d[candidate][0] - p.d[cur_pos][0]


def _build_route_greedy_prob(p: ProblemInstance, r: int,
                             beta: float, visit_count: list[int]) -> list[int]:
    """
    Будує один маршрут жадібно-ймовірнісним методом.

    beta — частка найкращих кандидатів, з яких випадково обирається об'єкт.
    Малі значення beta дають більш жадібний вибір, великі — більш випадковий.
    visit_count враховує, скільки разів об'єкти вже включались у попередні
    маршрути цієї особини.
    """
    beta = min(max(beta, 0.0), 1.0)
    route: list[int] = []
    cur_time = p.s[r]
    cur_pos = 0

    while len(route) < p.m[r]:
        candidates = []
        for i in range(1, p.n + 1):
            if i in route:
                continue
            new_route = route + [i]
            if not p.route_feasible(new_route, r):
                continue

            extra = _candidate_extra_time(p, cur_time, cur_pos, i)
            marginal_value = p.a[i - 1] * (p.alpha ** visit_count[i])
            score = marginal_value / max(extra, 1e-9)
            candidates.append((score, i))

        if not candidates:
            break

        candidates.sort(reverse=True)
        top_k = max(1, math.ceil(max(beta, 1e-9) * len(candidates)))
        _, chosen_i = random.choice(candidates[:top_k])

        arr_i = cur_time + p.d[cur_pos][chosen_i]
        cur_time = max(arr_i, p.o[chosen_i]) + p.t[chosen_i - 1]
        cur_pos = chosen_i
        route.append(chosen_i)
        visit_count[chosen_i] += 1

    return route


def _generate_individual(p: ProblemInstance, beta: float) -> list[list[int]]:
    """Генерує одну допустиму особину — портфель з k маршрутів."""
    routes: list[list[int]] = []
    visit_count = [0] * (p.n + 1)
    for r in range(p.k):
        route = _build_route_greedy_prob(p, r, beta, visit_count)
        routes.append(route)
    return routes


def _individual_equal(ind1: list[list[int]], ind2: list[list[int]]) -> bool:
    return len(ind1) == len(ind2) and all(r1 == r2 for r1, r2 in zip(ind1, ind2))


def _in_population(ind: list[list[int]], pop: list[list[list[int]]]) -> bool:
    return any(_individual_equal(ind, p) for p in pop)


def _best_index(scores: list[float]) -> int:
    return max(range(len(scores)), key=lambda i: scores[i])


def _worst_index(scores: list[float]) -> int:
    return min(range(len(scores)), key=lambda i: scores[i])


# ======================================================================= #
#  Кросовер                                                                 #
# ======================================================================= #

def _crossover(parent_a: list[list[int]], parent_b: list[list[int]]) -> list[list[int]]:
    """Для кожного маршруту рівноймовірно бере маршрут від одного з батьків."""
    child = []
    for r in range(len(parent_a)):
        child.append(list(parent_a[r] if random.random() < 0.5 else parent_b[r]))
    return child


# ======================================================================= #
#  Мутація                                                                  #
# ======================================================================= #

def _repair_route(p: ProblemInstance, route: list[int], r: int) -> list[int]:
    """Видаляє останні об'єкти, доки маршрут не стане допустимим."""
    repaired = list(route)
    seen = set()
    without_duplicates = []
    for obj in repaired:
        if obj not in seen:
            without_duplicates.append(obj)
            seen.add(obj)
    repaired = without_duplicates
    while repaired and not p.route_feasible(repaired, r):
        repaired.pop()
    return repaired


def _top_beta_choice(items: list[tuple], beta: float):
    items.sort(reverse=True)
    top_k = max(1, math.ceil(max(min(beta, 1.0), 1e-9) * len(items)))
    return random.choice(items[:top_k])


def _mutation_candidates_for_insert_or_replace(p: ProblemInstance,
                                               individual: list[list[int]],
                                               r: int,
                                               route: list[int],
                                               mode: str) -> list[tuple]:
    """Формує варіанти вставки/заміни з оцінкою приросту ЦФ."""
    variants = []
    base_obj = p.objective(individual)
    available_objects = [obj for obj in range(1, p.n + 1) if obj not in route]

    if mode == "insert":
        for obj in available_objects:
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [obj] + route[pos:]
                if p.route_feasible(new_route, r):
                    new_ind = copy.deepcopy(individual)
                    new_ind[r] = new_route
                    gain = p.objective(new_ind) - base_obj
                    score = gain if gain > 0 else p.a[obj - 1] * 1e-6
                    variants.append((score, obj, pos, new_route))
    elif mode == "replace" and route:
        for idx in range(len(route)):
            for obj in available_objects:
                new_route = list(route)
                new_route[idx] = obj
                new_route = _repair_route(p, new_route, r)
                if p.route_feasible(new_route, r):
                    new_ind = copy.deepcopy(individual)
                    new_ind[r] = new_route
                    gain = p.objective(new_ind) - base_obj
                    score = gain if gain > 0 else p.a[obj - 1] * 1e-6
                    variants.append((score, obj, idx, new_route))
    return variants


def _mutate(p: ProblemInstance, individual: list[list[int]],
            p_mut: float, beta: float) -> list[list[int]]:
    """
    З імовірністю p_mut виконує одну з операцій над випадковим маршрутом:
    заміна, видалення або додавання. Після зміни застосовується repair.
    """
    child = copy.deepcopy(individual)
    if random.random() >= p_mut:
        return child

    r = random.randrange(p.k)
    route = list(child[r])
    ops = []
    if route:
        ops.extend(["delete", "replace"])
    if len(route) < p.m[r] and len(route) < p.n:
        ops.append("insert")
    if not ops:
        return child

    op = random.choice(ops)

    if op == "delete" and route:
        route.pop(random.randrange(len(route)))
        child[r] = route

    elif op in ("insert", "replace"):
        variants = _mutation_candidates_for_insert_or_replace(p, child, r, route, op)
        if variants:
            *_, new_route = _top_beta_choice(variants, beta)
            child[r] = new_route

    child[r] = _repair_route(p, child[r], r)
    return child


# ======================================================================= #
#  Основний ГА                                                              #
# ======================================================================= #

def genetic_solve(problem: ProblemInstance,
                  I: int = 20,
                  b: float = 0.5,
                  p_mut: float = 0.3,
                  beta: float = 0.3,
                  k_no: int = 50,
                  A_max: int = 5) -> SolveResult:
    """
    Steady-state генетичний алгоритм для задачі TOPTW.

    Повертає SolveResult з history — динамікою рекорду по ітераціях.
    """
    t0 = time.perf_counter()
    p = problem
    I = max(2, int(I))
    b = min(max(float(b), 2 / I), 1.0)
    p_mut = min(max(float(p_mut), 0.0), 1.0)
    beta = min(max(float(beta), 0.0), 1.0)
    k_no = max(1, int(k_no))
    A_max = max(1, int(A_max))

    # Генерація початкової популяції: перша особина — результат жадібного алгоритму.
    from greedy import greedy_solve
    greedy_ind = greedy_solve(p).routes
    population: list[list[list[int]]] = [greedy_ind]

    while len(population) < I:
        attempts = 0
        candidate = None
        while attempts < A_max:
            attempts += 1
            ind = _generate_individual(p, beta)
            if not _in_population(ind, population):
                candidate = ind
                break
        if candidate is None:
            candidate = _generate_individual(p, beta)
        population.append(candidate)

    scores = [_compute_objective(p, ind) for ind in population]
    best_idx = _best_index(scores)
    Z_star = scores[best_idx]
    best_ind = copy.deepcopy(population[best_idx])

    h = max(2, math.ceil(I * b))
    no_improve = 0
    iteration = 0
    history = [Z_star]

    while no_improve < k_no:
        iteration += 1

        sorted_idx = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
        best_indices = sorted_idx[:h]
        pa_idx, pb_idx = random.sample(best_indices, 2)
        child = _crossover(population[pa_idx], population[pb_idx])
        child = _mutate(p, child, p_mut, beta)
        child_score = _compute_objective(p, child)

        if not _in_population(child, population):
            population.append(child)
            scores.append(child_score)
            wi = _worst_index(scores)
            del population[wi]
            del scores[wi]
        else:
            new_ind = _generate_individual(p, beta)
            if not _in_population(new_ind, population):
                wi = _worst_index(scores)
                population[wi] = new_ind
                scores[wi] = _compute_objective(p, new_ind)

        current_best_idx = _best_index(scores)
        current_best_score = scores[current_best_idx]
        if current_best_score > Z_star:
            Z_star = current_best_score
            best_ind = copy.deepcopy(population[current_best_idx])
            no_improve = 0
        else:
            no_improve += 1

        history.append(Z_star)

    runtime = time.perf_counter() - t0
    return SolveResult(
        algorithm="Генетичний алгоритм",
        routes=best_ind,
        objective=Z_star,
        runtime=runtime,
        extra={"history": history, "iterations": iteration,
               "k_no": k_no, "I": I, "b": b, "p_mut": p_mut,
               "beta": beta, "A_max": A_max}
    )


if __name__ == "__main__":
    from generator import generate_problem
    prob = generate_problem(n=5, k=2, seed=42)
    result = genetic_solve(prob, I=10, k_no=20)
    result.display()
