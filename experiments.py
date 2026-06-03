"""
experiments.py — модуль проведення експериментів для задачі TOPTW.
"""
from __future__ import annotations
import csv
import math
import os
import time
from problem import ProblemInstance
from generator import generate_problem
from greedy import greedy_solve
from genetic import genetic_solve

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False


def _ensure_output():
    os.makedirs("output", exist_ok=True)


# ======================================================================= #
#  Експеримент 1: визначення k_no                                           #
# ======================================================================= #

def experiment_kno(n_values: list, k: int = 2, R: int = 20,
                   I: int = 20, b: float = 0.5,
                   p_mut: float = 0.3, beta: float = 0.3,
                   A_max: int = 5):
    """
    Досліджує 4 варіанти залежності k_no(n):
      k_no1 = 5*n, k_no2 = 10*n,
      k_no3 = 2*n*log2(n), k_no4 = 4*n*log2(n)
    Для кожного n і варіанту: R повторів, середнє Z*.
    """
    _ensure_output()
    variants = {
        "5n":       lambda n: 5 * n,
        "10n":      lambda n: 10 * n,
        "2n·log2n": lambda n: max(5, int(2 * n * math.log2(max(n, 2)))),
        "4n·log2n": lambda n: max(5, int(4 * n * math.log2(max(n, 2)))),
    }

    results = {}  # (n, variant) → mean_Z
    for n in n_values:
        for vname, vfunc in variants.items():
            kno = vfunc(n)
            zvals = []
            for seed in range(R):
                prob = generate_problem(n=n, k=k, seed=seed * 100 + n)
                res = genetic_solve(prob, I=I, b=b, p_mut=p_mut, beta=beta,
                                    k_no=kno, A_max=A_max)
                zvals.append(res.objective)
            mean_z = sum(zvals) / len(zvals)
            results[(n, vname)] = mean_z
            print(f"  n={n:3d}, k_no={vname:10s} (={kno:4d}): mean Z* = {mean_z:.3f}")

    # CSV
    with open("output/exp_kno.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n"] + list(variants.keys()))
        for n in n_values:
            row = [n] + [round(results[(n, v)], 4) for v in variants]
            w.writerow(row)
    print("Збережено: output/exp_kno.csv")

    # Графік
    if MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(8, 5))
        for vname in variants:
            ys = [results[(n, vname)] for n in n_values]
            ax.plot(n_values, ys, marker="o", label=f"k_no = {vname}")
        ax.set_xlabel("Розмірність n")
        ax.set_ylabel("Середнє Z*")
        ax.set_title("Залежність середнього Z* від варіанту k_no(n)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("output/exp_kno.png", dpi=150)
        plt.close(fig)
        print("Збережено: output/exp_kno.png")

    return results


# ======================================================================= #
#  Експеримент 2: вплив β                                                   #
# ======================================================================= #

def experiment_beta(n: int = 20, k: int = 2, R: int = 30,
                    beta_values: list | None = None,
                    I: int = 20, b: float = 0.5,
                    p_mut: float = 0.3, k_no: int = 100,
                    A_max: int = 5):
    """
    Досліджує вплив β ∈ {0.1, 0.3, 0.5, 0.8} на середнє Z*. β — частка найкращих кандидатів, з яких виконується випадковий вибір.
    """
    _ensure_output()
    if beta_values is None:
        beta_values = [0.1, 0.3, 0.5, 0.8]

    results = {}
    for beta in beta_values:
        zvals = []
        for seed in range(R):
            prob = generate_problem(n=n, k=k, seed=seed * 13 + 7)
            res = genetic_solve(prob, I=I, b=b, p_mut=p_mut, beta=beta,
                                k_no=k_no, A_max=A_max)
            zvals.append(res.objective)
        mean_z = sum(zvals) / len(zvals)
        results[beta] = mean_z
        print(f"  β={beta:.1f}: mean Z* = {mean_z:.3f}")

    with open("output/exp_beta.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["beta", "mean_Z"])
        for beta, z in results.items():
            w.writerow([beta, round(z, 4)])
    print("Збережено: output/exp_beta.csv")

    if MATPLOTLIB:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(beta_values, [results[b] for b in beta_values],
                marker="o", color="steelblue")
        ax.set_xlabel("β (частка найкращих кандидатів)")
        ax.set_ylabel("Середнє Z*")
        ax.set_title(f"Залежність Z* від β (n={n}, k={k})")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("output/exp_beta.png", dpi=150)
        plt.close(fig)
        print("Збережено: output/exp_beta.png")

    return results


# ======================================================================= #
#  Експерименти 3 і 4: вплив розмірності (час і точність)                  #
# ======================================================================= #

def experiment_dimensionality(n_values: list, k: int = 2, R: int = 20,
                               I: int = 20, b: float = 0.5,
                               p_mut: float = 0.3, beta: float = 0.3,
                               kno_func=None, A_max: int = 5):
    """
    Для кожного n ∈ n_values генерує R задач, розв'язує ЖА та ГА,
    фіксує час і значення ЦФ.

    kno_func(n) — функція визначення k_no (за замовч. 5*n).
    """
    _ensure_output()
    if kno_func is None:
        kno_func = lambda n: 5 * n

    rows = []
    for n in n_values:
        kno = kno_func(n)
        g_times, ga_times = [], []
        g_scores, ga_scores = [], []
        deltas, wins = [], []

        for seed in range(R):
            prob = generate_problem(n=n, k=k, seed=seed * 7 + n * 3)

            g_res = greedy_solve(prob)
            ga_res = genetic_solve(prob, I=I, b=b, p_mut=p_mut, beta=beta,
                                   k_no=kno, A_max=A_max)

            g_times.append(g_res.runtime)
            ga_times.append(ga_res.runtime)
            g_scores.append(g_res.objective)
            ga_scores.append(ga_res.objective)

            zg, za = g_res.objective, ga_res.objective
            base = max(zg, 1e-9)
            delta = (za - zg) / base
            deltas.append(delta)
            wins.append(1 if za > zg else 0)

        mean_gt = sum(g_times) / R
        mean_gat = sum(ga_times) / R
        mean_gz = sum(g_scores) / R
        mean_gaz = sum(ga_scores) / R
        mean_delta = sum(deltas) / R
        win_rate = sum(wins) / R

        rows.append({
            "n": n, "k_no": kno,
            "mean_greedy_time": round(mean_gt, 5),
            "mean_ga_time": round(mean_gat, 4),
            "mean_greedy_Z": round(mean_gz, 4),
            "mean_ga_Z": round(mean_gaz, 4),
            "mean_delta": round(mean_delta, 4),
            "win_rate": round(win_rate, 3),
        })
        print(f"  n={n:3d}: t_G={mean_gt:.5f}s  t_GA={mean_gat:.4f}s  "
              f"Z_G={mean_gz:.2f}  Z_GA={mean_gaz:.2f}  "
              f"δ={mean_delta:.3f}  w={win_rate:.2f}")

    # CSV
    with open("output/exp_dim.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("Збережено: output/exp_dim.csv")

    if MATPLOTLIB:
        ns = [r["n"] for r in rows]

        # Час
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ns, [r["mean_greedy_time"] for r in rows], marker="o",
                label="Жадібний")
        ax.plot(ns, [r["mean_ga_time"] for r in rows], marker="s",
                label="Генетичний")
        ax.set_xlabel("Розмірність n")
        ax.set_ylabel("Середній час (сек)")
        ax.set_title("Час роботи алгоритмів від розмірності")
        ax.legend(); ax.grid(True)
        fig.tight_layout()
        fig.savefig("output/exp_dim_time.png", dpi=150)
        plt.close(fig)

        # Точність
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ns, [r["mean_greedy_Z"] for r in rows], marker="o",
                label="Жадібний Z*")
        ax.plot(ns, [r["mean_ga_Z"] for r in rows], marker="s",
                label="Генетичний Z*")
        ax.set_xlabel("Розмірність n")
        ax.set_ylabel("Середнє Z*")
        ax.set_title("Значення ЦФ алгоритмів від розмірності")
        ax.legend(); ax.grid(True)
        fig.tight_layout()
        fig.savefig("output/exp_dim_accuracy.png", dpi=150)
        plt.close(fig)

        print("Збережено: output/exp_dim_time.png, output/exp_dim_accuracy.png")

    return rows


# ======================================================================= #
#  Графік збіжності ГА                                                      #
# ======================================================================= #

def plot_convergence(history: list, path: str = "output/convergence.png",
                     title: str = "Динаміка рекорду Z*"):
    if not MATPLOTLIB:
        print("matplotlib не встановлено — графік не побудовано.")
        return
    _ensure_output()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Ітерація")
    ax.set_ylabel("Рекорд Z*")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Збережено: {path}")
