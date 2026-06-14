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

def experiment_kno(n_values: list, k_values: list | None = None, R: int = 10,
                   I: int = 20, b: float = 0.5,
                   p_mut: float = 0.3, beta: float = 0.3,
                   A_max: int = 5):
    """
    Досліджує 4 варіанти залежності k_no(n, k):
      k_no1 = 5*n*k,
      k_no2 = 10*n*k,
      k_no3 = 20*n*k,
      k_no4 = 30*n*k.

    Для кожної пари (n, k) і кожного варіанту генерується R задач.
    Фіксуються середнє значення цільової функції, середній час роботи
    та середня кількість ітерацій генетичного алгоритму.
    """
    _ensure_output()
    if k_values is None:
        k_values = [2, 3, 5]

    variants = {
        "5nk": lambda n, k: 5 * n * k,
        "10nk": lambda n, k: 10 * n * k,
        "20nk": lambda n, k: 20 * n * k,
        "30nk": lambda n, k: 30 * n * k,
    }

    raw_rows = []
    summary_rows = []

    for n in n_values:
        for k in k_values:
            for vname, vfunc in variants.items():
                kno = vfunc(n, k)
                zvals = []
                times = []
                iterations = []

                for seed in range(R):
                    prob = generate_problem(n=n, k=k, seed=seed * 1000 + n * 10 + k)
                    res = genetic_solve(
                        prob,
                        I=I,
                        b=b,
                        p_mut=p_mut,
                        beta=beta,
                        k_no=kno,
                        A_max=A_max
                    )

                    zvals.append(res.objective)
                    times.append(res.runtime)
                    iterations.append(res.extra.get("iterations", 0))

                    raw_rows.append({
                        "n": n,
                        "k": k,
                        "variant": vname,
                        "k_no": kno,
                        "seed": seed,
                        "Z_ga": round(res.objective, 6),
                        "time_ga": round(res.runtime, 6),
                        "iterations": res.extra.get("iterations", 0),
                    })

                mean_z = sum(zvals) / len(zvals)
                mean_time = sum(times) / len(times)
                mean_iter = sum(iterations) / len(iterations)

                summary_rows.append({
                    "n": n,
                    "k": k,
                    "variant": vname,
                    "k_no": kno,
                    "mean_Z": round(mean_z, 6),
                    "mean_time": round(mean_time, 6),
                    "mean_iterations": round(mean_iter, 2),
                })

                print(
                    f"n={n:3d}, k={k}, k_no={vname:4s} (={kno:5d}): "
                    f"mean Z*={mean_z:.3f}, time={mean_time:.4f}s, iter={mean_iter:.1f}"
                )

    with open("output/exp_kno_raw.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["n", "k", "variant", "k_no", "seed", "Z_ga", "time_ga", "iterations"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(raw_rows)

    with open("output/exp_kno.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["n", "k", "variant", "k_no", "mean_Z", "mean_time", "mean_iterations"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    print("Збережено: output/exp_kno_raw.csv")
    print("Збережено: output/exp_kno.csv")

    if MATPLOTLIB:
        variant_names = list(variants.keys())

        avg_by_variant = {}
        for v in variant_names:
            rows_v = [row for row in summary_rows if row["variant"] == v]
            avg_by_variant[v] = {
                "mean_Z": sum(row["mean_Z"] for row in rows_v) / len(rows_v),
                "mean_time": sum(row["mean_time"] for row in rows_v) / len(rows_v),
                "mean_iterations": sum(row["mean_iterations"] for row in rows_v) / len(rows_v),
            }

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            variant_names,
            [avg_by_variant[v]["mean_Z"] for v in variant_names],
            marker="o"
        )
        ax.set_xlabel("Варіант параметра k_no")
        ax.set_ylabel("Середнє Z*")
        ax.set_title("Залежність середнього Z* від параметра зупинки ГА")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("output/exp_kno_quality.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            variant_names,
            [avg_by_variant[v]["mean_time"] for v in variant_names],
            marker="o"
        )
        ax.set_xlabel("Варіант параметра k_no")
        ax.set_ylabel("Середній час, сек")
        ax.set_title("Залежність часу роботи ГА від параметра зупинки")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("output/exp_kno_time.png", dpi=150)
        plt.close(fig)

        print("Збережено: output/exp_kno_quality.png")
        print("Збережено: output/exp_kno_time.png")

    return summary_rows

# ======================================================================= #
#  Експеримент 2: вплив β                                                   #
# ======================================================================= #

def experiment_beta(n: int = 20, k: int = 2, R: int = 30,
                    beta_values: list | None = None,
                    I: int = 20, b: float = 0.5,
                    p_mut: float = 0.3, k_no: int = 400,
                    A_max: int = 5):
    """
    Досліджує вплив β ∈ {0.1, 0.3, 0.5, 0.8} на середнє Z*,
    час роботи та кількість ітерацій генетичного алгоритму.
    """
    _ensure_output()
    if beta_values is None:
        beta_values = [0.1, 0.3, 0.5, 0.8]

    rows = []

    for beta in beta_values:
        zvals = []
        times = []
        iterations = []

        for seed in range(R):
            prob = generate_problem(n=n, k=k, seed=seed * 13 + 7)
            res = genetic_solve(
                prob,
                I=I,
                b=b,
                p_mut=p_mut,
                beta=beta,
                k_no=k_no,
                A_max=A_max
            )

            zvals.append(res.objective)
            times.append(res.runtime)
            iterations.append(res.extra.get("iterations", 0))

        mean_z = sum(zvals) / len(zvals)
        mean_time = sum(times) / len(times)
        mean_iter = sum(iterations) / len(iterations)

        rows.append({
            "beta": beta,
            "mean_Z": round(mean_z, 6),
            "mean_time": round(mean_time, 6),
            "mean_iterations": round(mean_iter, 2),
        })

        print(
            f"β={beta:.1f}: mean Z*={mean_z:.3f}, "
            f"time={mean_time:.4f}s, iter={mean_iter:.1f}"
        )

    with open("output/exp_beta.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["beta", "mean_Z", "mean_time", "mean_iterations"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("Збережено: output/exp_beta.csv")

    if MATPLOTLIB:
        betas = [row["beta"] for row in rows]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(betas, [row["mean_Z"] for row in rows], marker="o")
        ax.set_xlabel("β")
        ax.set_ylabel("Середнє Z*")
        ax.set_title(f"Залежність середнього Z* від β (n={n}, k={k})")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("output/exp_beta_quality.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(betas, [row["mean_time"] for row in rows], marker="o")
        ax.set_xlabel("β")
        ax.set_ylabel("Середній час, сек")
        ax.set_title(f"Залежність часу роботи ГА від β (n={n}, k={k})")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("output/exp_beta_time.png", dpi=150)
        plt.close(fig)

        print("Збережено: output/exp_beta_quality.png")
        print("Збережено: output/exp_beta_time.png")

    return rows

# ======================================================================= #
#  Експерименти 3 і 4: вплив розмірності (час і точність)                  #
# ======================================================================= #

def experiment_dimensionality(n_values: list, k: int = 2, R: int = 20,
                              I: int = 20, b: float = 0.5,
                              p_mut: float = 0.3, beta: float = 0.1,
                              kno_func=None, A_max: int = 5):
    """
    Для кожного n ∈ n_values генерує R задач, розв'язує їх жадібним
    та генетичним алгоритмами на однакових вхідних даних.

    Фіксуються:
      - середні значення ЦФ;
      - середні часи роботи;
      - середня відносна різниця delta;
      - частка перемог, поразок і рівних результатів ГА.
    """
    _ensure_output()
    if kno_func is None:
        kno_func = lambda n: 10 * n * k

    raw_rows = []
    summary_rows = []

    for n in n_values:
        kno = kno_func(n)

        g_times, ga_times = [], []
        g_scores, ga_scores = [], []
        deltas = []
        ga_wins = 0
        greedy_wins = 0
        equal_results = 0

        for seed in range(R):
            prob = generate_problem(n=n, k=k, seed=seed * 7 + n * 3)

            g_res = greedy_solve(prob)
            ga_res = genetic_solve(
                prob,
                I=I,
                b=b,
                p_mut=p_mut,
                beta=beta,
                k_no=kno,
                A_max=A_max
            )

            zg = g_res.objective
            za = ga_res.objective
            delta = (za - zg) / max(zg, 1e-9)

            if za > zg:
                winner = "GA"
                ga_wins += 1
            elif za < zg:
                winner = "GREEDY"
                greedy_wins += 1
            else:
                winner = "EQUAL"
                equal_results += 1

            g_times.append(g_res.runtime)
            ga_times.append(ga_res.runtime)
            g_scores.append(zg)
            ga_scores.append(za)
            deltas.append(delta)

            raw_rows.append({
                "n": n,
                "k": k,
                "seed": seed,
                "k_no": kno,
                "beta": beta,
                "Z_greedy": round(zg, 6),
                "Z_ga": round(za, 6),
                "delta": round(delta, 6),
                "winner": winner,
                "time_greedy": round(g_res.runtime, 6),
                "time_ga": round(ga_res.runtime, 6),
                "iterations_ga": ga_res.extra.get("iterations", 0),
            })

        mean_gt = sum(g_times) / R
        mean_gat = sum(ga_times) / R
        mean_gz = sum(g_scores) / R
        mean_gaz = sum(ga_scores) / R
        mean_delta = sum(deltas) / R

        time_ratio = mean_gat / max(mean_gt, 1e-9)

        summary_rows.append({
            "n": n,
            "k": k,
            "k_no": kno,
            "beta": beta,
            "mean_greedy_time": round(mean_gt, 6),
            "mean_ga_time": round(mean_gat, 6),
            "time_ratio": round(time_ratio, 2),
            "mean_greedy_Z": round(mean_gz, 6),
            "mean_ga_Z": round(mean_gaz, 6),
            "mean_delta": round(mean_delta, 6),
            "ga_wins": ga_wins,
            "greedy_wins": greedy_wins,
            "equal_results": equal_results,
            "ga_win_rate": round(ga_wins / R, 3),
            "greedy_win_rate": round(greedy_wins / R, 3),
            "equal_rate": round(equal_results / R, 3),
        })

        print(
            f"n={n:3d}, k={k}: "
            f"t_G={mean_gt:.6f}s, t_GA={mean_gat:.4f}s, "
            f"Z_G={mean_gz:.3f}, Z_GA={mean_gaz:.3f}, "
            f"delta={mean_delta:.4f}, "
            f"GA/G/E={ga_wins}/{greedy_wins}/{equal_results}"
        )

    with open("output/exp_dim_raw.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "n", "k", "seed", "k_no", "beta",
            "Z_greedy", "Z_ga", "delta", "winner",
            "time_greedy", "time_ga", "iterations_ga"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(raw_rows)

    with open("output/exp_dim.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "n", "k", "k_no", "beta",
            "mean_greedy_time", "mean_ga_time", "time_ratio",
            "mean_greedy_Z", "mean_ga_Z", "mean_delta",
            "ga_wins", "greedy_wins", "equal_results",
            "ga_win_rate", "greedy_win_rate", "equal_rate"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    print("Збережено: output/exp_dim_raw.csv")
    print("Збережено: output/exp_dim.csv")

    if MATPLOTLIB:
        ns = [r["n"] for r in summary_rows]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ns, [r["mean_greedy_time"] for r in summary_rows], marker="o",
                label="Жадібний алгоритм")
        ax.plot(ns, [r["mean_ga_time"] for r in summary_rows], marker="s",
                label="Генетичний алгоритм")
        ax.set_xlabel("Кількість туристичних об'єктів n")
        ax.set_ylabel("Середній час, сек")
        ax.set_title("Залежність часу роботи алгоритмів від розмірності")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("output/exp_dim_time.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ns, [r["mean_greedy_Z"] for r in summary_rows], marker="o",
                label="Жадібний алгоритм")
        ax.plot(ns, [r["mean_ga_Z"] for r in summary_rows], marker="s",
                label="Генетичний алгоритм")
        ax.set_xlabel("Кількість туристичних об'єктів n")
        ax.set_ylabel("Середнє значення Z*")
        ax.set_title("Залежність значення ЦФ від розмірності задачі")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("output/exp_dim_accuracy.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ns, [r["mean_delta"] for r in summary_rows], marker="o")
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Кількість туристичних об'єктів n")
        ax.set_ylabel("Середня відносна різниця")
        ax.set_title("Відносна різниця результатів ГА та жадібного алгоритму")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig("output/exp_dim_delta.png", dpi=150)
        plt.close(fig)

        print("Збережено: output/exp_dim_time.png")
        print("Збережено: output/exp_dim_accuracy.png")
        print("Збережено: output/exp_dim_delta.png")

    return summary_rows

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
