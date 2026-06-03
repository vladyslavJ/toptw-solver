"""
main.py — консольний інтерфейс програмного продукту TOPTW.

Запуск:
    python main.py      (Windows)
    python3 main.py     (Linux/macOS)

Меню:
    1. Внести дані задачі
    2. Розв'язати задачу всіма алгоритмами
    3. Провести експерименти
    4. Вивести дані задачі
    0. Завершити роботу
"""
from __future__ import annotations
import json
import os
import sys

from problem import ProblemInstance
from generator import generate_problem
from greedy import greedy_solve
from genetic import genetic_solve
from experiments import (experiment_kno, experiment_beta,
                          experiment_dimensionality, plot_convergence)

# ── Глобальний стан ─────────────────────────────────────────────────────
_problem: ProblemInstance | None = None
_greedy_result = None
_genetic_result = None


# ======================================================================= #
#  Утиліти                                                                  #
# ======================================================================= #

def _ask(prompt: str, default=None):
    val = input(prompt).strip()
    return val if val else default


def _ask_int(prompt: str, default: int) -> int:
    while True:
        val = _ask(f"{prompt} (рекомендоване {default}): ", str(default))
        try:
            return int(val)
        except ValueError:
            print("  Помилка: введіть ціле число.")


def _ask_float(prompt: str, default: float) -> float:
    while True:
        val = _ask(f"{prompt} (рекомендоване {default}): ", str(default))
        try:
            return float(val)
        except ValueError:
            print("  Помилка: введіть дійсне число.")


def _status():
    if _problem is None:
        return "Задачу не задано."
    return f"Задача: n={_problem.n}, k={_problem.k}"


# ======================================================================= #
#  Меню 1 — Внести дані задачі                                              #
# ======================================================================= #

def menu_input():
    global _problem, _greedy_result, _genetic_result
    print("\nПідменю для внесення даних задачі.")
    print("Доступні опції:")
    print("  1. Внести дані вручну")
    print("  2. Згенерувати дані випадковим чином")
    print("  3. Зчитати дані з JSON-файлу")
    print("  0. Повернутись у головне меню")
    choice = _ask("Введіть число: ", "0")

    if choice == "1":
        _input_manual()
    elif choice == "2":
        _input_generate()
    elif choice == "3":
        _input_from_file()


def _input_manual():
    global _problem
    print("\nВнесення даних вручну.")
    try:
        n = int(input("Введіть кількість об'єктів n: "))
        k = int(input("Введіть кількість маршрутів k: "))
        alpha = float(input("Введіть коефіцієнт alpha (напр. 0.7): "))

        print("Введіть привабливості a[1..n] через пробіл:")
        a = list(map(float, input().split()))
        print("Введіть часи огляду t[1..n] через пробіл:")
        t_obj = list(map(float, input().split()))
        print("Введіть початки часових вікон o[0..n] через пробіл (o[0]=0 для депо):")
        o = list(map(float, input().split()))
        print("Введіть кінці часових вікон c[0..n] через пробіл:")
        c = list(map(float, input().split()))

        T_list, s_list, m_list = [], [], []
        for r in range(k):
            print(f"Маршрут {r+1}:")
            T_list.append(float(input(f"  Бюджет часу T[{r+1}]: ")))
            s_list.append(float(input(f"  Час старту s[{r+1}] (зазвичай 0): ")))
            m_list.append(int(input(f"  Макс. об'єктів m[{r+1}]: ")))

        print(f"Введіть матрицю відстаней d ({n+1}x{n+1}), по рядку через пробіл:")
        d = []
        for i in range(n + 1):
            row = list(map(float, input(f"  Рядок {i}: ").split()))
            d.append(row)

        _problem = ProblemInstance(n=n, k=k, alpha=alpha,
                                   a=a, t=t_obj, o=o, c=c,
                                   T=T_list, s=s_list, m=m_list, d=d)
        _greedy_result = None
        _genetic_result = None
        print("Нові дані задачі збережено успішно!")
    except Exception as e:
        print(f"Помилка при введенні даних: {e}")


def _input_generate():
    global _problem, _greedy_result, _genetic_result
    print("\nГенерація даних задачі.")
    n = _ask_int("Введіть кількість об'єктів n", 10)
    k = _ask_int("Введіть кількість маршрутів k", 2)
    seed_str = _ask("Seed (залиште порожнім для випадкового): ", "")
    seed = int(seed_str) if seed_str.isdigit() else None
    _problem = generate_problem(n=n, k=k, seed=seed)
    _greedy_result = None
    _genetic_result = None
    print(f"Задачу згенеровано: n={n}, k={k}. Збережено успішно!")


def _input_from_file():
    global _problem, _greedy_result, _genetic_result
    path = _ask("Введіть шлях до JSON-файлу: ", "problem.json")
    try:
        _problem = ProblemInstance.from_json(path)
        _greedy_result = None
        _genetic_result = None
        print(f"Задачу зчитано: n={_problem.n}, k={_problem.k}. Збережено успішно!")
    except FileNotFoundError:
        print(f"  Файл '{path}' не знайдено.")
    except Exception as e:
        print(f"  Помилка зчитування: {e}")


# ======================================================================= #
#  Меню 2 — Розв'язати задачу                                               #
# ======================================================================= #

def menu_solve():
    global _greedy_result, _genetic_result
    if _problem is None:
        print("Спочатку внесіть дані задачі (пункт 1).")
        return

    print("\nЗапуск жадібного алгоритму...")
    _greedy_result = greedy_solve(_problem)
    _greedy_result.display()

    print("\nПараметри генетичного алгоритму:")
    I = _ask_int("  Розмір популяції I", 20)
    b = _ask_float("  Частка Best b", 0.5)
    p_mut = _ask_float("  Імовірність мутації p_mut", 0.3)
    beta = _ask_float("  β — частка найкращих кандидатів (0..1)", 0.3)
    k_no = _ask_int("  Умова завершення k_no", 5 * _problem.n)
    A_max = _ask_int("  Макс. спроб A_max", 5)

    print("\nЗапуск генетичного алгоритму...")
    _genetic_result = genetic_solve(_problem, I=I, b=b, p_mut=p_mut,
                                    beta=beta, k_no=k_no, A_max=A_max)
    _genetic_result.display()

    # Збереження результатів
    save = _ask("\nЗберегти результати у JSON-файл? (так/ні): ", "ні")
    if save.lower() in ("так", "т", "yes", "y"):
        path = _ask("Введіть назву файлу: ", "results.json")
        data = {
            "greedy": {
                "routes": _greedy_result.routes,
                "Z": _greedy_result.objective,
                "time": _greedy_result.runtime,
            },
            "genetic": {
                "routes": _genetic_result.routes,
                "Z": _genetic_result.objective,
                "time": _genetic_result.runtime,
                "iterations": _genetic_result.extra.get("iterations"),
            }
        }
        if os.path.exists(path):
            overwrite = _ask(f"Файл '{path}' вже існує. Перезаписати? (так/ні): ", "ні")
            if overwrite.lower() not in ("так", "т", "yes", "y"):
                print("Збереження скасовано.")
                return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Результати збережено у '{path}'.")

    # Графік збіжності ГА
    if _genetic_result.extra.get("history"):
        plot_q = _ask("Зберегти графік збіжності ГА? (так/ні): ", "ні")
        if plot_q.lower() in ("так", "т", "yes", "y"):
            plot_convergence(_genetic_result.extra["history"])


# ======================================================================= #
#  Меню 3 — Провести експерименти                                           #
# ======================================================================= #

def menu_experiments():
    print("\nПідменю для проведення експериментів.")
    print("Доступні дослідження:")
    print("  1. Визначення параметра k_no")
    print("  2. Вплив параметра β на ефективність ГА")
    print("  3. Вплив розмірності на час та точність алгоритмів")
    print("  0. Повернутись у головне меню")
    choice = _ask("Введіть число: ", "0")

    if choice == "1":
        print("\nЕксперимент: визначення k_no.")
        n_str = _ask("Розмірності n через кому (напр. 10,20,50): ", "10,20,50")
        n_values = [int(x.strip()) for x in n_str.split(",")]
        R = _ask_int("  Кількість повторів R", 20)
        print("Розпочато. Це може зайняти декілька хвилин...")
        experiment_kno(n_values=n_values, R=R)

    elif choice == "2":
        print("\nЕксперимент: вплив β.")
        n = _ask_int("  Розмірність задачі n", 20)
        R = _ask_int("  Кількість повторів R", 30)
        print("Розпочато. Це може зайняти декілька хвилин...")
        experiment_beta(n=n, R=R)

    elif choice == "3":
        print("\nЕксперимент: вплив розмірності.")
        n_str = _ask("Розмірності n через кому (напр. 5,10,20,50,100): ",
                     "5,10,20,50,100")
        n_values = [int(x.strip()) for x in n_str.split(",")]
        R = _ask_int("  Кількість повторів R", 20)
        print("Розпочато. Це може зайняти декілька хвилин...")
        experiment_dimensionality(n_values=n_values, R=R)


# ======================================================================= #
#  Меню 4 — Вивести дані задачі                                             #
# ======================================================================= #

def menu_output():
    if _problem is None:
        print("Спочатку внесіть дані задачі (пункт 1).")
        return
    print("\nПідменю для виведення даних задачі.")
    print("  1. Вивести дані на екран")
    print("  2. Записати дані у JSON-файл")
    print("  0. Повернутись у головне меню")
    choice = _ask("Введіть число: ", "0")

    if choice == "1":
        print("\nВиводимо дані на екран...")
        _problem.display()

    elif choice == "2":
        path = _ask("Введіть назву файлу: ", "problem.json")
        if os.path.exists(path):
            overwrite = _ask(f"Файл '{path}' вже існує. Перезаписати? (так/ні): ", "ні")
            if overwrite.lower() not in ("так", "т", "yes", "y"):
                print("Збереження скасовано.")
                return
        _problem.to_json(path)
        print(f"Дані збережено у '{path}'.")


# ======================================================================= #
#  Головне меню                                                             #
# ======================================================================= #

def main():
    print("=" * 50)
    print("  TOPTW — Team Orienteering Problem with Time Windows")
    print("=" * 50)

    while True:
        print(f"\nГоловне меню.")
        print(f"Статус задачі: {_status()}")
        print()
        print("  1. Внести дані задачі")
        print("  2. Розв'язати задачу всіма алгоритмами")
        print("  3. Провести експерименти")
        print("  4. Вивести дані задачі")
        print("  0. Завершити роботу")
        print()
        choice = _ask("Введіть число: ", "")

        if choice == "0":
            print("Роботу завершено.")
            sys.exit(0)
        elif choice == "1":
            menu_input()
        elif choice == "2":
            menu_solve()
        elif choice == "3":
            menu_experiments()
        elif choice == "4":
            menu_output()
        else:
            print("Невідомий пункт меню. Спробуйте ще раз.")


if __name__ == "__main__":
    main()
