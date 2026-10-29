"""
Лабораторная работа №2 «Метод ломаных» (метод Пиявского).
Глобальный минимум липшицевой функции на отрезке [a, b].

Константу Липшица можно получить двумя способами:
  1) через вторую производную (точно, без сетки по значениям функции);
  2) численно, по разностям на сетке с запасом k (как запасной вариант).
"""

import time
import bisect

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import brentq
from sympy.parsing.sympy_parser import parse_expr


# ---------------------------------------------------------------------------
# элементы метода Пиявского
# ---------------------------------------------------------------------------

def g(x, u, W_u, L):
    """ «галочка» g(x, u) = W(u) - L*|x - u| """
    return W_u - L * abs(x - u)


def p(x, u_list, W_list, L):
    """ ломаная p_n(x) = max_i g(x, u_i) """
    return max(g(x, u, w, L) for u, w in zip(u_list, W_list))


def find_min_p(u_list, W_list, L):
    """
    Ищем u_n = argmin p_{n-1}(x).

    Минимум ломаной лежит в одной из нижних вершин, а нижняя вершина - это
    точка пересечения галочек двух соседних верхних вершин u_i и u_{i+1}:
        u_cross = (W_i - W_{i+1}) / (2L) + (u_i + u_{i+1}) / 2
        p(u_cross) = (W_i + W_{i+1}) / 2 - L*(u_{i+1} - u_i) / 2
    Значение в нижней вершине считаем сразу по формуле, поэтому за итерацию
    достаточно одного прохода по интервалам.
    """
    best_u, best_val = None, float("inf")

    for i in range(len(u_list) - 1):
        ui, uj = u_list[i], u_list[i + 1]
        Wi, Wj = W_list[i], W_list[i + 1]

        u_cross = (Wi - Wj) / (2 * L) + (ui + uj) / 2
        u_cross = min(max(u_cross, ui), uj)        # страховка от неточного L
        val = (Wi + Wj) / 2 - L * (uj - ui) / 2

        if val < best_val:
            best_u, best_val = u_cross, val

    return best_u, best_val


def pyavsky(W, a, b, eps, L, max_iter=10000):
    """ метод ломаных на отрезке [a, b], останов по W(u_n) - p_{n-1}(u_n) < eps """
    t_start = time.perf_counter()

    # две стартовые верхние вершины: u_0 = a, u_1 = b
    u_list = [a, b]
    W_list = [W(a), W(b)]

    iterations = 0
    gap = float("inf")
    p_val = -float("inf")

    while iterations < max_iter:
        iterations += 1

        u_new, p_val = find_min_p(u_list, W_list, L)
        W_new = W(u_new)

        # новая верхняя вершина - вставляем с сохранением порядка по x
        idx = bisect.bisect_left(u_list, u_new)
        u_list.insert(idx, u_new)
        W_list.insert(idx, W_new)

        gap = W_new - p_val
        if gap < eps:
            break

    elapsed = time.perf_counter() - t_start
    best = int(np.argmin(W_list))

    return {
        "u_list": u_list,
        "W_list": W_list,
        "u_min": u_list[best],
        "W_min": W_list[best],
        "lower_bound": p_val,        # нижняя оценка глобального минимума
        "gap": gap,
        "iterations": iterations,
        "time": elapsed,
    }


# ---------------------------------------------------------------------------
# константа Липшица
# ---------------------------------------------------------------------------

def lipschitz_second_derivative(expr, a, b, n_scan=4000):
    """
    L = max |W'(x)| на [a, b].

    Принцип: на отрезке максимум |W'| достигается либо на концах, либо в точках,
    где W'' = 0 (там W' имеет экстремум). Поэтому
      1) берём W'(x) и W''(x) символьно (sympy);
      2) находим корни W''(x) = 0 на [a, b];
      3) считаем |W'| в этих корнях и на концах и берём наибольшее.

    Сетка n_scan нужна только чтобы найти промежутки, где W'' меняет знак;
    сами корни уточняются методом Брента, а L считается по точным значениям W'.
    """
    x = sp.Symbol("x")
    local = {"x": x, "pi": sp.pi, "e": sp.E}
    f = parse_expr(expr, local_dict=local)

    d1 = sp.diff(f, x)
    d2 = sp.diff(d1, x)
    f1 = sp.lambdify(x, d1, "numpy")
    f2 = sp.lambdify(x, d2, "numpy")

    xs = np.linspace(a, b, n_scan)
    v2 = np.broadcast_to(np.asarray(f2(xs), dtype=float), xs.shape)

    critical = [a, b]
    for i in range(len(xs) - 1):
        if v2[i] == 0.0:
            critical.append(xs[i])
        elif v2[i] * v2[i + 1] < 0:
            critical.append(brentq(lambda t: float(f2(t)), xs[i], xs[i + 1], xtol=1e-14))

    values = [abs(float(f1(c))) for c in critical]
    k = int(np.argmax(values))

    info = {
        "W'": str(d1),
        "W''": str(d2),
        "critical": sorted(set(critical)),
        "argmax": critical[k],
    }
    return values[k], info


def lipschitz_grid(W, a, b, n=5000, k=1.2):
    """ запасной вариант: max |ΔW/Δx| по сетке, умноженный на запас k """
    xs = np.linspace(a, b, n)
    ys = np.array([W(t) for t in xs])
    return float(np.max(np.abs(np.diff(ys) / np.diff(xs)))) * k


def get_L(expr, W, a, b, method="d2"):
    """ method = 'd2' - вторая производная, 'grid' - по сетке """
    if method == "d2":
        try:
            L, info = lipschitz_second_derivative(expr, a, b)
            return L, f"через 2-ю производную (max|W'| в x = {info['argmax']:.5f})"
        except Exception as err:
            print(f"не получилось через 2-ю производную ({err}), считаю по сетке")
    return lipschitz_grid(W, a, b), "по сетке с запасом 1.2"


# ---------------------------------------------------------------------------
# функции из строки
# ---------------------------------------------------------------------------

SAFE_ENV = {
    "np": np,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": abs, "pi": np.pi, "e": np.e,
}


def make_function(expr):
    """ строка с выражением -> функция W(x) """
    code = compile(expr, "<expr>", "eval")

    def W(x):
        env = dict(SAFE_ENV)
        env["x"] = x
        return eval(code, {"__builtins__": {}}, env)

    return W


def input_validation():
    """ спрашиваем выражение, пока оно не начнёт вычисляться """
    while True:
        expr = input("f(x) = ").strip()
        try:
            W = make_function(expr)
            W(0.0)
            return expr, W
        except Exception as err:
            print(f"ошибка в выражении: {err}, попробуйте ещё раз")


# тестовые функции: у каждой есть несколько локальных минимумов на отрезке
TESTS = [
    {"name": "Растригин", "expr": "10 + (x - 1)**2 - 10*cos(2*pi*x)", "a": -2.0, "b": 3.0},
    {"name": "Из задания", "expr": "x + sin(3.14159*x)", "a": -2.0, "b": 2.0},
    {"name": "Гриванк", "expr": "1 + x**2/4000 - cos(x)", "a": -20.0, "b": 20.0},
]


# ---------------------------------------------------------------------------
# вывод
# ---------------------------------------------------------------------------

def print_result(name, result, L):
    print("\n" + "=" * 48)
    print(f"результаты ({name}):")
    print(f"  L                = {L:.6f}")
    print(f"  u*               = {result['u_min']:.10f}")
    print(f"  W(u*)            = {result['W_min']:.10f}")
    print(f"  нижняя оценка    = {result['lower_bound']:.10f}")
    print(f"  итераций         = {result['iterations']}")
    print(f"  время            = {result['time']:.4f} сек")
    print("=" * 48)


def plot_result(W, a, b, result, L, title="метод Пиявского", save_path=None, show=True):
    """ график: W(x), ломаная p_n(x), галочки, верхние вершины и найденный минимум """
    u = np.array(result["u_list"])
    Wv = np.array(result["W_list"])
    u_min, W_min = result["u_min"], result["W_min"]

    def poly(xs):
        return np.max(Wv[None, :] - L * np.abs(xs[:, None] - u[None, :]), axis=1)

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2.2, 1]})

    # слева - весь отрезок, справа - окрестность найденного минимума
    h = (b - a) * 0.04
    for axis, (lo, hi) in ((ax, (a, b)), (axz, (max(a, u_min - h), min(b, u_min + h)))):
        xs = np.linspace(lo, hi, 1500)
        for ui, wi in zip(u, Wv):
            if axis is ax or lo <= ui <= hi:
                axis.plot(xs, wi - L * np.abs(xs - ui), color="green", lw=0.4, alpha=0.3)
        axis.plot(xs, [W(t) for t in xs], "b-", lw=2, alpha=0.75, label="W(x)")
        axis.plot(xs, poly(xs), "r--", lw=1.4, label="ломаная p_n(x)")
        mask = (u >= lo) & (u <= hi)
        axis.scatter(u[mask], Wv[mask], color="green", s=8, zorder=5, label="верхние вершины")
        axis.scatter([u_min], [W_min], color="red", s=60, edgecolors="black", zorder=6,
                     label=f"минимум ({u_min:.4f}; {W_min:.4f})")
        axis.grid(True, alpha=0.3)
        axis.set_xlabel("x")

    # по оси y ориентируемся на значения функции, а не на длинные ветки галочек
    span = Wv.max() - Wv.min()
    ax.set_ylim(Wv.min() - 0.15 * span - 0.5, Wv.max() + 0.15 * span + 0.5)

    zx = np.linspace(max(a, u_min - h), min(b, u_min + h), 400)
    zy = np.array([W(t) for t in zx])
    pad = max((zy.max() - zy.min()) * 0.3, 1e-3)
    axz.set_ylim(min(zy.min(), W_min) - pad, zy.max() + pad)

    ax.set_title(title)
    ax.set_ylabel("W(x)")
    ax.legend(loc="upper right", fontsize=8)
    axz.set_title("окрестность найденного минимума")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# ввод данных и main
# ---------------------------------------------------------------------------

def choose_function():
    """ меню тестовых функций; возвращает (expr, W, a, b, name) """
    while True:
        print("\nвыберите тестовую функцию:")
        print("-" * 70)
        print(f"{'№':>2}  {'имя':<12} {'W(x)':<35} [a, b]")
        print("-" * 70)
        for i, t in enumerate(TESTS, 1):
            print(f"{i:>2}  {t['name']:<12} {t['expr']:<35} [{t['a']}, {t['b']}]")
        print(f"{len(TESTS) + 1:>2}  своя функция")
        print("-" * 70)

        choice = input("номер: ").strip()
        if not choice.isdigit():
            print("нужно ввести номер")
            continue

        n = int(choice)
        if 1 <= n <= len(TESTS):
            t = TESTS[n - 1]
            return t["expr"], make_function(t["expr"]), t["a"], t["b"], t["name"]

        if n == len(TESTS) + 1:
            expr, W = input_validation()
            try:
                a = float(input("a = "))
                b = float(input("b = "))
            except ValueError:
                print("a и b должны быть числами")
                continue
            if a >= b:
                print("должно быть a < b")
                continue
            return expr, W, a, b, "своя функция"

        print("такого номера нет")


def ask_eps():
    while True:
        try:
            eps = float(input("eps = "))
            if eps > 0:
                return eps
            print("eps должно быть положительным")
        except ValueError:
            print("нужно ввести число")


def ask_method():
    print("\nкак искать константу Липшица?")
    print("  1 - через вторую производную (по умолчанию)")
    print("  2 - по сетке с запасом 1.2")
    return "grid" if input("номер: ").strip() == "2" else "d2"


def main():
    expr, W, a, b, name = choose_function()
    eps = ask_eps()
    method = ask_method()

    L, how = get_L(expr, W, a, b, method)
    print(f"\nфункция: {name}:  {expr}")
    print(f"отрезок: [{a}, {b}],  eps = {eps}")
    print(f"L = {L:.6f}  ({how})")

    result = pyavsky(W, a, b, eps, L)
    print_result(name, result, L)
    plot_result(W, a, b, result, L, title=f"{name}, eps = {eps}")


if __name__ == "__main__":
    main()
