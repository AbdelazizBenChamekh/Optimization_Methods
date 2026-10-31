"""
Лабораторная работа №2 «Метод ломаных» (метод Пиявского).
Глобальный минимум липшицевой функции на отрезке [a, b].

Константу Липшица можно получить тремя способами:
  1) через вторую производную (точно, гарантия точности eps сохраняется);
  2) численно, по разностям на сетке с запасом k (гарантии нет);
  3) ввести вручную (гарантия зависит от того, не занижена ли L).

Гарантия «найденный минимум отличается от истинного меньше чем на eps» верна
только если L не меньше настоящей константы Липшица функции на [a, b].
Программа проверяет это по ходу работы и не сообщает об успехе, если видит,
что L занижена.
"""

import math
import time
import textwrap
import bisect

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import brentq
from sympy.parsing.sympy_parser import parse_expr


MAX_ITER = 10000      # лимит итераций
TOL = 1e-9            # относительный допуск на ошибки округления
L_MIN = 1e-12         # при L <= L_MIN считаем, что функция постоянна


# ---------------------------------------------------------------------------
# элементы метода Пиявского
# ---------------------------------------------------------------------------

class LipschitzViolation(Exception):
    """ найдены соседние точки с |W(u_j) - W(u_i)| > L*(u_j - u_i): L занижена """

    def __init__(self, u1, u2, slope):
        super().__init__(f"наклон на [{u1:.6g}; {u2:.6g}] равен {slope:.6g}")
        self.u1, self.u2, self.slope = u1, u2, slope


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

    Если L настоящая, u_cross всегда попадает в [u_i, u_{i+1}]. Выход за
    границы интервала означает |W_i - W_{i+1}| > L*(u_{i+1} - u_i), то есть L
    занижена. Раньше такую точку молча «подтягивали» к интервалу, теперь
    выбрасывается LipschitzViolation.
    """
    best_u, best_val = None, float("inf")

    for i in range(len(u_list) - 1):
        ui, uj = u_list[i], u_list[i + 1]
        Wi, Wj = W_list[i], W_list[i + 1]

        slope = abs(Wi - Wj) / (uj - ui)
        if slope > L * (1 + TOL) + 1e-12:
            raise LipschitzViolation(ui, uj, slope)

        u_cross = (Wi - Wj) / (2 * L) + (ui + uj) / 2
        val = (Wi + Wj) / 2 - L * (uj - ui) / 2

        if val < best_val:
            best_u, best_val = u_cross, val

    return best_u, best_val


def pyavsky(W, a, b, eps, L, max_iter=MAX_ITER):
    """
    Метод ломаных на [a, b]. Останов по W(u_n) - p_{n-1}(u_n) < eps.

    Причина остановки записывается в result['status']:
      converged  - точность eps достигнута (success = True)
      constant   - L = 0, функция постоянна, любая точка подходит (success = True)
      max_iter   - исчерпан лимит итераций, точность НЕ достигнута
      L_too_small- обнаружено, что L меньше настоящей константы Липшица
      bad_value  - функция вернула NaN, бесконечность или ошибку
    """
    t_start = time.perf_counter()

    u_list = [a, b]
    try:
        W_list = [float(W(a)), float(W(b))]
    except Exception as err:
        raise ValueError(f"не удалось вычислить W на концах отрезка: {err}")
    if not all(math.isfinite(w) for w in W_list):
        raise ValueError("значение функции на конце отрезка не конечно (NaN или inf)")

    def finish(status, message, iterations, gap, p_val):
        best = int(np.argmin(W_list))
        return {
            "u_list": u_list, "W_list": W_list,
            "u_min": u_list[best], "W_min": W_list[best],
            "lower_bound": p_val, "gap": gap,
            "iterations": iterations,
            "time": time.perf_counter() - t_start,
            "status": status,
            "success": status in ("converged", "constant"),
            "message": message,
        }

    # L = 0: формулы с делением на 2L не работают, разбираем случай отдельно
    if L <= L_MIN:
        mid = (a + b) / 2
        try:
            vals = W_list + [float(W(mid))]
        except Exception:
            vals = [0.0, 1.0]
        scale = 1 + max(abs(v) for v in vals)
        if max(vals) - min(vals) <= TOL * scale:
            value = W_list[0]
            return finish("constant",
                          "L = 0: функция постоянна на отрезке, минимум достигается в любой точке "
                          "(взят левый конец)", 0, 0.0, value)
        return finish("L_too_small",
                      "L = 0, но значения функции в трёх точках разные: функция не постоянна, "
                      "L нужно увеличить", 0, float("inf"), -float("inf"))

    iterations, gap, p_val = 0, float("inf"), -float("inf")

    while iterations < max_iter:
        iterations += 1

        try:
            u_new, p_val = find_min_p(u_list, W_list, L)
        except LipschitzViolation as v:
            return finish("L_too_small",
                          f"L = {L:.6g} занижена: на отрезке [{v.u1:.6g}; {v.u2:.6g}] наклон функции "
                          f"{v.slope:.6g} больше L. Настоящая константа Липшица не меньше "
                          f"{v.slope:.6g}", iterations, gap, p_val)

        try:
            W_new = float(W(u_new))
        except Exception as err:
            return finish("bad_value", f"не удалось вычислить W({u_new:.6g}): {err}",
                          iterations, gap, p_val)
        if not math.isfinite(W_new):
            return finish("bad_value", f"W({u_new:.6g}) = {W_new}: значение не конечно",
                          iterations, gap, p_val)

        # новая верхняя вершина - вставляем с сохранением порядка по x
        idx = bisect.bisect_left(u_list, u_new)
        u_list.insert(idx, u_new)
        W_list.insert(idx, W_new)

        gap = W_new - p_val

        # W(u_n) ниже ломаной: миноранта оказалась выше функции, значит L занижена.
        # Просто «gap < eps» здесь нельзя - отрицательный gap тоже меньше eps.
        if gap < -TOL * (1 + abs(W_new)):
            return finish("L_too_small",
                          f"L = {L:.6g} занижена: W(u_n) = {W_new:.6g} оказалось ниже ломаной "
                          f"({p_val:.6g}), а миноранта не может лежать выше функции",
                          iterations, gap, p_val)

        if gap < eps:
            return finish("converged",
                          f"достигнута точность: W(u_n) - p = {gap:.3e} < eps = {eps:g}",
                          iterations, gap, p_val)

    return finish("max_iter",
                  f"исчерпан лимит {max_iter} итераций, достигнута только W(u_n) - p = {gap:.3e} "
                  f"(нужно < {eps:g})", iterations, gap, p_val)


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
    Если вторая производная не определена (например, |x - 1|) или вычисляется
    нечисловым значением, бросаем исключение - вызывающий код перейдёт на сетку.
    """
    x = sp.Symbol("x")
    f = parse_expr(expr, local_dict={"x": x, "pi": sp.pi, "e": sp.E})
    if f.free_symbols - {x}:
        raise ValueError(f"в выражении есть лишние переменные: {f.free_symbols - {x}}")

    d1 = sp.diff(f, x)
    d2 = sp.diff(d1, x)
    f1 = sp.lambdify(x, d1, "numpy")
    f2 = sp.lambdify(x, d2, "numpy")

    xs = np.linspace(a, b, n_scan)
    v2 = np.broadcast_to(np.asarray(f2(xs), dtype=float), xs.shape)
    if not np.all(np.isfinite(v2)):
        raise ValueError("вторая производная не определена или не конечна на отрезке")

    critical = [a, b]
    if not np.all(v2 == 0.0):               # если W'' тождественно 0, W' постоянна
        for i in range(len(xs) - 1):
            if v2[i] == 0.0:
                critical.append(xs[i])
            elif v2[i] * v2[i + 1] < 0:
                critical.append(brentq(lambda t: float(f2(t)), xs[i], xs[i + 1], xtol=1e-14))

    values = [abs(float(f1(c))) for c in critical]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("первая производная не определена или не конечна в критических точках")
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
    ys = np.array([W(t) for t in xs], dtype=float)
    if not np.all(np.isfinite(ys)):
        raise ValueError("на отрезке есть точки, где функция не конечна")
    return float(np.max(np.abs(np.diff(ys) / np.diff(xs)))) * k


def get_L(expr, W, a, b, method="d2"):
    """
    Возвращает (L, описание, exact).
    exact = True только если L найдена через вторую производную и гарантия
    точности eps сохраняется; для сетки и ручного ввода гарантии нет.
    """
    if method == "d2":
        try:
            L, info = lipschitz_second_derivative(expr, a, b)
            return L, f"через 2-ю производную (max|W'| в x = {info['argmax']:.5f})", True
        except Exception as err:
            print("\n" + "!" * 64)
            print("ВНИМАНИЕ: через вторую производную L найти не удалось.")
            print("  Вторая производная не определена или не вычисляется (так бывает,")
            print("  например, у функций с модулем |x - c|, где у W' есть излом).")
            print(f"  Техническая причина: {str(err).splitlines()[0][:70]}")
            print("Перехожу на оценку по сетке с запасом 1.2. Эта оценка приближённая,")
            print("гарантии точности eps в этом режиме нет.")
            print("!" * 64)
    return lipschitz_grid(W, a, b), "по сетке с запасом 1.2 (приближённо)", False


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


def check_finite(W, a, b, n=200):
    """ проверяем, что на отрезке функция вычисляется и значения конечны """
    # равномерная сетка плюс несколько «неудобных» долей отрезка (середина и т.п.)
    extra = [a + (b - a) * q for q in (0.5, 0.1234567, 0.3141592, 0.7182818, 0.8660254)]
    for x in list(np.linspace(a, b, n)) + extra:
        try:
            with np.errstate(all="ignore"):
                v = float(W(x))
        except Exception as err:
            return False, f"не удалось вычислить W({x:.6g}): {err}"
        if not math.isfinite(v):
            return False, f"W({x:.6g}) = {v}: функция не конечна на отрезке"
    return True, ""


# тестовые функции: у каждой есть несколько локальных минимумов на отрезке
TESTS = [
    {"name": "Растригин", "expr": "10 + (x - 1)**2 - 10*cos(2*pi*x)", "a": -2.0, "b": 3.0},
    {"name": "Из задания", "expr": "x + sin(3.14159*x)", "a": -2.0, "b": 2.0},
    {"name": "Гриванк", "expr": "1 + x**2/4000 - cos(x)", "a": -20.0, "b": 20.0},
]


# ---------------------------------------------------------------------------
# вывод
# ---------------------------------------------------------------------------

STATUS_TEXT = {
    "converged": "точность достигнута",
    "constant": "функция постоянна",
    "max_iter": "ТОЧНОСТЬ НЕ ДОСТИГНУТА (лимит итераций)",
    "L_too_small": "ТОЧНОСТЬ НЕ ДОСТИГНУТА (L занижена)",
    "bad_value": "ОСТАНОВ: функция не конечна",
}


def print_result(name, result, L, exact=True):
    width = 64
    print("\n" + "=" * width)
    print(f"результаты ({name}):")
    print(f"  статус           = {STATUS_TEXT[result['status']]}")
    print(f"  L                = {L:.6f}")
    print(f"  u*               = {result['u_min']:.10f}")
    print(f"  W(u*)            = {result['W_min']:.10f}")
    if math.isfinite(result["lower_bound"]):
        print(f"  нижняя оценка    = {result['lower_bound']:.10f}")
    print(f"  итераций         = {result['iterations']}")
    print(f"  время            = {result['time']:.4f} сек")
    print("-" * width)
    if result["success"]:
        if exact:
            print("  гарантия: найденный минимум отличается от истинного меньше чем на eps")
        else:
            print("  гарантии точности нет: L приближённая, она может быть меньше настоящей")
    else:
        print("!!! РЕЗУЛЬТАТ ПРИБЛИЖЁННЫЙ, ТОЧНОСТЬ eps НЕ ДОСТИГНУТА")
        print(textwrap.fill(f"причина: {result['message']}", width=62,
                            initial_indent="!!! ", subsequent_indent="!!!   "))
        if result["status"] == "max_iter":
            print(f"!!! на данный момент погрешность не больше {result['W_min'] - result['lower_bound']:.3e}"
                  " (если L верная)")
        if result["status"] == "L_too_small":
            print("!!! выход: увеличьте L (или выберите оценку через 2-ю производную)")
    print("=" * width)


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
    zy = np.array([W(t) for t in zx], dtype=float)
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

def ask_custom():
    """ своя функция: выражение, отрезок, проверка. Возвращает кортеж или None """
    expr = input("f(x) = ").strip()
    try:
        W = make_function(expr)
    except SyntaxError as err:
        print(f"ошибка в выражении: {err}")
        return None
    try:
        a = float(input("a = "))
        b = float(input("b = "))
    except ValueError:
        print("a и b должны быть числами")
        return None
    if a >= b:
        print("должно быть a < b")
        return None
    ok, msg = check_finite(W, a, b)
    if not ok:
        print(f"функция не подходит: {msg}")
        return None
    return expr, W, a, b, "своя функция"


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
            custom = ask_custom()
            if custom:
                return custom
            continue

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
    print("\nкак получить константу Липшица?")
    print("  1 - через вторую производную (по умолчанию)")
    print("  2 - по сетке с запасом 1.2")
    print("  3 - ввести L вручную")
    return {"2": "grid", "3": "manual"}.get(input("номер: ").strip(), "d2")


def ask_L():
    while True:
        try:
            L = float(input("L = "))
            if L >= 0 and math.isfinite(L):
                return L
            print("L должна быть неотрицательным числом")
        except ValueError:
            print("нужно ввести число")


def main():
    expr, W, a, b, name = choose_function()
    eps = ask_eps()
    method = ask_method()

    if method == "manual":
        L, how, exact = ask_L(), "введена вручную", False
    else:
        try:
            L, how, exact = get_L(expr, W, a, b, method)
        except ValueError as err:
            print(f"не удалось оценить L: {err}")
            return

    print(f"\nфункция: {name}:  {expr}")
    print(f"отрезок: [{a}, {b}],  eps = {eps}")
    print(f"L = {L:.6f}  ({how})")

    try:
        result = pyavsky(W, a, b, eps, L, max_iter=MAX_ITER)
    except ValueError as err:
        print(f"ошибка: {err}")
        return

    print_result(name, result, L, exact)

    title = f"{name}, eps = {eps}"
    if not result["success"]:
        title += "  (ТОЧНОСТЬ НЕ ДОСТИГНУТА)"
    plot_result(W, a, b, result, L, title=title)


if __name__ == "__main__":
    main()
