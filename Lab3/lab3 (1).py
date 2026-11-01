"""
Лабораторная работа №3 по МетОпт.
Метод множителей Лагранжа, вариант 2:
    f(x, y) = 2x - y  ->  extr,   x^2 + y^2 = 5
"""
from itertools import combinations

import sympy as sp

# ---------- условие задачи ----------

x, y = sp.symbols("x y", real=True)
variables = [x, y]

f = 2 * x - y
constraints = [x**2 + y**2 - 5]          # h_j(x) = 0

EPS = 1e-8                               # допуск для невязок и собственных чисел

# Эти два флага выставлены вручную по результатам анализа задачи (см. отчет):
#   окружность x^2 + y^2 = 5 замкнута и ограничена -> компакт, f непрерывна;
#   grad h = (2x, 2y) в нуль на ней не обращается -> особых точек нет.
SET_IS_COMPACT = True
CANDIDATES_COMPLETE = True


# ---------- вспомогательные функции ----------

def is_real_number(v):
    """Значение должно быть конкретным действительным числом."""
    v = sp.simplify(v)
    if v.free_symbols:
        return False
    if v.is_real is not None:
        return bool(v.is_real) and v.is_finite is not False
    try:
        return abs(complex(sp.N(v)).imag) <= EPS
    except (TypeError, ValueError):
        return False


def is_zero(expr):
    """Равно ли выражение нулю (точно или с допуском EPS)."""
    expr = sp.simplify(expr)
    if expr == 0:
        return True
    try:
        return abs(complex(sp.N(expr))) <= EPS
    except (TypeError, ValueError):
        return False


def same_point(a, b, names):
    return all(sp.simplify(a[n] - b[n]) == 0 for n in names)


def parse_solutions(raw, unknowns):
    """
    Приводит ответ solve / nonlinsolve к списку словарей {символ: значение}.
    Возвращает (список точек, флаг "решатель вернул семейство").
    """
    found, family = [], False
    if raw is None:
        return found, True
    if isinstance(raw, (dict, sp.Tuple, tuple)):
        raw = [raw]
    try:
        items = list(raw)
    except TypeError:
        return found, True

    for item in items:
        if isinstance(item, dict):
            sol = dict(item)
        elif len(item) == len(unknowns):
            sol = dict(zip(unknowns, item))
        else:
            family = True
            continue
        if any(u not in sol for u in unknowns):
            family = True                      # какая-то переменная свободна
            continue
        sol = {u: sp.simplify(sol[u]) for u in unknowns}
        if any(val.free_symbols for val in sol.values()):
            family = True                      # в ответе остался параметр
            continue
        if not all(is_real_number(val) for val in sol.values()):
            continue                           # комплексные корни не берем
        if not any(same_point(sol, old, unknowns) for old in found):
            found.append(sol)
    return found, family


def solve_all(equations, unknowns):
    """Решаем систему двумя решателями sympy и объединяем результаты."""
    points, family, used = [], False, []

    try:
        raw = sp.solve(equations, unknowns, dict=True)
        pts, fam = parse_solutions(raw, unknowns)
        points += pts
        family |= fam
        used.append("solve")
    except Exception as err:
        print("  solve не отработал:", err)

    try:
        raw = sp.nonlinsolve(equations, unknowns)
        pts, fam = parse_solutions(raw, unknowns)
        for p in pts:
            if not any(same_point(p, q, unknowns) for q in points):
                points.append(p)
        family |= fam
        used.append("nonlinsolve")
    except Exception as err:
        print("  nonlinsolve не отработал:", err)

    if points:
        status = "конечное число решений"
    elif family:
        status = "параметрическое семейство"
    elif used:
        status = "решений нет"
    else:
        status = "решить не удалось"
    return {"points": points, "family": family, "status": status, "methods": used}


def jacobian():
    return sp.Matrix([[sp.diff(c, v) for v in variables] for c in constraints])


def singular_system():
    """h_j = 0 и равенство нулю всех миноров Якоби порядка m (там ранг < m)."""
    J = jacobian()
    m, n = J.shape
    eqs = list(constraints)
    if m <= n:
        for cols in combinations(range(n), m):
            eqs.append(sp.expand(J[:, list(cols)].det()))
    return eqs


def classify(L, values):
    """Второй порядок: гессиан L сужаем на касательное подпространство."""
    J = jacobian().subs(values)
    basis = J.nullspace()
    if not basis:
        return "касательное пространство нулевое, второй порядок не применим"
    B = sp.Matrix.hstack(*basis)
    H = sp.hessian(L, variables).subs(values)
    Hr = sp.simplify(B.T * H * B)

    eig = []
    for val, mult in Hr.eigenvals().items():
        num = complex(sp.N(val))
        if abs(num.imag) > EPS:
            return "комплексные собственные числа, классификация не удалась"
        eig += [num.real] * mult

    if all(e > EPS for e in eig):
        return "строгий локальный минимум"
    if all(e < -EPS for e in eig):
        return "строгий локальный максимум"
    if any(e > EPS for e in eig) and any(e < -EPS for e in eig):
        return "не экстремум (седло)"
    return "второй порядок ответа не дает"


def show_point(d, names):
    return "(" + ", ".join(f"{n} = {d[n]}" for n in names) + ")"


# ---------- основная программа ----------

def main():
    print("Целевая функция: f =", f)
    print("Ограничения:")
    for j, c in enumerate(constraints, 1):
        print(f"  h{j} = {c} = 0")

    # множители и функция Лагранжа
    lams = list(sp.symbols(f"lam1:{len(constraints) + 1}", real=True))
    L = sp.expand(f + sum(l * c for l, c in zip(lams, constraints)))
    print("\nФункция Лагранжа:\n  L =", L)

    # система стационарности + исходные ограничения
    stationarity = [sp.diff(L, v) for v in variables]
    system = stationarity + constraints
    unknowns = variables + lams

    print("\nУравнения стационарности:")
    for v, e in zip(variables, stationarity):
        print(f"  dL/d{v} = {e} = 0")
    print("Ограничения:")
    for j, c in enumerate(constraints, 1):
        print(f"  h{j} = {c} = 0")

    print("\nМатрица Якоби ограничений:", list(jacobian()))

    # решение системы Лагранжа
    res = solve_all(system, unknowns)
    print("\nРешение системы Лагранжа:")
    print("  решатели:", ", ".join(res["methods"]))
    print("  статус:", res["status"])
    print("  найдено действительных решений:", len(res["points"]))

    # особые точки
    sing = solve_all(singular_system(), variables)
    print("\nПоиск особых точек (rank J < m):")
    print("  статус:", sing["status"])
    print("  найдено точек:", len(sing["points"]))

    # собираем кандидатов
    cands = []
    for sol in res["points"]:
        pt = {v: sol[v] for v in variables}
        if not all(is_zero(c.subs(pt)) for c in constraints):
            print("  отброшено, ограничение не выполнено:", show_point(pt, variables))
            continue
        cands.append({"pt": pt, "lam": {l: sol[l] for l in lams}, "kind": "Лагранж"})
    for sol in sing["points"]:
        pt = {v: sol[v] for v in variables}
        if any(same_point(pt, c["pt"], variables) for c in cands):
            continue
        cands.append({"pt": pt, "lam": None, "kind": "особая"})

    # анализ каждого кандидата
    J = jacobian()
    m = len(constraints)
    print("\nКандидаты:")
    for i, c in enumerate(cands, 1):
        pt = c["pt"]
        c["value"] = sp.simplify(f.subs(pt))
        c["resid"] = [sp.simplify(h.subs(pt)) for h in constraints]
        c["feasible"] = all(is_zero(r) for r in c["resid"])
        c["rank"] = int(J.subs(pt).rank())
        if c["lam"] is not None:
            c["class"] = classify(L, {**pt, **c["lam"]})
        else:
            c["class"] = "особая точка, метод Лагранжа неприменим"

        print(f"\nКандидат {i} ({c['kind']})")
        print("  точка:", show_point(pt, variables))
        if c["lam"] is not None:
            print("  множители:", show_point(c["lam"], lams))
        print("  f =", c["value"])
        print("  невязки:", c["resid"])
        print("  допустимость:", "да" if c["feasible"] else "нет")
        print(f"  ранг Якоби: {c['rank']} из {m}")
        print("  классификация:", c["class"])

    # итог
    print("\n" + "=" * 50)
    print("Итог")
    print("=" * 50)

    can_declare = (
        SET_IS_COMPACT and CANDIDATES_COMPLETE and cands
        and res["points"] and not res["family"]
        and sing["status"] in ("решений нет", "конечное число решений")
        and not sing["family"]
        and all(c["feasible"] for c in cands)
    )
    if not can_declare:
        print("Глобальные экстремумы автоматически не объявлены:")
        print("не выполнено хотя бы одно из условий (компактность, полнота")
        print("списка, конечность решений, допустимость). Нужен ручной анализ.")
        return

    best = min(cands, key=lambda c: float(sp.N(c["value"])))
    worst = max(cands, key=lambda c: float(sp.N(c["value"])))
    print("Глобальный минимум:", show_point(best["pt"], variables), "f =", best["value"])
    print("Глобальный максимум:", show_point(worst["pt"], variables), "f =", worst["value"])
    print("Основание: допустимое множество компактно, f непрерывна,")
    print("кандидаты перечислены полностью, значения f сравнены.")


if __name__ == "__main__":
    main()
