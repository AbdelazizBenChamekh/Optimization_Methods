"""
Лабораторная работа: решение задачи линейного программирования (ЗЛП)
симплекс-методом с вспомогательной задачей (метод из лекции).

Что делает программа:
  1. приводит ЗЛП к каноническому виду (max -> min, свободные переменные
     x = x+ - x-, отрицательная правая часть, добавочные переменные
     для неравенств) и печатает полученную каноническую задачу;
  2. если допустимый базис сразу не виден, строит и решает
     вспомогательную задачу: искусственная переменная добавляется в те
     ограничения, где нет готовой базисной переменной (столбца с единицей
     только в этой строке); цель - сумма искусственных переменных;
  3. записывает начальную симплекс-таблицу основной задачи и доводит
     её до оптимума;
  4. возвращается к исходным переменным и печатает ответ.

Все вычисления идут в обыкновенных дробях (fractions.Fraction),
поэтому результат можно сверять с ручным решением без погрешностей.

Формат task.txt:
    строка 1  - коэффициенты целевой функции c1 ... cn
    строка 2  - 'max' или 'min'
    далее     - ограничения:  a1 ... an  знак  b   (знак: <=, =, >=)
    необязательная строка:  free 2 3
                 номера переменных (с единицы), у которых нет условия x >= 0

Запуск:  python lab1.py [task.txt] [-v]
    -v  печатать все симплекс-таблицы и выбранный разрешающий элемент
"""
import sys
from fractions import Fraction as Fr


# ---------------------------------------------------------------- ввод ---
def read_task(path="task.txt"):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    c = [Fr(x) for x in lines[0].split()]
    mode = lines[1].lower()
    if mode not in ("min", "max"):
        raise ValueError("вторая строка task.txt должна быть 'min' или 'max'")
    A, ops, b, free = [], [], [], set()
    for line in lines[2:]:
        parts = line.replace(":", " ").split()
        if parts[0].lower() == "free":
            free |= {int(k) - 1 for k in parts[1:]}
            continue
        if parts[-2] not in ("<=", "=", ">="):
            raise ValueError(f"не понял знак в строке: {line}")
        row = [Fr(x) for x in parts[:-2]]
        if len(row) != len(c):
            raise ValueError(f"в строке '{line}' не {len(c)} коэффициентов")
        A.append(row)
        ops.append(parts[-2])
        b.append(Fr(parts[-1]))
    return c, mode, A, ops, b, free


# ---------------------------------------------------------- оформление ---
def fmt(v):
    """Дробь в виде '3', '-7/2'."""
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def fmt_dec(v):
    s = fmt(v)
    return s if v.denominator == 1 else f"{s} (≈{float(v):.6g})"


def expr(coefs, names=None):
    """Строка вида 'x1 + 2x2 - x3' по списку коэффициентов."""
    out = []
    for k, a in enumerate(coefs):
        if a == 0:
            continue
        name = names[k] if names else f"x{k + 1}"
        mag = abs(a)
        term = name if mag == 1 else f"{fmt(mag)}{name}"
        out.append(("- " if a < 0 else "+ ") + term if out else ("-" if a < 0 else "") + term)
    return " ".join(out) if out else "0"


# --------------------------------------------- приведение к канон. виду ---
def to_canonical(c, mode, A, ops, b, free):
    """Возвращает (c_k, A_k, b_k, info).
    Переменные канонической задачи нумеруются подряд:
      1..n         исходные (у свободной это x+),
      n+1..n+f     x- свободных переменных,
      далее        добавочные (slack: +1 для <=, surplus: -1 для >=)."""
    n, m = len(c), len(A)
    c = list(c) if mode == "min" else [-v for v in c]        # max -> min
    A = [row[:] for row in A]
    b, ops = b[:], ops[:]

    minus_of = {}                                             # j -> номер столбца x-
    for j in sorted(free):
        minus_of[j] = len(c)
        c.append(-c[j])
        for row in A:
            row.append(-row[j])

    for i in range(m):                                        # b >= 0
        if b[i] < 0:
            A[i] = [-v for v in A[i]]
            b[i] = -b[i]
            ops[i] = {"<=": ">=", ">=": "<="}.get(ops[i], ops[i])

    for i in range(m):                                        # добавочные
        if ops[i] == "=":
            continue
        for k in range(m):
            A[k].append(Fr(0))
        A[i][-1] = Fr(1) if ops[i] == "<=" else Fr(-1)
        c.append(Fr(0))

    info = {"n": n, "m": m, "minus_of": minus_of, "ops": ops}
    return c, A, b, info


def print_canonical(c, A, b, info):
    print("Каноническая задача (все переменные >= 0, правые части >= 0):")
    print("  W =", expr(c), "-> min")
    for row, bi in zip(A, b):
        print("  ", expr(row), "=", fmt(bi))
    for j, k in info["minus_of"].items():
        print(f"  (свободная x{j + 1} = x{j + 1} - x{k + 1})")


# ---------------------------------------------------- симплекс-таблица ---
class Table:
    """Компактная таблица из лекции: строки - базисные переменные,
    столбцы - свободные, внизу p_j и справа внизу -Q."""

    def __init__(self, rows, cols, A, b, p, q):
        self.rows, self.cols = rows[:], cols[:]
        self.A = [r[:] for r in A]
        self.b, self.p, self.q = b[:], p[:], q

    def show(self, title, pivot=None):
        print(title)
        head = [""] + [f"x{k + 1}" for k in self.cols] + [""]
        body = []
        for i, r in enumerate(self.rows):
            line = [f"x{r + 1}"]
            for j in range(len(self.cols)):
                s = fmt(self.A[i][j])
                line.append(f"[{s}]" if pivot == (i, j) else s)
            line.append(fmt(self.b[i]))
            body.append(line)
        body.append([""] + [fmt(v) for v in self.p] + [fmt(self.q)])
        w = [max(len(row[k]) for row in [head] + body) for k in range(len(head))]
        for row in [head] + body:
            print("  " + "  ".join(s.rjust(w[k]) for k, s in enumerate(row)))

    def choose(self, aux_from=None):
        """Выбор разрешающего элемента. Возвращает ('optimal'|'unbounded'|'step', (i, j)).
        Если несколько столбцов имеют одинаковое наименьшее p_j (в лекции можно
        брать любой), во вспомогательной задаче берём тот, при котором из базиса
        уходит искусственная переменная: так задача заканчивается быстрее."""
        if not self.p or min(self.p) >= 0:
            return "optimal", None
        best = min(self.p)
        picks = []
        for j in range(len(self.p)):
            if self.p[j] != best:
                continue
            cand = [(self.b[i] / self.A[i][j], i)
                    for i in range(len(self.rows)) if self.A[i][j] > 0]
            if cand:
                picks.append((j, min(cand)[1]))
        if not picks:
            return "unbounded", None
        if aux_from is not None:
            for j, i in picks:
                if self.rows[i] >= aux_from:
                    return "step", (i, j)
        j, i = picks[0]
        return "step", (i, j)

    def pivot(self, i, j):
        """Шаги 3-4 лекции (формулы пересчёта) + обмен индексов."""
        a = self.A[i][j]
        m, k = len(self.rows), len(self.cols)
        nA = [[None] * k for _ in range(m)]
        nb, np_ = [None] * m, [None] * k
        for r in range(m):
            for s in range(k):
                if r == i and s == j:
                    nA[r][s] = 1 / a
                elif r == i:
                    nA[r][s] = self.A[i][s] / a
                elif s == j:
                    nA[r][s] = -self.A[r][j] / a
                else:
                    nA[r][s] = self.A[r][s] - self.A[i][s] * self.A[r][j] / a
            nb[r] = self.b[i] / a if r == i else self.b[r] - self.b[i] * self.A[r][j] / a
        for s in range(k):
            np_[s] = -self.p[j] / a if s == j else self.p[s] - self.A[i][s] * self.p[j] / a
        self.q = self.q - self.b[i] * self.p[j] / a
        self.A, self.b, self.p = nA, nb, np_
        self.rows[i], self.cols[j] = self.cols[j], self.rows[i]

    def drop_col(self, j):
        del self.cols[j], self.p[j]
        for r in self.A:
            del r[j]


def run_simplex(T, verbose, title, drop_from=None):
    """Цикл симплекс-шагов. drop_from - с какого номера переменные
    искусственные: их столбец вычёркивается, когда переменная стала свободной."""
    step = 0
    while True:
        status, rc = T.choose(aux_from=drop_from)
        if status != "step":
            return status
        step += 1
        i, j = rc
        if verbose:
            T.show(f"{title}, шаг {step}: вводим x{T.cols[j] + 1}, "
                   f"выводим x{T.rows[i] + 1}, разрешающий элемент = {fmt(T.A[i][j])}",
                   pivot=rc)
        T.pivot(i, j)
        if drop_from is not None and T.cols[j] >= drop_from:
            T.drop_col(j)                       # правило 1 лекции (слайд 12)


def row_basis(A):
    """Для каждой строки номер готовой базисной переменной (столбец с единицей
    в этой строке и нулями в остальных) или None, если такой нет."""
    m, N = len(A), len(A[0])
    res, used = [None] * m, set()
    for i in range(m):
        for k in range(N):
            if k not in used and A[i][k] == 1 and all(A[r][k] == 0 for r in range(m) if r != i):
                res[i] = k
                used.add(k)
                break
    return res


# ------------------------------------------------------------ решение ---
def solve(path="task.txt", verbose=False):
    c0, mode, A0, ops, b0, free = read_task(path)
    c, A, b, info = to_canonical(c0, mode, A0, ops, b0, free)
    m, N = len(A), len(c)
    print_canonical(c, A, b, info)

    rb = row_basis(A)
    if any(v is None for v in rb):
        # ---- вспомогательная задача: искусственные x_{N+1}, ... только там, где нет базиса
        aux_rows = [i for i in range(m) if rb[i] is None]
        aux_of = {i: N + k for k, i in enumerate(aux_rows)}
        names = ", ".join(f"x{aux_of[i] + 1} (в ограничение {i + 1})" for i in aux_rows)
        print(f"\nДопустимый базис сразу не виден: строим вспомогательную задачу, "
              f"искусственные переменные: {names}.")
        rows0 = [rb[i] if rb[i] is not None else aux_of[i] for i in range(m)]
        cols0 = [k for k in range(N) if k not in rows0]
        T = Table(rows0, cols0, [[A[i][k] for k in cols0] for i in range(m)], b,
                  [-sum(A[i][k] for i in aux_rows) for k in cols0],
                  -sum(b[i] for i in aux_rows))
        if verbose:
            T.show("Начальная таблица вспомогательной задачи:")
        run_simplex(T, verbose, "Вспомогательная задача", drop_from=N)
        w = -T.q
        if verbose:
            T.show("Итоговая таблица вспомогательной задачи:")
        if w > 0:
            print(f"\nРешений нет: вспомогательная задача решена с W^a = {fmt(w)} > 0, "
                  "допустимая область пуста.")
            return None
        print(f"\nВспомогательная задача решена, W^a = 0: допустимая область не пуста.")
        # искусственная переменная могла остаться в базисе с нулевым значением
        for i in range(m - 1, -1, -1):
            if T.rows[i] >= N:
                js = [j for j in range(len(T.cols)) if T.A[i][j] != 0]
                if js:
                    T.pivot(i, js[0])
                    if T.cols[js[0]] >= N:
                        T.drop_col(js[0])
                else:                                   # строка лишняя (зависимое ограничение)
                    del T.rows[i], T.A[i], T.b[i]
        rows, cols, TA, Tb = T.rows, T.cols, T.A, T.b
    else:
        print("\nДопустимый базис виден сразу: вспомогательная задача не нужна.")
        rows = rb
        cols = [k for k in range(N) if k not in rb]
        TA = [[A[i][j] for j in cols] for i in range(m)]
        Tb = b[:]

    # ---- начальная таблица основной задачи: выражаем W через свободные
    p = [c[cols[j]] - sum(c[rows[i]] * TA[i][j] for i in range(len(rows)))
         for j in range(len(cols))]
    q = -sum(c[rows[i]] * Tb[i] for i in range(len(rows)))
    T = Table(rows, cols, TA, Tb, p, q)
    if verbose:
        T.show("\nНачальная таблица основной задачи:")
    status = run_simplex(T, verbose, "Основная задача")
    if status == "unbounded":
        print("\nРешений нет: целевая функция не ограничена снизу "
              "(в столбце с отрицательным p_j нет положительных элементов).")
        return None
    if verbose:
        T.show("Итоговая таблица основной задачи:")

    # ---- ответ: базисные = правая часть, свободные = 0, затем x = x+ - x-
    xk = [Fr(0)] * N
    for i, r in enumerate(T.rows):
        xk[r] = T.b[i]
    x = [xk[j] - (xk[info["minus_of"][j]] if j in info["minus_of"] else 0)
         for j in range(info["n"])]
    z = sum(ci * xi for ci, xi in zip(c0, x))
    print("\nОптимальная точка: x = (" + ", ".join(fmt(v) for v in x) + ")")
    label = "Максимальное" if mode == "max" else "Минимальное"
    print(f"{label} значение Z = {fmt_dec(z)}")
    return x, z


if __name__ == "__main__":
    task_path = "task.txt"
    for a in sys.argv[1:]:
        if not a.startswith("-"):
            task_path = a
    solve(task_path, verbose="-v" in sys.argv)
