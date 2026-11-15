"""
Лабораторная работа №4
Решение многоэтапных оптимизационных задач методом динамического программирования
Задача пошагового управления инвестиционным портфелем

Автор: Бен Шамех Абделазиз, группа К3339, поток 1.2

Состояние перед этапом:  s = (A1, A2, D, C)
    A1, A2 - стоимость ЦБ1 и ЦБ2, D - депозит, C - свободные деньги
Управление:              u = (u1, u2, u3), целое число пакетов по каждому активу
    u > 0 - покупка, u < 0 - продажа, u = 0 - ничего не делаем
Критерий:                максимум ожидаемой конечной стоимости портфеля
"""

import sys
from math import floor
from time import perf_counter

try:
    import resource      # есть в Linux/macOS, в Windows его нет
except ImportError:
    resource = None

# ---------------------------------------------------------------
# Исходные данные (отдельно от алгоритма)
# ---------------------------------------------------------------

START_ASSETS = (100.0, 800.0, 400.0)   # ЦБ1, ЦБ2, депозит на начало периода
START_CASH = 600.0                     # свободные средства
LOT = (25.0, 200.0, 100.0)             # размер пакета = 1/4 начальной стоимости
FEE = (0.04, 0.07, 0.05)               # комиссия за операцию (доля от суммы сделки)
MIN_ASSET = (30.0, 150.0, 100.0)       # минимум, который нужно иметь

# Сценарии по этапам: (вероятность, (множитель ЦБ1, ЦБ2, депозита))
# порядок: благоприятная, нейтральная, негативная
SCENARIO_NAMES = ("Благопр.", "Нейтр.", "Негатив.")
SCENARIOS = (
    ((0.60, (1.20, 1.10, 1.07)), (0.30, (1.05, 1.02, 1.03)), (0.10, (0.80, 0.95, 1.00))),
    ((0.30, (1.40, 1.15, 1.01)), (0.20, (1.05, 1.00, 1.00)), (0.50, (0.60, 0.90, 1.00))),
    ((0.40, (1.15, 1.12, 1.05)), (0.40, (1.05, 1.01, 1.01)), (0.20, (0.70, 0.94, 1.00))),
)
N_STAGES = 3
EPS = 1e-9


def check_input():
    """Проверка исходных данных."""
    if not (len(START_ASSETS) == len(LOT) == len(FEE) == len(MIN_ASSET) == 3):
        raise ValueError("Для каждого из трех активов нужны все параметры")
    if START_CASH < 0 or any(a < 0 for a in START_ASSETS):
        raise ValueError("Начальные суммы не могут быть отрицательными")
    if any(q <= 0 for q in LOT):
        raise ValueError("Размер пакета должен быть положительным")
    if any(not 0 <= c < 1 for c in FEE):
        raise ValueError("Комиссия должна лежать в диапазоне [0, 1)")
    if any(m < 0 for m in MIN_ASSET):
        raise ValueError("Минимальные остатки не могут быть отрицательными")
    if len(SCENARIOS) != N_STAGES:
        raise ValueError("Должно быть три этапа")
    for stage in SCENARIOS:
        if abs(sum(p for p, _ in stage) - 1.0) > EPS:
            raise ValueError("Сумма вероятностей на этапе должна быть равна 1")
        for p, mult in stage:
            if p < 0 or len(mult) != 3 or any(m <= 0 for m in mult):
                raise ValueError("Неверная вероятность или множитель")


# ---------------------------------------------------------------
# Модель: допустимые управления, сделка, смена рыночной ситуации
# ---------------------------------------------------------------

def feasible_controls(state):
    """
    Все допустимые управления u в состоянии state.
    Продажи выполняются раньше покупок, поэтому сначала перебираем продажи,
    потом на все деньги (свои + выручка) перебираем покупки.
    Один и тот же актив нельзя и покупать, и продавать.
    Возвращает список пар (u, состояние после сделок).
    """
    A = state[:3]
    cash = state[3]

    # сколько пакетов можно продать, чтобы остаток не стал меньше минимума
    max_sell = [max(0, floor((A[i] - MIN_ASSET[i]) / LOT[i] + EPS)) for i in range(3)]
    price = [LOT[i] * (1 + FEE[i]) for i in range(3)]      # цена пакета с комиссией
    income = [LOT[i] * (1 - FEE[i]) for i in range(3)]     # выручка за пакет

    result = []
    for s1 in range(max_sell[0] + 1):
        for s2 in range(max_sell[1] + 1):
            for s3 in range(max_sell[2] + 1):
                sells = (s1, s2, s3)
                money = cash + sum(sells[i] * income[i] for i in range(3))

                # покупать можно только то, что не продаем
                max_buy = [0 if sells[i] > 0 else floor(money / price[i] + EPS)
                           for i in range(3)]
                for b1 in range(max_buy[0] + 1):
                    left1 = money - b1 * price[0]
                    for b2 in range(min(max_buy[1], floor(left1 / price[1] + EPS)) + 1):
                        left2 = left1 - b2 * price[1]
                        for b3 in range(min(max_buy[2], floor(left2 / price[2] + EPS)) + 1):
                            u = (b1 - s1, b2 - s2, b3 - s3)
                            new_cash = left2 - b3 * price[2]
                            new_A = tuple(A[i] + u[i] * LOT[i] for i in range(3))
                            result.append((u, new_A + (round(new_cash, 8),)))
    return result


def apply_scenario(state, mult):
    """После сделок стоимость активов умножается на множители сценария."""
    return (state[0] * mult[0], state[1] * mult[1], state[2] * mult[2], state[3])


def capital(state):
    """Конечный капитал W = A1 + A2 + D + C."""
    return state[0] + state[1] + state[2] + state[3]


def mean_multipliers(stage):
    """Средние множители E[m_i] на этапе (нумерация этапов с нуля)."""
    return tuple(sum(p * m[i] for p, m in SCENARIOS[stage]) for i in range(3))


NO_TRADE = [(0, 0, 0)]
_memo = {}


def prepare():
    """
    Считает средние множители третьего этапа и проверяет условие,
    при котором на последнем этапе сделки невыгодны. Очищает таблицу Беллмана.
    Вызывается после любого изменения исходных данных.
    """
    global MEAN3, NO_TRADE_AT_LAST_STAGE
    MEAN3 = mean_multipliers(2)
    NO_TRADE_AT_LAST_STAGE = all(1 - FEE[i] < MEAN3[i] < 1 + FEE[i] for i in range(3))
    _memo.clear()


prepare()


# ---------------------------------------------------------------
# Функция Беллмана
# V_3(s) - последний этап, V_2(s) и V_1(s) - через рекуррентное соотношение
# ---------------------------------------------------------------

def last_stage_value(state):
    """
    V3(s) - последний этап (будущего нет), возвращает (значение, оптимальные u).
    Если средний множитель актива лежит строго между (1 - c) и (1 + c),
    то и покупка, и продажа уменьшают ожидаемый капитал (см. отчет), поэтому
    оптимально ничего не делать: V3 = E[m1]*A1 + E[m2]*A2 + E[m3]*D + C.
    Если условие не выполнено - честно перебираем все управления.
    """
    if NO_TRADE_AT_LAST_STAGE:
        return MEAN3[0] * state[0] + MEAN3[1] * state[1] + MEAN3[2] * state[2] + state[3], NO_TRADE

    best, best_u = float("-inf"), []
    for u, after in feasible_controls(state):
        value = sum(p * capital(apply_scenario(after, m)) for p, m in SCENARIOS[2])
        if value > best + EPS:
            best, best_u = value, [u]
        elif abs(value - best) <= EPS:
            best_u.append(u)
    return best, best_u


def expected_next_value(k, after):
    """
    Ожидаемое значение продолжения после сделок: sum_j p_kj * V_{k+1}(f(s, u, j)).
    Для предпоследнего этапа V3 берется прямо по формуле (так считается быстрее).
    """
    total = 0.0
    if k == N_STAGES - 2 and NO_TRADE_AT_LAST_STAGE:
        a1, a2, d, c = after
        m1, m2, m3 = MEAN3
        for p, (x1, x2, x3) in SCENARIOS[k]:
            total += p * (m1 * a1 * x1 + m2 * a2 * x2 + m3 * d * x3 + c)
        return total
    for p, mult in SCENARIOS[k]:
        total += p * bellman(k + 1, apply_scenario(after, mult))[0]
    return total


def bellman(k, state):
    """
    V_k(s) = max_u  sum_j p_kj * V_{k+1}( f(s, u, j) ),   k = 0, 1 (этапы 1 и 2)
    Возвращает (значение, список всех оптимальных управлений).
    """
    if k == N_STAGES - 1:
        return last_stage_value(state)

    key = (k, tuple(round(x, 6) for x in state))
    if key in _memo:
        return _memo[key]

    best, best_u = float("-inf"), []
    for u, after in feasible_controls(state):
        value = expected_next_value(k, after)
        if value > best + EPS:
            best, best_u = value, [u]
        elif abs(value - best) <= EPS:
            best_u.append(u)

    _memo[key] = (best, best_u)
    return best, best_u


def trade(state, u):
    """Состояние после сделок для заданного u (для прямого прохода)."""
    for i in range(3):
        if u[i] > 0:
            cash_change = -u[i] * LOT[i] * (1 + FEE[i])
        elif u[i] < 0:
            cash_change = -u[i] * LOT[i] * (1 - FEE[i])
        else:
            cash_change = 0.0
        state = state[:3] + (state[3] + cash_change,)
    return tuple(state[i] + u[i] * LOT[i] for i in range(3)) + (state[3],)


# ---------------------------------------------------------------
# Прямой проход: стратегия для всех рыночных ситуаций
# ---------------------------------------------------------------

def print_policy():
    """Печатает оптимальную стратегию во всех достижимых состояниях."""
    print(f"{'Этап':<7}{'Состояние (A1; A2; D; C)':<40}{'u*':<14}{'V_k':>12}")
    print("-" * 73)

    def walk(k, state):
        if k == N_STAGES:
            return
        value, us = bellman(k, state)
        u = us[0]
        text = "({:.2f}; {:.2f}; {:.2f}; {:.2f})".format(*state)
        print(f"{k + 1:<7}{text:<40}{str(u):<14}{value:>12.4f}")
        after = trade(state, u)
        for p, mult in SCENARIOS[k]:
            walk(k + 1, apply_scenario(after, mult))

    walk(0, START_ASSETS + (START_CASH,))
    print("-" * 73)


def main():
    check_input()
    prepare()
    start = START_ASSETS + (START_CASH,)
    start_capital = capital(start)

    t0 = perf_counter()
    value, controls = bellman(0, start)
    elapsed = perf_counter() - t0

    print("=" * 73)
    print(f"Начальный капитал:                   {start_capital:.2f} д.е.")
    print(f"Оптимальное управление 1-го этапа:   {controls[0]}  (число оптимумов: {len(controls)})")
    print(f"Максимальный ожидаемый капитал V1:   {value:.6f} д.е.")
    print(f"Максимальный ожидаемый доход:        {value - start_capital:.6f} д.е.")
    print("=" * 73)
    print("\nОптимальная стратегия (прямой проход):")
    print_policy()

    print("\nЗначения V2 и V3 для проверки ручного расчета:")
    after1 = trade(start, controls[0])
    for p, mult in SCENARIOS[0]:
        s2 = apply_scenario(after1, mult)
        v2, u2 = bellman(1, s2)
        print("  V2({:.2f}; {:.2f}; {:.2f}; {:.2f}) = {:.6f}, u2* = {}".format(*s2, v2, u2[0]))
        after2 = trade(s2, u2[0])
        if abs(mult[0] - 1.20) < EPS:           # ветвь "благоприятная" на этапе 1
            for p2, m2 in SCENARIOS[1]:
                s3 = apply_scenario(after2, m2)
                print("      V3({:.2f}; {:.2f}; {:.2f}; {:.2f}) = {:.6f}".format(*s3, last_stage_value(s3)[0]))

    print("\nВремя работы: {:.2f} с, состояний в таблице Беллмана: {}".format(elapsed, len(_memo)))
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak *= 1024                      # в Linux значение в КБ, в macOS - в байтах
        print("Пиковая память процесса: {:.1f} МБ".format(peak / 1024 / 1024))


if __name__ == "__main__":
    main()
