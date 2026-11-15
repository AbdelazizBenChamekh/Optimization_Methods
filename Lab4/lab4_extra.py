"""
Дополнительное задание: полное дерево рыночных исходов для найденной стратегии.

Запуск:  python lab4_extra.py
Создает файл decision_tree.txt (дерево) и печатает таблицу 27 конечных исходов
и сводную статистику. Стратегию берем из lab4.py (функция bellman).
"""

from itertools import product

import lab4

SHORT = ("Благ", "Нейт", "Нег")        # краткие названия ситуаций для таблицы


def fmt(state):
    return "({:.2f}; {:.2f}; {:.2f}; {:.2f})".format(*state)


def commission(u):
    """Комиссия за этап: по каждому активу |u_i| * размер пакета * ставка комиссии."""
    return sum(abs(u[i]) * lab4.LOT[i] * lab4.FEE[i] for i in range(3))


def write_tree(file, k, state, prob, indent, start_capital):
    """Рекурсивно записывает узел решения и все ветви рыночных ситуаций."""
    pad = "    " * indent
    if k == lab4.N_STAGES:
        w = lab4.capital(state)
        print(f"{pad}[КОНЕЦ] капитал W = {w:.2f}, чистый доход = {w - start_capital:+.2f}, "
              f"доходность = {(w - start_capital) / start_capital * 100:+.2f}%, P(пути) = {prob:.4f}",
              file=file)
        return

    _, controls = lab4.bellman(k, state)
    u = controls[0]
    after = lab4.trade(state, u)
    portfolio = state[0] + state[1] + state[2]
    print(f"{pad}ЭТАП {k + 1}. Портфель: ЦБ1={state[0]:.2f}, ЦБ2={state[1]:.2f}, Деп={state[2]:.2f} "
          f"(всего {portfolio:.2f}); свободные средства = {state[3]:.2f}", file=file)
    print(f"{pad}    Решение u = {u}, комиссия = {commission(u):.2f}; "
          f"после сделок: {fmt(after)}", file=file)

    for j, (p, mult) in enumerate(lab4.SCENARIOS[k]):
        nxt = lab4.apply_scenario(after, mult)
        print(f"{pad}    --> {lab4.SCENARIO_NAMES[j]} (p = {p:.2f}, накопленная P = {prob * p:.4f}), "
              f"множители {mult}", file=file)
        write_tree(file, k + 1, nxt, prob * p, indent + 2, start_capital)


def collect_outcomes():
    """Проходим стратегию по всем 27 последовательностям ситуаций."""
    start = lab4.START_ASSETS + (lab4.START_CASH,)
    start_capital = lab4.capital(start)
    outcomes = []
    for path in product(range(3), repeat=lab4.N_STAGES):
        state, prob, fees, ops = start, 1.0, 0.0, []
        for k, j in enumerate(path):
            u = lab4.bellman(k, state)[1][0]
            fees += commission(u)
            ops.append(u)
            p, mult = lab4.SCENARIOS[k][j]
            prob *= p
            state = lab4.apply_scenario(lab4.trade(state, u), mult)
        w = lab4.capital(state)
        outcomes.append({
            "path": path, "prob": prob, "ops": ops, "fees": fees,
            "capital": w, "income": w - start_capital,
            "ret": (w - start_capital) / start_capital * 100,
        })
    return outcomes, start_capital


def main():
    lab4.check_input()
    lab4.prepare()
    start = lab4.START_ASSETS + (lab4.START_CASH,)
    outcomes, start_capital = collect_outcomes()

    with open("decision_tree.txt", "w", encoding="utf-8") as f:
        print(f"Дерево исходов стратегии. Начальный капитал {start_capital:.2f} д.е.\n", file=f)
        write_tree(f, 0, start, 1.0, 0, start_capital)
    print("Дерево записано в decision_tree.txt\n")

    # --- таблица 27 исходов ---
    print(f"{'№':>2} {'Ситуации':<14}{'P':>8}  {'Операции по этапам':<30}{'Комис.':>7}"
          f"{'Капитал':>10}{'Доход':>9}{'Дох-ть':>8}")
    print("-" * 92)
    for n, o in enumerate(outcomes, 1):
        seq = "/".join(SHORT[j] for j in o["path"])
        ops = " ".join(str(u).replace(" ", "") for u in o["ops"])
        print(f"{n:>2} {seq:<14}{o['prob']:>8.4f}  {ops:<30}{o['fees']:>7.2f}"
              f"{o['capital']:>10.2f}{o['income']:>9.2f}{o['ret']:>7.2f}%")
    print("-" * 92)

    # --- статистика ---
    total_p = sum(o["prob"] for o in outcomes)
    exp_capital = sum(o["prob"] * o["capital"] for o in outcomes)
    exp_income = exp_capital - start_capital
    loss_p = sum(o["prob"] for o in outcomes if o["income"] < 0)
    worst = min(outcomes, key=lambda o: o["income"])
    best = max(outcomes, key=lambda o: o["income"])

    print(f"Сумма вероятностей всех исходов:  {total_p:.4f}")
    print(f"Ожидаемый конечный капитал:       {exp_capital:.4f} д.е.")
    print(f"Ожидаемый чистый доход:           {exp_income:.4f} д.е.")
    print(f"Ожидаемая доходность:             {exp_income / start_capital * 100:.4f} %")
    print(f"Вероятность убытка:               {loss_p * 100:.2f} %")
    print(f"Минимальный доход:                {worst['income']:.2f} д.е. "
          f"(ситуации: {'/'.join(SHORT[j] for j in worst['path'])})")
    print(f"Максимальный доход:               {best['income']:.2f} д.е. "
          f"(ситуации: {'/'.join(SHORT[j] for j in best['path'])})")
    print(f"Ожидаемые комиссии:               {sum(o['prob'] * o['fees'] for o in outcomes):.4f} д.е.")


if __name__ == "__main__":
    main()
