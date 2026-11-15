"""
Тесты к лабораторной работе №4.
Запуск:  python -m pytest -v lab4_test.py
(основной тест на полных данных считается около 10 секунд)
"""

from itertools import product

import pytest

import lab4


@pytest.fixture(autouse=True)
def restore_data():
    """После каждого теста возвращаем пересчитанные константы к исходным данным."""
    yield
    lab4.prepare()


def small_task(monkeypatch, fee=(0.04, 0.07, 0.05), scenarios=None, cash=50.0):
    """Маленькая задача, чтобы перебор помещался в доли секунды."""
    monkeypatch.setattr(lab4, "START_ASSETS", (50.0, 200.0, 100.0))
    monkeypatch.setattr(lab4, "START_CASH", cash)
    monkeypatch.setattr(lab4, "FEE", fee)
    if scenarios is not None:
        monkeypatch.setattr(lab4, "SCENARIOS", scenarios)
    lab4.prepare()


def flat_scenarios(mult=(1.0, 1.0, 1.0)):
    """Три этапа, в каждом одна ситуация с вероятностью 1."""
    return tuple(((1.0, mult),) for _ in range(3))


# ---------------- основной результат и сверка с ручным расчетом ----------------

def test_main_result_matches_manual_solution():
    lab4.prepare()
    start = lab4.START_ASSETS + (lab4.START_CASH,)
    value, controls = lab4.bellman(0, start)
    assert value == pytest.approx(2034.2388838)
    assert controls == [(23, 0, 0)]


def test_stage2_and_stage3_values_match_manual_tables():
    lab4.prepare()
    assert lab4.bellman(1, (810.0, 880.0, 428.0, 2.0))[0] == pytest.approx(2129.411816)
    assert lab4.bellman(1, (708.75, 816.0, 412.0, 2.0))[0] == pytest.approx(1949.565714)
    assert lab4.bellman(1, (540.0, 760.0, 400.0, 2.0))[0] == pytest.approx(1717.2208)
    assert lab4.last_stage_value((49.0, 1012.0, 432.28, 746.0))[0] == pytest.approx(2291.11472)
    assert lab4.last_stage_value((21.0, 792.0, 428.0, 746.0))[0] == pytest.approx(2029.372)


# ---------------- допустимые управления ----------------

def test_no_credit_and_minimum_respected():
    for state in [(100.0, 800.0, 400.0, 600.0), (810.0, 880.0, 428.0, 2.0)]:
        for u, after in lab4.feasible_controls(state):
            assert after[3] >= -1e-9                       # нет кредита
            for i in range(3):
                if u[i] < 0:
                    assert after[i] >= lab4.MIN_ASSET[i] - 1e-9   # не продаем ниже минимума


def test_assets_at_minimum_cannot_be_sold():
    controls = lab4.feasible_controls((30.0, 150.0, 100.0, 600.0))
    assert all(min(u) >= 0 for u, _ in controls)


def test_zero_money_and_zero_assets_only_idle_control():
    assert lab4.feasible_controls((0.0, 0.0, 0.0, 0.0)) == [((0, 0, 0), (0.0, 0.0, 0.0, 0.0))]


def test_fee_is_charged_separately():
    """Покупка ЦБ1 на 25 д.е. стоит 26, продажа приносит 24 (пример из условия)."""
    state = (100.0, 800.0, 400.0, 600.0)
    assert lab4.trade(state, (1, 0, 0))[3] == pytest.approx(574.0)
    assert lab4.trade(state, (-1, 0, 0))[3] == pytest.approx(624.0)
    assert lab4.trade(state, (0, 1, 0))[3] == pytest.approx(386.0)      # ЦБ2: 200 + 14
    assert lab4.trade(state, (0, 0, -1))[3] == pytest.approx(695.0)     # депозит: 100 - 5


# ---------------- сравнение с полным перебором ----------------

def brute_force(state, k):
    """Независимая проверка: перебор всех u в коробке без хитростей, без таблицы."""
    if k == 3:
        return sum(state)
    best = float("-inf")
    for u in product(range(-4, 5), repeat=3):
        cash = state[3]
        assets = list(state[:3])
        ok = True
        for i in range(3):
            if u[i] > 0:
                cash -= u[i] * lab4.LOT[i] * (1 + lab4.FEE[i])
            elif u[i] < 0:
                cash += -u[i] * lab4.LOT[i] * (1 - lab4.FEE[i])
            assets[i] += u[i] * lab4.LOT[i]
            if u[i] < 0 and assets[i] < lab4.MIN_ASSET[i] - 1e-9:
                ok = False
        if not ok or cash < -1e-9:
            continue
        value = sum(p * brute_force((assets[0] * m[0], assets[1] * m[1], assets[2] * m[2], cash), k + 1)
                    for p, m in lab4.SCENARIOS[k])
        best = max(best, value)
    return best


def test_dp_equals_brute_force_on_small_task(monkeypatch):
    small_task(monkeypatch)
    start = lab4.START_ASSETS + (lab4.START_CASH,)
    assert lab4.bellman(0, start)[0] == pytest.approx(brute_force(start, 0))


def test_dp_equals_brute_force_when_trading_is_profitable(monkeypatch):
    """Комиссий нет, ЦБ2 сильно растет -> на последнем этапе работает полный перебор."""
    scen = tuple(((0.5, (1.1, 1.3, 1.02)), (0.5, (0.95, 1.1, 1.0))) for _ in range(3))
    small_task(monkeypatch, fee=(0.0, 0.0, 0.0), scenarios=scen, cash=250.0)
    assert not lab4.NO_TRADE_AT_LAST_STAGE
    start = lab4.START_ASSETS + (lab4.START_CASH,)
    assert lab4.bellman(0, start)[0] == pytest.approx(brute_force(start, 0))


# ---------------- особые случаи ----------------

def test_zero_returns_and_fees_give_several_optima(monkeypatch):
    small_task(monkeypatch, fee=(0.0, 0.0, 0.0), scenarios=flat_scenarios())
    start = lab4.START_ASSETS + (lab4.START_CASH,)
    value, controls = lab4.bellman(0, start)
    assert value == pytest.approx(sum(start))
    assert len(controls) > 1 and (0, 0, 0) in controls


def test_zero_returns_with_fees_best_is_do_nothing(monkeypatch):
    small_task(monkeypatch, scenarios=flat_scenarios())
    start = lab4.START_ASSETS + (lab4.START_CASH,)
    value, controls = lab4.bellman(0, start)
    assert value == pytest.approx(sum(start))
    assert controls == [(0, 0, 0)]


# ---------------- проверка входных данных ----------------

def test_valid_input_passes():
    lab4.check_input()


def test_negative_cash_rejected(monkeypatch):
    monkeypatch.setattr(lab4, "START_CASH", -1.0)
    with pytest.raises(ValueError):
        lab4.check_input()


def test_zero_lot_rejected(monkeypatch):
    monkeypatch.setattr(lab4, "LOT", (25.0, 0.0, 100.0))
    with pytest.raises(ValueError):
        lab4.check_input()


def test_fee_one_rejected(monkeypatch):
    monkeypatch.setattr(lab4, "FEE", (0.04, 1.0, 0.05))
    with pytest.raises(ValueError):
        lab4.check_input()


def test_probabilities_not_one_rejected(monkeypatch):
    bad = list(lab4.SCENARIOS)
    bad[0] = ((0.5, (1.2, 1.1, 1.07)), (0.3, (1.05, 1.02, 1.03)), (0.1, (0.8, 0.95, 1.0)))
    monkeypatch.setattr(lab4, "SCENARIOS", tuple(bad))
    with pytest.raises(ValueError):
        lab4.check_input()


def test_incomplete_scenario_table_rejected(monkeypatch):
    monkeypatch.setattr(lab4, "SCENARIOS", lab4.SCENARIOS[:2])
    with pytest.raises(ValueError):
        lab4.check_input()
