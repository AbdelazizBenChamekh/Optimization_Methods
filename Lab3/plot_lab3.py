"""
Лабораторная работа №3 (МетОпт), вариант 2 --- дополнительное задание.
Графики, иллюстрирующие геометрический смысл метода множителей Лагранжа:
    f(x, y) = 2x - y,   x^2 + y^2 = 5.

Запуск:  python plot_lab3.py
Результат: три png-файла в текущей папке.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (нужен для projection="3d")

R = np.sqrt(5.0)

# точки-кандидаты, найденные методом Лагранжа (см. lab3.py): (x, y, lambda)
P_MAX = (2.0, -1.0, -0.5)
P_MIN = (-2.0, 1.0, 0.5)


def f(x, y):
    return 2 * x - y


def grad_f():
    return np.array([2.0, -1.0])


def grad_h(x, y):
    return np.array([2 * x, 2 * y])


t = np.linspace(0, 2 * np.pi, 400)
cx, cy = R * np.cos(t), R * np.sin(t)


# ---------- 1. Плоскость: линии уровня, окружность, градиенты ----------

def plot_2d():
    fig, ax = plt.subplots(figsize=(7.5, 7))
    gx, gy = np.meshgrid(np.linspace(-4.2, 4.2, 400), np.linspace(-4.2, 4.2, 400))
    gz = f(gx, gy)

    cs = ax.contour(gx, gy, gz, levels=np.arange(-8, 9, 1), colors="0.75", linewidths=0.8)
    ax.clabel(cs, levels=[-6, -3, 0, 3, 6], fmt="f=%d", fontsize=8)

    # линии уровня, касающиеся окружности
    ax.contour(gx, gy, gz, levels=[-5, 5], colors=["tab:blue", "tab:red"], linewidths=2.2)

    ax.plot(cx, cy, color="black", lw=2.2, label=r"допустимое множество $x^2+y^2=5$")

    for (px, py, lam), color, name in ((P_MAX, "tab:red", "максимум"),
                                       (P_MIN, "tab:blue", "минимум")):
        ax.plot(px, py, "o", color=color, ms=9, zorder=5)
        gf = grad_f()
        gh = grad_h(px, py)
        ax.annotate("", xy=(px + gf[0], py + gf[1]), xytext=(px, py),
                    arrowprops=dict(arrowstyle="-|>", color="darkgreen", lw=2.2), zorder=6)
        ax.annotate("", xy=(px + 0.3 * gh[0], py + 0.3 * gh[1]), xytext=(px, py),
                    arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=2.2), zorder=6)
        val = f(px, py)
        tx, ty, ha = (4.0, -3.7, "right") if px > 0 else (-4.0, 3.7, "left")
        ax.text(tx, ty, f"{name}: ({px:.0f}, {py:.0f})\nf = {val:.0f}, λ = {lam:+.1f}",
                ha=ha, va="center", fontsize=9.5, color=color, zorder=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1))

    ax.plot([], [], color="darkgreen", lw=2.2, label=r"$\nabla f=(2,-1)$")
    ax.plot([], [], color="darkorange", lw=2.2, label=r"$0.3\,\nabla h$ (в точке)")
    ax.plot([], [], color="tab:red", lw=2.2, label="линия уровня f = 5")
    ax.plot([], [], color="tab:blue", lw=2.2, label="линия уровня f = −5")

    ax.set_aspect("equal")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Условный экстремум: линии уровня f касаются окружности\n"
                 r"в точках, где $\nabla f \parallel \nabla h$")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("lagrange_2d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- 2. Поверхность z = f(x, y) и кривая над окружностью ----------

def plot_3d():
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    gx, gy = np.meshgrid(np.linspace(-3.4, 3.4, 40), np.linspace(-3.4, 3.4, 40))
    ax.plot_surface(gx, gy, f(gx, gy), alpha=0.5, cmap="coolwarm", linewidth=0)

    ax.plot(cx, cy, np.zeros_like(t), color="black", lw=2, label="окружность в плоскости z = 0")
    ax.plot(cx, cy, f(cx, cy), color="purple", lw=2.5, label="значения f на окружности")

    for (px, py, _), color, name in ((P_MAX, "tab:red", "max"), (P_MIN, "tab:blue", "min")):
        ax.scatter([px], [py], [0], color=color, s=40)
        ax.scatter([px], [py], [f(px, py)], color=color, s=70)
        ax.plot([px, px], [py, py], [0, f(px, py)], color=color, ls="--", lw=1)
        ax.text(px, py, f(px, py) + 1.2, f"{name} = {f(px, py):.0f}", color=color, fontsize=11)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_title("Поверхность f = 2x − y и ее сечение над окружностью")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_zlim(-9, 9)
    ax.set_box_aspect((1, 1, 0.9))
    ax.view_init(elev=24, azim=-58)
    fig.savefig("lagrange_3d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- 3. f вдоль окружности как функция угла ----------

def plot_profile():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, f(cx, cy), color="purple", lw=2.2, label=r"$f(t)=\sqrt{5}\,(2\cos t-\sin t)$")
    phi = np.arctan2(1.0, 2.0)
    t_max = 2 * np.pi - phi
    t_min = np.pi - phi
    ax.plot(t_max, 5, "o", color="tab:red", ms=9, label="максимум 5, точка (2, −1)")
    ax.plot(t_min, -5, "o", color="tab:blue", ms=9, label="минимум −5, точка (−2, 1)")
    ax.axhline(5, color="tab:red", ls=":", lw=1)
    ax.axhline(-5, color="tab:blue", ls=":", lw=1)
    ax.set_xlabel("угол t на окружности, рад")
    ax.set_ylabel("f")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-7.5, 7.5)
    ax.set_title("Значения f на допустимом множестве: от −5 до 5")
    ax.legend(fontsize=8, loc="upper center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("f_on_circle.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_2d()
    plot_3d()
    plot_profile()
    print("готово: lagrange_2d.png, lagrange_3d.png, f_on_circle.png")
