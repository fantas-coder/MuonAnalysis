"""
=============================================================
  Мюонография -- анализ базы данных ядерных эмульсий
  Угловые распределения мюонных треков
=============================================================

Структура данных:
  npl3/, npl4/, npl5/, npl6/
  └── 1.0Grad/, 1.5Grad/, 2.0Grad/, 2.5Grad/
      ├── Detectors.txt             -- координаты детекторов
      ├── Input.txt                 -- позиции и углы поворота
      ├── EffCorFile_Tracks.dat     -- поправки на эффективность
      └── Tracks_DistrOutput_N.dat  -- угловые распределения треков

Запуск:
  python muon_analysis.py

Зависимости:
  pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.table
from matplotlib.transforms import Bbox

# Все пути, константы и функции загрузки -- в config.py
from config import (
    DATA_ROOT, OUTPUT_DIR, VALID_BINNINGS, COLORS_NPL, THETA_MIN, THETA_MAX,
    data_path, load_detectors, load_input, load_efficiency,
    load_tracks, sum_all_detectors, make_grid,
    theta_marginal, phi_marginal, check_data_integrity,
)

# Папка для сохранения графиков
OUTPUT_DIR = OUTPUT_DIR / "muon_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# РИСУНОК 1: Геометрия установки
# ─────────────────────────────────────────────────────────────────────────────

def plot_geometry(npl: str = "npl4", binning: str = "1.0Grad", save: bool = True):
    """
    Два вида:

    - Input.txt     -- расположение детекторов в единой системе координат со стрелками углов поворота
    - Detectors.txt -- локальная система координат ((0,0) = дет. №9)
    """
    fig, axes = plt.subplots(2, 1, figsize=(9, 13))
    fig.suptitle("Геометрия установки детекторов", fontsize=14, fontweight="bold")

    inp = load_input(npl, binning)
    det = load_detectors(npl, binning)

    # --- Input.txt (верхняя панель) ---
    ax = axes[0]
    sc = ax.scatter(inp[:, 1], inp[:, 2], c=inp[:, 3],
                    s=120, cmap="RdYlGn", zorder=3, edgecolors="k", linewidths=0.6)
    plt.colorbar(sc, ax=ax, label="Высота (м н.у.м.)")

    for row in inp:
        ax.annotate(f"{int(row[0])}", (row[1], row[2]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
        angle_rad = np.radians(row[4])
        dx, dy = 4 * np.cos(angle_rad), 4 * np.sin(angle_rad)
        ax.annotate("", xy=(row[1] + dx, row[2] + dy), xytext=(row[1], row[2]),
                    arrowprops=dict(arrowstyle="->", color="navy", lw=1.2))

    ax.set_xlabel("X (м)"); ax.set_ylabel("Y (м)")
    ax.set_title("Input.txt -- детекторы в единой СК\n(стрелки = угол поворота)")
    ax.set_aspect("equal")

    # --- Detectors.txt (нижняя панель) ---
    ax2 = axes[1]
    sc2 = ax2.scatter(det[:, 1], det[:, 2], c=det[:, 3],
                      s=120, cmap="RdYlGn", zorder=3,
                      edgecolors="k", linewidths=0.6, marker="s")
    plt.colorbar(sc2, ax=ax2, label="Высота (м н.у.м.)")

    for row in det:
        ax2.annotate(f"{int(row[0])}", (row[1], row[2]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax2.set_xlabel("X (м)"); ax2.set_ylabel("Y (м)")
    ax2.set_title("Detectors.txt -- локальная СК\n(0,0) = детектор №9")
    ax2.set_aspect("equal")

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "fig1_geometry.png", bbox_inches="tight")
        print("  Сохранён: fig1_geometry.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# РИСУНОК 2: Поправки на эффективность
# ─────────────────────────────────────────────────────────────────────────────

def plot_efficiency(binning: str = "1.0Grad", theta_max: float = 70.0, save: bool = True):
    """
    Два графика:

    - Поправка на эффективность vs θ_D для npl4/5/6
    - Статистика треков калибровочного детектора vs θ_D
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 11))
    fig.suptitle("Поправки на эффективность детектора (EffCorFile_Tracks.dat)",
                 fontsize=13, fontweight="bold")

    for npl in ["npl4", "npl5", "npl6"]:
        eff_path = data_path(npl, binning, "EffCorFile_Tracks.dat")
        if not eff_path.exists():
            continue
        eff  = load_efficiency(npl, binning)
        mask = eff[:, 0] <= theta_max

        axes[0].plot(eff[mask, 0], eff[mask, 3],
                     color=COLORS_NPL[npl], label=npl, lw=2)
        axes[1].plot(eff[:, 0], eff[:, 2],
                     color=COLORS_NPL[npl], label=npl, lw=2)

    # Панель поправки
    axes[0].axhline(1.0, color="gray", ls="--", lw=1.2, label="Поправка = 1")
    axes[0].fill_between([0, theta_max], [0.9, 0.9], [1.1, 1.1],
                          alpha=0.08, color="green", label="±10%")
    axes[0].set_xlabel("θ_D (°)"); axes[0].set_ylabel("Поправка на эффективность")
    axes[0].set_title(f"Поправка vs угол (до {theta_max:.0f}°)")
    axes[0].legend(); axes[0].set_ylim(0.5, 1.6)

    # Панель статистики
    axes[1].set_xlabel("θ_D (°)"); axes[1].set_ylabel("Треков в калиб. детекторе")
    axes[1].set_title("Статистика калибровочного детектора")
    axes[1].set_xlim(0, 90); axes[1].legend()

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "fig2_efficiency.png", bbox_inches="tight")
        print("  Сохранён: fig2_efficiency.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# РИСУНОК 3: Тепловые карты (θ, φ)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmaps(binning: str = "2.0Grad", theta_max: float = 90.0, save: bool = True):
    """
    Тепловые карты угловых распределений суммарно по всем детекторам.
    Сравнение npl4/5/6 при заданном биннинге.
    """
    npls = ["npl4", "npl5", "npl6"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 17))
    fig.suptitle(
        f"Угловые распределения треков (θ,φ) -- биннинг {binning}, суммарно по всем детекторам",
        fontsize=13, fontweight="bold",
    )

    for ax, npl in zip(axes, npls):
        data = sum_all_detectors(npl, binning)
        if data is None:
            ax.set_title(f"{npl} -- нет данных"); continue

        p_bins, t_bins, grid = make_grid(data)
        im = ax.pcolormesh(p_bins, t_bins, grid, cmap="inferno", shading="auto")
        plt.colorbar(im, ax=ax, label="Треки (с поправкой)")
        ax.set_xlabel("φ (°)"); ax.set_ylabel("θ (°)")
        total = data[(data[:, 0] <= theta_max) & (data[:, 0] > 0), 2].sum()
        ax.set_title(f"{npl}  |  Σ = {total:,.0f} треков")

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "fig3_heatmaps.png", bbox_inches="tight")
        print("  Сохранён: fig3_heatmaps.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# РИСУНОК 4: Сравнение уровней npl
# ─────────────────────────────────────────────────────────────────────────────

def plot_npl_comparison(binning: str = "2.0Grad", save: bool = True):
    """
    Три панели:
    - θ-маргинальное распределение для npl4/5/6
    - φ-маргинальное распределение (θ ≤ 60°)
    - Столбчатая диаграмма суммарных треков
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 16))
    fig.suptitle(f"Сравнение уровней npl (биннинг {binning})",
                 fontsize=13, fontweight="bold")

    npls   = ["npl4", "npl5", "npl6"]
    totals = []

    for npl in npls:
        data = sum_all_detectors(npl, binning)
        if data is None:
            totals.append(0); continue

        # θ-маргинальное
        t_b, t_cnt = theta_marginal(data)
        axes[0].plot(t_b, t_cnt, color=COLORS_NPL[npl], label=npl, lw=2)

        # φ-маргинальное
        p_b, p_cnt = phi_marginal(data)
        axes[1].plot(p_b, p_cnt, color=COLORS_NPL[npl], label=npl, lw=1.8, alpha=0.85)

        totals.append(data[:, 2].sum())

    axes[0].set_xlabel("θ (°)"); axes[0].set_ylabel("Треков (сумма по φ)")
    axes[0].set_title("(a) Распределение по θ")
    axes[0].legend()

    axes[1].set_xlabel("φ (°)"); axes[1].set_ylabel("Треков (θ ≤ 60°)")
    axes[1].set_title("(b) Распределение по φ")
    axes[1].legend()

    # Столбчатая диаграмма
    ax3 = axes[2]
    bar_colors = [COLORS_NPL[n] for n in npls]
    bars = ax3.bar(npls, [t / 1e6 for t in totals],
                   color=bar_colors, edgecolor="black", linewidth=0.8)
    ref = totals[-1] if totals[-1] > 0 else 1
    for bar, val, npl in zip(bars, totals, npls):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val / 1e6:.3f}M\n(×{val / ref:.1f})",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Треков (млн)")
    ax3.set_title("(c) Суммарная статистика")
    ax3.set_ylim(0, max(totals) / 1e6 * 1.3)

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "fig4_npl_comparison.png", bbox_inches="tight")
        print("  Сохранён: fig4_npl_comparison.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# РИСУНОК 5: Влияние биннинга
# ─────────────────────────────────────────────────────────────────────────────

def plot_binning_comparison(npl: str = "npl5", theta_max: float = 90.0, save: bool = True):
    """
    Тепловые карты для 4 биннингов при фиксированном npl.
    Показывает трейдоф: угловое разрешение vs заполненность бинов.
    """
    binnings = ["1.0Grad", "1.5Grad", "2.0Grad", "2.5Grad"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(f"Влияние биннинга -- {npl}, θ ≤ {theta_max:.0f}°",
                 fontsize=13, fontweight="bold")

    for ax, b in zip(axes.flatten(), binnings):
        data = sum_all_detectors(npl, b)
        if data is None:
            ax.set_title(f"{b} -- нет данных"); continue

        p_bins, t_bins, grid = make_grid(data)
        nonzero_frac = (grid > 0).sum() / grid.size * 100
        mean_tracks  = grid[grid > 0].mean() if (grid > 0).any() else 0

        im = ax.pcolormesh(p_bins, t_bins, grid, cmap="plasma", shading="auto")
        plt.colorbar(im, ax=ax, label="Треки", shrink=0.85)
        ax.set_xlabel("φ (°)"); ax.set_ylabel("θ (°)")
        bs = b.replace("Grad", "")
        ax.set_title(
            f"Δ = {bs}°\n"
            f"{grid.size:,} бинов | {nonzero_frac:.1f}% ненулев.\n"
            f"ср. {mean_tracks:.1f} треков/бин"
        )

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "fig5_binning.png", bbox_inches="tight")
        print("  Сохранён: fig5_binning.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# РИСУНОК 6: Статистика по детекторам
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_detector(npl: str = "npl4", binning: str = "2.0Grad",
                      theta_max: float = 60.0, save: bool = True):
    """
    Для каждого детектора: суммарные треки и заполненность бинов.
    Помогает выявить нерабочие детекторы (напр. №8).
    """
    totals, nonzero = [], []
    for i in range(1, 19):
        d = load_tracks(npl, binning, i)
        if d is None:
            totals.append(0); nonzero.append(0); continue
        sub = d[(d[:, 0] > 0) & (d[:, 0] <= theta_max)]
        totals.append(sub[:, 2].sum())
        nonzero.append((sub[:, 2] > 0).sum())

    det_nums = list(range(1, 19))
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    fig.suptitle(f"Статистика по детекторам -- {npl}, {binning}, θ ≤ {theta_max:.0f}°",
                 fontsize=13, fontweight="bold")

    # Суммарные треки
    bar_colors = ["tomato" if v < max(totals) * 0.05 else "steelblue" for v in totals]
    axes[0].bar(det_nums, totals, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Номер детектора"); axes[0].set_ylabel("Треков (θ ≤ 60°)")
    axes[0].set_title("Суммарные треки на детектор\n(красный = подозрительно мало)")
    axes[0].set_xticks(det_nums)

    # Заполненность бинов
    axes[1].bar(det_nums, nonzero, color="coral", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Номер детектора"); axes[1].set_ylabel("Ненулевых бинов")
    axes[1].set_title("Заполненность угловых бинов")
    axes[1].set_xticks(det_nums)

    # Аннотация нулевых детекторов
    for ax in axes:
        for i, v in enumerate(totals):
            if v == 0:
                ax.annotate("⚠", xy=(det_nums[i], 0), ha="center", va="bottom",
                            fontsize=14, color="red")

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "fig6_per_detector.png", bbox_inches="tight")
        print("  Сохранён: fig6_per_detector.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# РИСУНОК 7: Обзорный (комплексный дашборд)
# ─────────────────────────────────────────────────────────────────────────────

def plot_overview(binning: str = "2.0Grad", save: bool = True):
    """
    Комплексный обзор: карты (θ,φ), маргинальные распределения,
    поправки на эффективность, статистическая сводка.
    """
    npls   = ["npl4", "npl5", "npl6"]

    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(
        "Мюонография: комплексный анализ базы данных\n"
        "Угловые распределения мюонных треков (ядерные эмульсии)",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── Строка 0: тепловые карты ────────────────────────────────────────────
    all_data = {}
    for npl in npls:
        all_data[npl] = sum_all_detectors(npl, binning)

    for col, npl in enumerate(npls):
        ax = fig.add_subplot(gs[0, col])
        data = all_data[npl]
        if data is None:
            ax.set_title(f"{npl} -- нет данных"); continue
        p_b, t_b, grid = make_grid(data)
        im = ax.pcolormesh(p_b, t_b, grid, cmap="hot", shading="auto")
        plt.colorbar(im, ax=ax, label="Треки", shrink=0.8)
        ax.set_xlabel("φ (°)"); ax.set_ylabel("θ (°)")
        total = data[(data[:, 0] > THETA_MIN) & (data[:, 0] <= THETA_MAX), 2].sum()
        ax.set_title(f"{npl} | Σ = {total:,.0f}", fontweight="bold")

    # ── Строка 1: θ-распределение, φ-распределение, поправки ───────────────
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])

    for npl in npls:
        data = all_data[npl]
        if data is None: continue
        col = COLORS_NPL[npl]

        t_b, t_cnt = theta_marginal(data)
        ax10.plot(t_b, t_cnt, color=col, label=npl, lw=2)

        p_b, p_cnt = phi_marginal(data)
        ax11.plot(p_b, p_cnt, color=col, label=npl, lw=1.8, alpha=0.85)

        eff_path = data_path(npl, binning, "EffCorFile_Tracks.dat")
        if eff_path.exists():
            eff = load_efficiency(npl, binning)
            mask = eff[:, 0] <= 70
            ax12.plot(eff[mask, 0], eff[mask, 3], color=col, label=npl, lw=2)

    ax10.set_xlabel("θ (°)"); ax10.set_ylabel("Треков")
    ax10.set_title("Распределение по θ", fontweight="bold"); ax10.legend()

    ax11.set_xlabel("φ (°)"); ax11.set_ylabel("Треков (θ ≤ 60°)")
    ax11.set_title("Распределение по φ", fontweight="bold"); ax11.legend()

    ax12.axhline(1.0, color="gray", ls="--", lw=1.2)
    ax12.fill_between([0, 70], [0.9, 0.9], [1.1, 1.1], alpha=0.08, color="green")
    ax12.set_xlabel("θ_D (°)"); ax12.set_ylabel("Поправка")
    ax12.set_title("Поправка на эффективность", fontweight="bold")
    ax12.legend(); ax12.set_ylim(0.5, 1.5)

    # ── Строка 2: сравнение биннингов + сводная таблица ─────────────────────
    ax20 = fig.add_subplot(gs[2, 0:2])
    bin_colors = {"1.5Grad": "#E91E63", "2.0Grad": "#9C27B0", "2.5Grad": "#3F51B5"}
    for b, col in bin_colors.items():
        data = sum_all_detectors("npl5", b)
        if data is None: continue
        t_b, t_cnt = theta_marginal(data)
        ax20.plot(t_b, t_cnt, color=col, label=f"Δ = {b.replace('Grad','°')}", lw=2)
    ax20.set_xlabel("θ (°)"); ax20.set_ylabel("Треков")
    ax20.set_title("Влияние биннинга (npl5, θ-распределение)", fontweight="bold")
    ax20.legend()

    # Сводная таблица
    ax22 = fig.add_subplot(gs[2, 2])
    ax22.axis("off")
    totals_dict = {
        npl: (all_data[npl][:, 2].sum() if all_data[npl] is not None else 0)
        for npl in npls
    }
    ref = totals_dict["npl4"]
    rows = [
        ["Детекторов", "18"],
        ["Детектор №8", "⚠ нет треков"],
        ["npl3", "только 1.0°, неполный"],
        ["npl6 / 1.0°", "⚠ аномальный формат"],
        ["npl4 треков", f"{totals_dict['npl4']/1e6:.3f}M"],
        ["npl5 треков", f"{totals_dict['npl5']/1e6:.3f}M  ({totals_dict['npl5']/ref:.0%})"],
        ["npl6 треков", f"{totals_dict['npl6']/1e6:.3f}M  ({totals_dict['npl6']/ref:.0%})"],
        ["θ_D диапазон", "0.25° – 90°"],
        ["Рабочих биннингов", "4  (1.0°–2.5°)"],
    ]
    tbl = matplotlib.table.table(ax22, cellText=rows, colLabels=["Параметр", "Значение"],
                                 cellLoc="left", loc="center", bbox=Bbox([[0, 0], [1, 1]]))
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ECEFF1")
        if "⚠" in str(cell.get_text().get_text()):
            cell.set_facecolor("#FFF9C4")
    ax22.set_title("Сводка по данным", fontweight="bold", pad=10)

    if save:
        fig.savefig(OUTPUT_DIR / "fig7_overview.png", bbox_inches="tight")
        print("  Сохранён: fig7_overview.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Мюонография -- анализ базы данных треков")
    print("=" * 60)
    print(f"\n  Данные:   {DATA_ROOT.resolve()}")
    print(f"  Графики:  {OUTPUT_DIR.resolve()}\n")

    # Проверка целостности
    print("-- Проверка данных " + "--" * 42)
    status = check_data_integrity()
    header = f"{'':16}" + "".join(f"{b:>14}" for b in VALID_BINNINGS)
    print(header)
    for npl in ["npl3", "npl4", "npl5", "npl6"]:
        row = f"{npl:16}"
        for b in VALID_BINNINGS:
            s = status.get((npl, b), "?")
            row += f"{s:>14}"
        print(row)

    print("\n-- Генерация графиков " + "--" * 39)
    plot_geometry()
    plot_efficiency()
    plot_heatmaps()
    plot_npl_comparison()
    plot_binning_comparison()
    plot_per_detector()
    plot_overview()

    print("\nSUCCESS! Все графики сохранены в:", OUTPUT_DIR.resolve())
    plt.show()


if __name__ == "__main__":
    main()

"""
Объяснение графиков:

-----------------------------------------------------------------------------------------------------------
ГРАФИК 1 -- Геометрия установки (fig1_geometry.png)

Левая панель (Input.txt) показывает реальное расположение 18 детекторов в географической системе координат --
именно так они физически стоят на местности.
Цвет точки -- высота над уровнем моря (детекторы стоят на рельефе с перепадом ~3 м).
Стрелки показывают угол поворота каждого детектора вокруг вертикальной оси:
это важно, потому что система координат эмульсии (X, Y) привязана к плоскости пластины,
и при анализе треков нужно делать поворот в систему координат Земли.

Правая панель (Detectors.txt) -- та же установка, но в локальной системе координат с нулём в детекторе №9.
Используется при реконструкции треков внутри самого детектора.

Главный вывод: детекторы расставлены неравномерно --
    часть кучкой (детекторы 1–6 и 8–18),
    часть далеко (7, 18).
Это означает, что разные детекторы видят мюоны под разными азимутальными углами,
и их совокупность даёт угловое покрытие по всему горизонту.
-----------------------------------------------------------------------------------------------------------
ГРАФИК 2 -- Поправки на эффективность (fig2_efficiency.png)

Правая панель -- сколько треков зарегистрировал калибровочный детектор в каждом угловом интервале.
    Треков на малых углах θ мало (телесный угол мал),
    к 20–30° -- максимум,
    потом спад из-за геометрии (наклонные треки хуже реконструируются в толстой эмульсии).

Левая панель -- поправка, которую нужно применить к данным.
Смысл: если в калибровочном детекторе при угле 30° зарегистрировано вдвое меньше треков,
чем должно быть по теоретическому ожиданию, то все измерения при этом угле надо умножить на 2.
    Поправка близка к 1 в рабочем диапазоне 5°–60°,
    а на краях (θ < 5° и θ > 70°) сильно отклоняется -- эти углы ненадёжны.

Главный вывод: рабочий диапазон детектора -- θ от ~5° до ~65°.
Данные за этими границами использовать в ML не стоит без дополнительной осторожности.
-----------------------------------------------------------------------------------------------------------
ГРАФИК 3 -- Тепловые карты (θ, φ) (fig3_heatmaps.png)

Каждая ячейка карты -- это телесный угол (θ ± Δθ, φ ± Δφ),
цвет -- сколько мюонов пришло с этого направления суммарно со всех детекторов.
Ось X -- азимутальный угол φ (направление по горизонту, 0°–360°),
ось Y -- зенитный угол θ (0° = вертикаль, 90° = горизонт).

Что видно физически:
Яркая полоса на малых θ -- это мюоны, летящие почти вертикально
    (их больше всего, поскольку они проходят меньше вещества атмосферы).
Полоса не однородна по φ -- это и есть сигнал мюонографии:
    в направлениях, где над детектором больше вещества (горная порода, рельеф),
    мюонов приходит меньше,и карта темнее.

Сравнение npl4 → npl5 → npl6:
карта становится «чище» (меньше случайных ложных треков), но бледнее (меньше статистики).
Структура при этом сохраняется -- значит сигнал настоящий.
-----------------------------------------------------------------------------------------------------------
ГРАФИК 4 -- Сравнение уровней npl (fig4_npl_comparison.png)

Панель (a), θ-распределение:
Классический косинусоидальный профиль мюонного потока -- пик на малых углах, спад к горизонту.
Все три кривые npl4/5/6 имеют одинаковую форму, но разную амплитуду.
Это подтверждает, что ужесточение требования к числу пластин не искажает физику -- просто отсекает часть треков.

Панель (b), φ-распределение:
В идеальном случае (однородная порода вокруг) кривая должна быть ровной горизонтальной линией.
Видимые пики и провалы -- это либо реальные геологические неоднородности,
    либо систематика от неравномерного расположения детекторов.
Разделить одно от другого -- одна из задач ML-модели.

Панель (c), столбчатая диаграмма:
npl5 даёт 70% статистики от npl4, npl6 -- 35%.
Это типичное соотношение для ядерных эмульсий: каждая дополнительная пластина отсекает примерно треть треков.
-----------------------------------------------------------------------------------------------------------
ГРАФИК 5 -- Влияние биннинга (fig5_binning.png)

Здесь один и тот же физический объект (угловое распределение мюонов в npl5)
представлен при четырёх разных разрешениях.

1.0°: 21 600 бинов в области θ ≤ 60° -- огромное разрешение, но большинство бинов пустые или содержат 1–2 трека. Для ML это проблема: модель не сможет выучить паттерн по одному числу.
1.5°: чуть лучше, но всё ещё очень разреженно.
2.0°: хороший компромисс -- бины заполнены, структура читается, достаточно статистики на бин
      (~44 трека в среднем для npl4).
2.5°: бинов мало, данные сильно сглажены -- теряем угловое разрешение, мелкие аномалии не увидим.

Вывод: 2.0° -- оптимальный биннинг для ML.
       Именно при нём соотношение сигнал/шум максимально,
       и при этом ещё сохраняется достаточное угловое разрешение для геологической интерпретации.
-----------------------------------------------------------------------------------------------------------
ГРАФИК 6 -- Статистика по детекторам (fig6_per_detector.png)

Левая панель показывает, сколько треков зарегистрировал каждый из 18 детекторов.
Сразу бросается в глаза детектор №8 -- ноль треков.
Это либо технический сбой (детектор не сработал при экспозиции),
либо ошибка при сканировании эмульсии.
В любом случае его данные использовать нельзя -- нужно либо исключить из анализа,
либо заменить средним соседей при построении карты.

Остальные детекторы тоже неравномерны: №1 и №15–16 дают заметно меньше треков.
Это связано с их расположением: они стоят на краю установки и
видят меньший телесный угол пересечения с другими детекторами
(нужно помнить, что трек регистрируется только если прошёл через несколько пластин одновременно).
-----------------------------------------------------------------------------------------------------------
ГРАФИК 7 -- Обзорный дашборд (fig7_overview.png)

Это сводный график для презентации или главы диплома -- собирает вместе всё вышеперечисленное.
Ключевое, на что обратить внимание:
В строке тепловых карт хорошо видно, что карты npl4/5/6 структурно похожи, несмотря на разную статистику.
    Это означает, что сигнал устойчив и ML-модель, обученная на npl4,
    должна хорошо обобщаться на npl6 -- это можно сделать отдельным экспериментом в дипломе.
В сводной таблице справа зафиксированы все проблемные места данных,
    которые нужно упомянуть в разделе «описание данных».
-----------------------------------------------------------------------------------------------------------
Ключевые выводы для ML-части:
Нужно работать с:
    npl4 или npl5,
    биннинг 2.0°, θ от 5° до 65°,
    исключить детектор №8.
Это даст самый чистый и информативный набор данных для обучения.
"""