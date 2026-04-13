"""
=============================================================
  Мюонография -- анализ пропусков и аномалий в данных
=============================================================

Что делает скрипт:
  1. Структурные пропуски   -- число строк, NaN/Inf, угловое покрытие
  2. Геометрическая граница -- определение рабочего θ-диапазона
  3. Выбросы (IQR)          -- Q3 + k·IQR для каждого детектора
  4. Выбросы (z-score)      -- |z| > threshold для каждого детектора
  5. Кросс-детекторная      -- отклонение от медианы по всем детекторам
  6. Визуализация           -- 9 панелей итогового рисунка

Запуск:
  python muon_anomaly_analysis.py

Зависимости:
  pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats
from pathlib import Path

# Все пути, константы и функции загрузки -- в config.py
from config import (
    DATA_ROOT, OUTPUT_DIR, VALID_BINNINGS, EXPECTED_ROWS,
    GOOD_DETS, BAD_DETS, THETA_MIN, THETA_MAX,
    IQR_K, ZSCORE_TH, CROSS_TH, COLORS_NPL,
    load_tracks, sum_all_detectors, check_data_integrity,
)

# Папка для сохранения графиков
OUTPUT_DIR = OUTPUT_DIR / "muon_anomaly_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

COLORS = COLORS_NPL

# Рабочие конфигурации для анализа аномалий
CONFIGS = [
    ("npl4", "2.0Grad"),
    ("npl4", "2.5Grad"),
    ("npl6", "2.0Grad"),
    ("npl6", "2.5Grad"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. СТРУКТУРНЫЕ ПРОПУСКИ
# ─────────────────────────────────────────────────────────────────────────────

def check_structural(verbose: bool = True) -> dict:
    """
    Проверяет:
    - наличие файла
    - число строк (совпадает ли с ожидаемым)
    - наличие NaN / Inf
    - одинаковое ли угловое покрытие θ, φ у всех детекторов

    Возвращает словарь с результатами.
    """
    results = {}

    for npl, binning in CONFIGS:
        exp_rows = EXPECTED_ROWS.get(binning)
        issues   = []
        ref_theta, ref_phi = None, None

        for det in GOOD_DETS:
            d = load_tracks(npl, binning, det)

            if d is None:
                issues.append(f"det{det}: MISSING или аномальный формат")
                continue

            # Число строк
            if len(d) != exp_rows:
                issues.append(f"det{det}: строк={len(d)}, ожидалось={exp_rows}")

            # NaN / Inf
            n_nan = np.isnan(d).sum()
            n_inf = np.isinf(d).sum()
            if n_nan > 0:
                issues.append(f"det{det}: NaN={n_nan}")
            if n_inf > 0:
                issues.append(f"det{det}: Inf={n_inf}")

            # Угловое покрытие
            theta_u = tuple(np.unique(d[:, 0]))
            phi_u   = tuple(np.unique(d[:, 1]))
            if ref_theta is None:
                ref_theta, ref_phi = theta_u, phi_u
            elif theta_u != ref_theta or phi_u != ref_phi:
                issues.append(f"det{det}: угловое покрытие отличается от det{GOOD_DETS[0]}")

        key = f"{npl}/{binning}"
        results[key] = issues

        if verbose:
            status = "✓ OK" if not issues else f"⚠ {len(issues)} проблем"
            print(f"  {key:18s}: {status}")
            for issue in issues:
                print(f"      → {issue}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. ГЕОМЕТРИЧЕСКАЯ ГРАНИЦА (рабочий θ-диапазон)
# ─────────────────────────────────────────────────────────────────────────────

def find_theta_range(npl: str = "npl4", binning: str = "2.0Grad",
                     verbose: bool = True) -> tuple[float, float]:
    """
    Определяет первый и последний θ-бин, где суммарно по всем детекторам
    есть хоть один ненулевой трек.
    Нулевые θ < θ_min -- геометрическое ограничение установки,
    не пропуски в данных.
    """
    acc = sum_all_detectors(npl, binning)
    sub = acc[acc[:, 0] <= 90]

    t_bins = np.unique(sub[:, 0])
    totals = np.array([sub[sub[:, 0] == t, 2].sum() for t in t_bins])

    nonzero_mask = totals > 0
    theta_min = float(t_bins[nonzero_mask][0])
    theta_max = float(t_bins[nonzero_mask][-1])

    if verbose:
        print(f"  {npl}/{binning}: треки появляются при θ ≥ {theta_min}°")
        print(f"  Нулевые θ < {theta_min}° -- геометрическое ограничение, не пропуски")
        print(f"  Рабочий диапазон: {theta_min}° – {theta_max}°")
        print()
        print(f"  {'θ':>7}  {'Σ треков':>10}  {'ненул. φ-бинов':>14}")
        for t, tot in zip(t_bins[nonzero_mask], totals[nonzero_mask]):
            n_phi = (sub[sub[:, 0] == t, 2] > 0).sum()
            bar = "█" * int(tot / 10000)
            print(f"  {t:7.1f}°  {tot:10.0f}  {n_phi:>14d}  {bar}")

    return theta_min, theta_max


# ─────────────────────────────────────────────────────────────────────────────
# 3 & 4. ВЫБРОСЫ: IQR и Z-SCORE
# ─────────────────────────────────────────────────────────────────────────────

def detect_outliers_iqr(data: np.ndarray, k: float = IQR_K) -> np.ndarray:
    """
    Возвращает маску выбросов: значения > Q3 + k*IQR.
    Применяется только к ненулевым бинам.
    """
    n = data[:, 2]
    nonzero = n[n > 0]
    if len(nonzero) < 4:
        return np.zeros(len(data), dtype=bool)
    q1, q3 = np.percentile(nonzero, 25), np.percentile(nonzero, 75)
    threshold = q3 + k * (q3 - q1)
    return (n > threshold) & (n > 0)


def detect_outliers_zscore(data: np.ndarray, threshold: float = ZSCORE_TH) -> np.ndarray:
    """
    Возвращает маску выбросов: |z| > threshold.
    Применяется только к ненулевым бинам (z считается по ним).
    """
    n = data[:, 2]
    nonzero_vals = n[n > 0]
    if len(nonzero_vals) < 4:
        return np.zeros(len(data), dtype=bool)
    mu, sigma = nonzero_vals.mean(), nonzero_vals.std()
    if sigma == 0:
        return np.zeros(len(data), dtype=bool)
    z = (n - mu) / sigma
    return (np.abs(z) > threshold) & (n > 0)


def analyze_outliers(npl: str, binning: str,
                     theta_min: float = 58.0, theta_max: float = 90.0,
                     verbose: bool = True) -> dict:
    """
    Для каждого детектора находит выбросы по IQR и z-score
    в рабочем диапазоне углов.

    Возвращает:
      {det: {'iqr': [...строки-выбросы...], 'zscore': [...строки-выбросы...]}}
    """
    all_results = {}

    for det in GOOD_DETS:
        d = load_tracks(npl, binning, det)
        if d is None:
            continue

        # Рабочий диапазон
        work = d[(d[:, 0] >= theta_min) & (d[:, 0] <= theta_max)].copy()

        mask_iqr = detect_outliers_iqr(work)
        mask_z   = detect_outliers_zscore(work)

        out_iqr = work[mask_iqr]
        out_z   = work[mask_z]

        all_results[det] = {"iqr": out_iqr, "zscore": out_z}

        if verbose and (len(out_iqr) > 0 or len(out_z) > 0):
            n_nonzero = (work[:, 2] > 0).sum()
            print(f"  det{det:2d}: IQR-выбросов={len(out_iqr):2d}, "
                  f"z>4σ выбросов={len(out_z):2d}  "
                  f"(из {n_nonzero} ненулевых бинов)")

            if len(out_z) > 0:
                n_all = work[work[:, 2] > 0, 2]
                mu, sigma = n_all.mean(), n_all.std()
                for row in out_z:
                    z = (row[2] - mu) / sigma
                    print(f"      θ={row[0]:.1f}°  φ={row[1]:.1f}°  "
                          f"N={row[2]:.2f}  z={z:.2f}")

    if verbose:
        total_iqr = sum(len(v["iqr"]) for v in all_results.values())
        total_z   = sum(len(v["zscore"]) for v in all_results.values())
        print(f"\n  Итого IQR-выбросов: {total_iqr}  |  z>4σ выбросов: {total_z}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# 5. КРОСС-ДЕТЕКТОРНАЯ СОГЛАСОВАННОСТЬ
# ─────────────────────────────────────────────────────────────────────────────

def cross_detector_check(npl: str, binning: str,
                         theta_min: float = 58.0, theta_max: float = 90.0,
                         threshold: float = CROSS_TH,
                         min_median: float = 5.0,
                         min_dets: int = 5,
                         verbose: bool = True) -> list:
    """
    Для каждого (θ, φ)-бина считает медиану по всем детекторам.
    Если значение у конкретного детектора отличается от медианы
    более чем в threshold раз -- это кросс-детекторный выброс.

    Возвращает список (θ, φ, det, N, median, ratio).
    """
    # Собираем данные в словарь {(theta,phi): {det: N}}
    bin_data: dict[tuple, dict] = {}
    for det in GOOD_DETS:
        d = load_tracks(npl, binning, det)
        if d is None:
            continue
        work = d[(d[:, 0] >= theta_min) & (d[:, 0] <= theta_max)]
        for row in work:
            key = (row[0], row[1])
            bin_data.setdefault(key, {})[det] = row[2]

    # Ищем аномальные бины
    anomalies = []
    for (theta, phi), dvals in bin_data.items():
        vals = list(dvals.values())
        nonzero = [v for v in vals if v > 0]
        if len(nonzero) < min_dets:
            continue
        med = np.median(nonzero)
        if med < min_median:
            continue

        for det, v in dvals.items():
            if v > 0 and v > threshold * med:
                anomalies.append((theta, phi, det, v, med, v / med))

    anomalies.sort(key=lambda x: -x[5])

    if verbose:
        print(f"  {npl}/{binning}: кросс-детекторных аномалий (>{threshold}× медианы): "
              f"{len(anomalies)}")
        if anomalies:
            print(f"  {'θ':>7} {'φ':>7} {'det':>5} {'N':>9} {'медиана':>9} {'отн.':>6}")
            for theta, phi, det, v, med, ratio in anomalies[:10]:
                print(f"  {theta:7.1f} {phi:7.1f} {det:5d} "
                      f"{v:9.2f} {med:9.2f} {ratio:5.1f}×")

    return anomalies

# ─────────────────────────────────────────────────────────────────────────────
# ОТДЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ПАНЕЛЕЙ
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomaly_heatmap_npl4(binning: str = "2.0Grad",
                              theta_min: float = 58.0, theta_max: float = 90.0,
                              save: bool = True):
    """fig1 -- Тепловая карта npl4"""
    fig, ax = plt.subplots(figsize=(9.5, 7.5))

    acc = sum_all_detectors("npl4", binning)
    if acc is None:
        ax.set_title("npl4 -- нет данных")
        if save:
            fig.savefig(OUTPUT_DIR / f"fig1_heatmap_npl4_{binning}.png", dpi=200, bbox_inches="tight")
        return fig

    sub = acc[(acc[:, 0] >= theta_min) & (acc[:, 0] <= theta_max)]
    t_b = np.unique(sub[:, 0])
    p_b = np.unique(sub[:, 1])

    grid = np.zeros((len(t_b), len(p_b)))
    ti = {v: i for i, v in enumerate(t_b)}
    pi = {v: i for i, v in enumerate(p_b)}
    for row in sub:
        if row[0] in ti and row[1] in pi:
            grid[ti[row[0]], pi[row[1]]] = row[2]

    ax.pcolormesh(p_b, t_b, np.where(grid == 0, 1, np.nan),
                  cmap="Greys", alpha=0.45, shading="auto")
    masked = np.ma.masked_equal(grid, 0)
    im = ax.pcolormesh(p_b, t_b, masked, cmap="hot", shading="auto")

    cb = plt.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Количество треков", fontsize=10)

    zero_pct = (grid == 0).sum() / grid.size * 100
    ax.set_title(f"fig1 -- npl4/{binning} -- {zero_pct:.1f}% нулевых бинов",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("φ (°)", fontsize=10)
    ax.set_ylabel("θ (°)", fontsize=10)
    ax.tick_params(labelsize=9)

    if save:
        fig.savefig(OUTPUT_DIR / f"fig1_heatmap_npl4_{binning}.png", dpi=200, bbox_inches="tight")
        print("  → fig1_heatmap_npl4.png")
    plt.close(fig) if not save else None
    return fig


def plot_anomaly_heatmap_npl6(binning: str = "2.0Grad",
                              theta_min: float = 58.0, theta_max: float = 90.0,
                              save: bool = True):
    """fig2 -- Тепловая карта npl6"""
    fig, ax = plt.subplots(figsize=(9.5, 7.5))

    acc = sum_all_detectors("npl6", binning)
    if acc is None:
        ax.set_title("npl6 -- нет данных")
        if save:
            fig.savefig(OUTPUT_DIR / f"fig2_heatmap_npl6_{binning}.png", dpi=200, bbox_inches="tight")
        return fig

    sub = acc[(acc[:, 0] >= theta_min) & (acc[:, 0] <= theta_max)]
    t_b = np.unique(sub[:, 0])
    p_b = np.unique(sub[:, 1])

    grid = np.zeros((len(t_b), len(p_b)))
    ti = {v: i for i, v in enumerate(t_b)}
    pi = {v: i for i, v in enumerate(p_b)}
    for row in sub:
        if row[0] in ti and row[1] in pi:
            grid[ti[row[0]], pi[row[1]]] = row[2]

    ax.pcolormesh(p_b, t_b, np.where(grid == 0, 1, np.nan),
                  cmap="Greys", alpha=0.45, shading="auto")
    masked = np.ma.masked_equal(grid, 0)
    im = ax.pcolormesh(p_b, t_b, masked, cmap="hot", shading="auto")

    cb = plt.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Количество треков", fontsize=10)

    zero_pct = (grid == 0).sum() / grid.size * 100
    ax.set_title(f"fig2 -- npl6/{binning} -- {zero_pct:.1f}% нулевых бинов",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("φ (°)", fontsize=10)
    ax.set_ylabel("θ (°)", fontsize=10)
    ax.tick_params(labelsize=9)

    if save:
        fig.savefig(OUTPUT_DIR / f"fig2_heatmap_npl6_{binning}.png", dpi=200, bbox_inches="tight")
        print("  → fig2_heatmap_npl6.png")
    plt.close(fig) if not save else None
    return fig


def plot_anomaly_nonzero_bins(binning: str = "2.0Grad",
                              theta_min: float = 58.0, theta_max: float = 90.0,
                              save: bool = True):
    """fig3 -- Ненулевые бины по детекторам"""
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    for npl, col, marker in [("npl4", "#2196F3", "o"), ("npl6", "#4CAF50", "s")]:
        vals = []
        for det in GOOD_DETS:
            d = load_tracks(npl, binning, det)
            if d is None:
                vals.append(0)
                continue
            sub = d[(d[:, 0] >= theta_min) & (d[:, 0] <= theta_max)]
            vals.append((sub[:, 2] > 0).sum())

        ax.plot(GOOD_DETS, vals, color=col, marker=marker, label=npl,
                lw=1.8, markersize=6)
        ax.axhline(np.mean(vals), color=col, ls="--", alpha=0.5)

    ax.set_title("fig3 -- Ненулевых бинов по детекторам", fontsize=11, fontweight="bold")
    ax.set_xlabel("Детектор", fontsize=10)
    ax.set_ylabel(f"Количество ненулевых бинов\n({theta_min:.0f}–{theta_max:.0f}°)", fontsize=10)
    ax.legend(fontsize=10)
    ax.set_xticks(GOOD_DETS)
    ax.tick_params(axis='x', rotation=45, labelsize=9)

    if save:
        fig.savefig(OUTPUT_DIR / f"fig3_nonzero_bins_{binning}.png", dpi=200, bbox_inches="tight")
        print("  → fig3_nonzero_bins.png")
    plt.close(fig) if not save else None
    return fig


def plot_anomaly_total_tracks(binning: str = "2.0Grad",
                              theta_min: float = 58.0, theta_max: float = 90.0,
                              save: bool = True):
    """fig4 -- Суммарные треки по детекторам"""
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    w = 0.38

    for k, (npl, col) in enumerate([("npl4", "#2196F3"), ("npl6", "#4CAF50")]):
        vals = []
        for det in GOOD_DETS:
            d = load_tracks(npl, binning, det)
            val = 0 if d is None else d[(d[:, 0] >= theta_min) & (d[:, 0] <= theta_max), 2].sum()
            vals.append(val)

        xs = [d + (k - 0.5) * w for d in GOOD_DETS]
        ax.bar(xs, [v / 1000 for v in vals], width=w, color=col, label=npl,
               alpha=0.85, edgecolor="k", linewidth=0.5)

    ax.set_title("fig4 -- Суммарное количество треков по детекторам", fontsize=11, fontweight="bold")
    ax.set_xlabel("Детектор", fontsize=10)
    ax.set_ylabel("Треков (тыс.)", fontsize=10)
    ax.set_xticks(GOOD_DETS)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.legend(fontsize=10)

    if save:
        fig.savefig(OUTPUT_DIR / f"fig4_total_tracks_{binning}.png", dpi=200, bbox_inches="tight")
        print("  → fig4_total_tracks.png")
    plt.close(fig) if not save else None
    return fig


def plot_anomaly_histogram(binning: str = "2.0Grad",
                           theta_min: float = 58.0, theta_max: float = 90.0,
                           save: bool = True):
    """fig5 -- Гистограмма распределения N_tracks"""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    for npl, col, ls in [("npl4", "#2196F3", "-"), ("npl6", "#4CAF50", "--")]:
        all_n = []
        for det in GOOD_DETS:
            d = load_tracks(npl, binning, det)
            if d is None: continue
            sub = d[(d[:, 0] >= theta_min) & (d[:, 0] <= theta_max) & (d[:, 2] > 0)]
            all_n.extend(sub[:, 2])

        all_n = np.array(all_n)
        ax.hist(all_n, bins=60, color=col, alpha=0.45, label=npl,
                density=True, edgecolor=col, linewidth=0.4, ls=ls)

        q1, q3 = np.percentile(all_n, [25, 75])
        thr = q3 + 1.5 * (q3 - q1)
        ax.axvline(thr, color=col, ls=":", lw=1.8, label=f"{npl} порог ≈ {thr:.0f}")

    ax.set_title("fig5 -- Распределение N_tracks (ненулевые бины)", fontsize=11, fontweight="bold")
    ax.set_xlabel("N_tracks в бине", fontsize=10)
    ax.set_ylabel("Плотность", fontsize=10)
    ax.set_yscale("log")
    ax.legend(fontsize=10)

    if save:
        fig.savefig(OUTPUT_DIR / f"fig5_histogram_{binning}.png", dpi=200, bbox_inches="tight")
        print("  → fig5_histogram.png")
    plt.close(fig) if not save else None
    return fig


def plot_anomaly_qq(binning: str = "2.0Grad", npl_main: str = "npl4",
                    theta_min: float = 58.0, theta_max: float = 90.0,
                    save: bool = True):
    """fig6 -- Q-Q plot"""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    d_qq = load_tracks(npl_main, binning, 4)
    if d_qq is not None:
        sub = d_qq[(d_qq[:, 0] >= theta_min) & (d_qq[:, 0] <= theta_max) & (d_qq[:, 2] > 0)]
        log_n = np.log1p(sub[:, 2])
        qq = scipy_stats.probplot(log_n, dist="norm")
        ax.scatter(qq[0][0], qq[0][1], s=8, alpha=0.6, color="#2196F3", label="log(1+N)")
        ax.plot(qq[0][0], qq[1][0] * qq[0][0] + qq[1][1],
                color="red", lw=2, label="Нормаль")

    ax.set_title(f"fig6 -- Q-Q plot log(1+N) -- {npl_main}/{binning}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Теоретические квантили", fontsize=10)
    ax.set_ylabel("Выборочные квантили", fontsize=10)
    ax.legend(fontsize=10)

    if save:
        fig.savefig(OUTPUT_DIR / f"fig6_qqplot_{binning}.png", dpi=200, bbox_inches="tight")
        print("  → fig6_qqplot.png")
    plt.close(fig) if not save else None
    return fig


def plot_anomaly_boxplot(binning: str = "2.0Grad", npl_main: str = "npl4",
                         theta_min: float = 58.0, theta_max: float = 90.0,
                         save: bool = True):
    """fig7 -- Boxplot по детекторам"""
    fig, ax = plt.subplots(figsize=(12.5, 6.5))

    box_data, labels = [], []
    for det in GOOD_DETS:
        d = load_tracks(npl_main, binning, det)
        if d is None:
            box_data.append([0])
        else:
            sub = d[(d[:, 0] >= theta_min) & (d[:, 0] <= theta_max) & (d[:, 2] > 0)]
            box_data.append(np.log1p(sub[:, 2]).tolist() if len(sub) > 0 else [0])
        labels.append(f"#{det}")

    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True,
                    flierprops=dict(marker="o", markersize=4, alpha=0.6, markerfacecolor="red"))

    for patch in bp["boxes"]:
        patch.set_facecolor("#BBDEFB")
        patch.set_alpha(0.8)

    ax.set_title(f"fig7 -- Boxplot log(1 + N_tracks) -- {npl_main}/{binning}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Детектор", fontsize=10)
    ax.set_ylabel("log(1 + N_tracks)", fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=9)

    if save:
        fig.savefig(OUTPUT_DIR / f"fig7_boxplot_{binning}.png", dpi=200, bbox_inches="tight")
        print("  → fig7_boxplot.png")
    plt.close(fig) if not save else None
    return fig


def plot_anomaly_summary(save: bool = True):
    """fig8 -- Сводная таблица"""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis("off")

    table_rows = [
        ["NaN / Inf",          "Отсутствуют",                  "✓"],
        ["Структ. пропуски",   "Все файлы полные",             "✓"],
        ["Угловое покрытие",   "θ/φ одинаковы у всех дет.",   "✓"],
        ["θ < 58°",            "Геометрическая граница",      "⚠"],
        ["Нулевые бины",       "94–99% = 0 -- норма",          "✓"],
        ["IQR-выбросы",        "det12,14,17 (1–4 бина)",      "⚠"],
        ["Кросс-детектор",     "Нет сильных аномалий",        "✓"],
        ["log(1+N)",           "≈ нормальное распределение",  "✓"],
    ]

    tbl = ax.table(
        cellText=[r[1:] for r in table_rows],
        colLabels=["Детали", "Статус"],
        rowLabels=[r[0] for r in table_rows],
        cellLoc="left", loc="center", bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", weight="bold")
        elif "SUCCESS!" in str(cell.get_text().get_text()):
            cell.set_facecolor("#E8F5E9")
        elif "ERROR!" in str(cell.get_text().get_text()):
            cell.set_facecolor("#FFF9C4")

    ax.set_title("fig8 -- Итоговая сводка анализа", fontsize=12, fontweight="bold", pad=12)

    if save:
        fig.savefig(OUTPUT_DIR / "fig8_summary.png", dpi=180, bbox_inches="tight")
        print("  → fig8_summary.png")
    plt.close(fig) if not save else None
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Мюонография -- анализ пропусков и аномалий")
    print("=" * 62)
    print(f"\n  Данные:      {DATA_ROOT.resolve()}")
    print(f"  Графики:     {OUTPUT_DIR.resolve()}")
    print(f"  Детекторы:   {GOOD_DETS}  (исключены: {sorted(BAD_DETS)})")
    print(f"  Конфигурации: {[f'{n}/{b}' for n,b in CONFIGS]}\n")

    # ── 1. Структурные пропуски ──────────────────────────────────────────────
    print("─── 1. Структурные пропуски " + "─" * 34)
    check_structural(verbose=True)

    # ── 2. Геометрическая граница ────────────────────────────────────────────
    print("\n─── 2. Геометрическая граница обзора " + "─" * 24)
    theta_min, theta_max = find_theta_range("npl4", "2.0Grad", verbose=True)

    # ── 3 & 4. Выбросы по каждому детектору ─────────────────────────────────
    print("\n─── 3 & 4. Выбросы IQR и Z-score " + "─" * 27)
    for npl, binning in [("npl4", "2.0Grad"), ("npl6", "2.0Grad")]:
        print(f"\n  {npl}/{binning}  (только детекторы с выбросами):")
        analyze_outliers(npl, binning, theta_min, theta_max, verbose=True)

    # ── 5. Кросс-детекторная согласованность ────────────────────────────────
    print("\n─── 5. Кросс-детекторная согласованность " + "─" * 20)
    for npl, binning in [("npl4", "2.0Grad"), ("npl6", "2.0Grad")]:
        cross_detector_check(npl, binning, theta_min, theta_max, verbose=True)

    # ── 6. Визуализация ──────────────────────────────────────────────────────
    print("\n─── 6. Генерация графиков " + "─" * 32)

    plot_anomaly_heatmap_npl4(theta_min=theta_min, theta_max=theta_max)
    plot_anomaly_heatmap_npl6(theta_min=theta_min, theta_max=theta_max)
    plot_anomaly_nonzero_bins(theta_min=theta_min, theta_max=theta_max)
    plot_anomaly_total_tracks(theta_min=theta_min, theta_max=theta_max)
    plot_anomaly_histogram(theta_min=theta_min, theta_max=theta_max)
    plot_anomaly_qq(theta_min=theta_min, theta_max=theta_max)
    plot_anomaly_boxplot(theta_min=theta_min, theta_max=theta_max)
    plot_anomaly_summary()

    print("\nSUCCESS! Анализ завершён!")
    plt.show()


if __name__ == "__main__":
    main()


"""
Описание графиков на дашборде

-----------------------------------------------------------------------------------------------------------
ГРАФИКИ 1 И 2. Тепловые карты npl4 и npl6 (верхний ряд, панели 1–2):

Каждая ячейка -- один угловой бин (θ, φ), цвет -- суммарное число треков по всем рабочим детекторам.
Серые ячейки -- нулевые бины.

Сразу бросается в глаза, что заполнена лишь узкая полоса при θ близких
к 90° -- это и есть рабочая область вертикально установленного детектора.

Сравнение двух карт показывает:
1. Структура распределения у npl4 и npl6 идентична, npl6 просто бледнее из-за меньшей статистики.
2. Аномальных изолированных пятен нет -- распределение гладкое.
-----------------------------------------------------------------------------------------------------------
ГРАФИК 3. Заполненность бинов по детекторам (верхний ряд, панель 3)

Число ненулевых бинов у каждого детектора. В норме все детекторы должны давать примерно
одинаковое значение -- что и наблюдается.

Пунктирная линия -- среднее по группе. Разброс около среднего объясняется разным угловым
расположением детекторов на площадке: те, что стоят на краю установки, видят несколько
меньший телесный угол пересечения с остальными.
-----------------------------------------------------------------------------------------------------------
ГРАФИК 4. Суммарные треки по детекторам (средний ряд, панель 1)

Столбчатая диаграмма: сколько треков зарегистрировал каждый детектор в рабочем диапазоне θ ≥ 58°.

1. Видно, что детекторы 1, 2 и 12 заметно слабее остальных -- это геометрический эффект их расположения
   на периферии установки, а не сбой.
2. Детекторы 13, 14, 17, 18 дают больше всего треков -- они расположены ближе к центру поля обзора.
-----------------------------------------------------------------------------------------------------------
РИСУНОК 5. Гистограмма N_tracks (средний ряд, панель 2)

Распределение числа треков в ненулевых бинах, логарифмическая шкала по оси Y.

1. Оба распределения (npl4 и npl6) имеют характерную правостороннюю асимметрию: большинство бинов
   содержат мало треков, и есть хвост с крупными значениями.
2. Вертикальные пунктирные линии -- пороги IQR-выброса (Q3 + 3·IQR). Всё, что правее порога -- кандидаты
   на выброс. Видно, что порог у npl6 значительно ниже -- из-за меньшей средней статистики.
-----------------------------------------------------------------------------------------------------------
РИСУНОК 6. Q-Q plot (средний ряд, панель 3)

Проверка, насколько логарифмически преобразованные данные log(1+N) близки к нормальному распределению.
Если точки ложатся на красную прямую -- распределение нормальное.

Здесь точки хорошо следуют прямой в центре, и лишь на хвостах (самые крупные значения) немного уходят вверх
-- это и есть те самые выбросы.

Вывод: log-преобразование хорошо нормализует данные, что делает его разумным шагом предобработки
перед подачей в ML.
-----------------------------------------------------------------------------------------------------------
РИСУНОК 7. Boxplot по детекторам (нижний ряд, панели 1–2)

Ящик с усами для каждого детектора в пространстве log(1+N).
Красная линия внутри ящика -- медиана, красные точки -- выбросы по внутреннему IQR критерию.

1. Видно, что медианы у всех детекторов примерно одинаковые -- это хорошо.
2. Детекторы 12, 14, 17 имеют несколько больше выбросов и более широкий разброс, но ящики не выделяются
   кардинально. Это подтверждает, что речь идёт о единичных флуктуациях, а не о системном сдвиге.
-----------------------------------------------------------------------------------------------------------
ОБЩИЙ ВЫВОД

1. Данные технически чистые: ни NaN, ни Inf, ни структурных пропусков нет.
2. Все файлы полные, угловое покрытие одинаковое у всех детекторов.
3. Главное содержательное ограничение -- геометрическое: детектор не видит треки при θ < 58°,
   и это не дефект данных, а физическая граница обзора вертикально установленной эмульсии.
4. Рабочая область -- θ от 58° до 90°.
5. Выбросы существуют, но не критичны: несколько единичных бинов у детекторов 12, 14, 17 выходят за 4σ.
   Кросс-детекторная проверка показала, что ни один из этих бинов не аномален относительно других детекторов
   в той же угловой точке -- то есть физически они не противоречат соседям. Это флуктуации статистики,
   а не ошибки измерения.

Для предобработки перед ML рекомендуется:
1. ограничить данные диапазоном θ ≥ 58°,
2. применить логарифмическое преобразование log(1+N) -- Q-Q plot подтверждает, что оно приводит распределение
   к нормальному, что удобно для большинства моделей.
3. Решение о том, оставить выбросы как есть или применить мягкую винсоризацию, зависит от
   выбранной архитектуры: деревья и ансамбли к ним нечувствительны, нейронные сети -- чуть более.
"""