"""
=============================================================
  Мюонография -- сглаживание и шумоподавление
=============================================================

Реализация трёх базовых методов сглаживания угловых распределений
мюонных треков. Методы используются как базовые линии (baseline)
для сравнения с U-Net моделью по метрикам MSE и SSIM.

Методы:
  1. gaussian   -- гауссовский фильтр (scipy.ndimage.gaussian_filter)
  2. gradient   -- градиентно-взвешенный фильтр (адаптивный, нелинейный)
  3. wavelet    -- вейвлет-сглаживание BayesShrink (skimage.restoration)

Источник методов:
  Мартьянова О.А. «Методы сглаживания угловых распределений
  в задачах мюонографии», МИФИ, 2026.

Зависимости:
  pip install numpy matplotlib scipy scikit-image
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Literal
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_wavelet

from config import (
    OUTPUT_DIR,
    GAUSS_SIGMA_THETA, GAUSS_SIGMA_PHI,
    GRAD_WINDOW, WAVELET_LEVELS, WAVELET_TYPE,
)
from muon_preprocessing import (
    load_preprocessed, inverse_transform,
)


# Папка для сохранения графиков
OUTPUT_DIR = OUTPUT_DIR / "muon_smoothing"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# МЕТОД 1: ГАУССОВСКИЙ ФИЛЬТР
# ─────────────────────────────────────────────────────────────────────────────

def smooth_gaussian(
    grid: np.ndarray,
    sigma_theta: float = GAUSS_SIGMA_THETA,
    sigma_phi:   float = GAUSS_SIGMA_PHI,
) -> np.ndarray:
    """
    Двумерная гауссовская свёртка угловой сетки N(θ, φ).

    Математически:
        N_smooth(θ, φ) = N(θ, φ) ∗ G(Δθ, Δφ)

        G(Δθ, Δφ) = 1/(2π σ_θ σ_φ) · exp(−Δθ²/2σ_θ² − Δφ²/2σ_φ²)

    :param grid: 2D-массив (T × P), θ по строкам, φ по столбцам
    :param sigma_theta: ширина ядра по θ (в единицах бинов)
    :param sigma_phi: ширина ядра по φ (в единицах бинов)

    Физический смысл sigma:
        σ = 0.5  ->  слабое сглаживание, скачки между соседними бинами сглажены слабо
        σ = 1.0  ->  умеренное, оптимально для биннинга 1.5°–2.0°
        σ = 1.5  ->  сильное, физические детали начинают размываться

    Анизотропный вариант (sigma_theta ≠ sigma_phi):
        Если скачки сильнее по θ (наш случай -- cos²θ тренд),
        рекомендуется sigma_theta > sigma_phi.
        По результатам Мартьяновой О.А.: sigma_theta=1.0, sigma_phi=0.5.

    Нулевые бины не маскируются: сглаживание размазывает нули по соседним
    ячейкам, что фактически реализует интерполяцию в пустые области.

    :return: Сглаженный 2D-массив
    """
    return gaussian_filter(grid.astype(float), sigma=(sigma_theta, sigma_phi))


# ─────────────────────────────────────────────────────────────────────────────
# МЕТОД 2: ГРАДИЕНТНО-ВЗВЕШЕННЫЙ ФИЛЬТР
# ─────────────────────────────────────────────────────────────────────────────

def smooth_gradient(
    grid:        np.ndarray,
    window_size: int = GRAD_WINDOW,
) -> np.ndarray:
    """
    Адаптивный нелинейный фильтр на основе модуля градиента.

    Идея: там, где данные однородны (малый градиент) -- сглаживать сильно;
    там, где есть резкий переход (граница полости) -- не сглаживать.

    Алгоритм:
        1. Вычислить градиент: ∇N = (∂N/∂θ, ∂N/∂φ), |∇N| = √(...)
        2. Нормировать: g(θ,φ) = |∇N| / max|∇N|
        3. Весовая функция: w(θ,φ) = exp(−g²(θ,φ))
           -> 1 в однородных зонах, ≈0 на границах
        4. Локальное взвешенное среднее в окне window_size × window_size:
           Ñ_ij = Σ N_kl·w_kl / Σ w_kl


    :param grid: 2D-массив (T × P)
    :param window_size: размер окна усреднения (3 или 5, нечётное)

    Преимущество перед гауссом: сохраняет локальные экстремумы
    и границы физических структур, не размывая их.

    :return: Сглаженный 2D-массив
    """
    grid = grid.astype(float)
    half = window_size // 2

    grad_theta, grad_phi = np.gradient(grid)
    grad_mag  = np.sqrt(grad_theta ** 2 + grad_phi ** 2)
    max_grad  = grad_mag.max()
    grad_norm = grad_mag / max_grad if max_grad > 0 else grad_mag

    weights = np.exp(-grad_norm ** 2)

    # Дополнение крайними значениями для корректной обработки границ
    grid_pad    = np.pad(grid,    half, mode='edge')
    weights_pad = np.pad(weights, half, mode='edge')

    result = np.zeros_like(grid)
    T, P = grid.shape
    for i in range(T):
        for j in range(P):
            patch_g = grid_pad[i:i + window_size, j:j + window_size]
            patch_w = weights_pad[i:i + window_size, j:j + window_size]
            w_sum   = patch_w.sum()
            result[i, j] = (patch_g * patch_w).sum() / w_sum if w_sum > 0 else grid[i, j]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# МЕТОД 3: ВЕЙВЛЕТ-СГЛАЖИВАНИЕ (BayesShrink)
# ─────────────────────────────────────────────────────────────────────────────

def smooth_wavelet(
    grid:         np.ndarray,
    wavelet:      str = WAVELET_TYPE,
    levels:       int = WAVELET_LEVELS,
    method:       str = "BayesShrink",
) -> np.ndarray:
    """
    Адаптивное вейвлет-сглаживание с пороговым преобразованием BayesShrink.

    Математика:
        1. Двумерное DWT уровня L:
           N(θ,φ) = A_L + Σ_{l=1}^L [H_l + V_l + D_l]
           где A_L -- аппроксимация (крупный масштаб),
           H_l, V_l, D_l -- горизонтальные/вертикальные/диагональные детали.

        2. BayesShrink для каждого уровня и поддиапазона:
           T_bayes = σ²_noise / σ_signal
           Мягкое пороговое преобразование: sgn(w)·max(|w|−T, 0)

        3. Обратное DWT -> сглаженное распределение.


    :param grid: 2D-массив (T × P)
    :param wavelet: тип вейвлета ('db1' = Добеши-1, Хаар)
    :param levels: число уровней разложения:
                    1 -> убирает только мелкий шум (высокие частоты)
                    4 -> убирает шум на нескольких масштабах (рекомендуется)
    :param method: 'BayesShrink' (адаптивный) или 'VisuShrink' (фиксированный)

    Преимущество: автоматически оценивает дисперсию шума на каждом уровне,
    не требует ручного задания порога.

    :return: Сглаженный 2D-массив
    """
    grid_f = grid.astype(float)

    # skimage требует данные в [0, 1] для корректной оценки σ_noise
    g_min, g_max = grid_f.min(), grid_f.max()
    if g_max > g_min:
        grid_norm = (grid_f - g_min) / (g_max - g_min)
    else:
        return grid_f.copy()

    smoothed_norm = denoise_wavelet(
        grid_norm,
        method=method,
        mode='soft',
        wavelet_levels=levels,
        wavelet=wavelet,
        rescale_sigma=True,
    )

    # Возвращаем в исходный масштаб
    return smoothed_norm * (g_max - g_min) + g_min


# ─────────────────────────────────────────────────────────────────────────────
# ЛИНЕЙНОЕ ВЕЙВЛЕТ-СГЛАЖИВАНИЕ (обнуление деталей)
# ─────────────────────────────────────────────────────────────────────────────

def smooth_wavelet_linear(
    grid:    np.ndarray,
    wavelet: str = WAVELET_TYPE,
    levels:  int = 1,
) -> np.ndarray:
    """
    Линейное вейвлет-сглаживание: оставить только аппроксимацию A_L,
    обнулив все детальные коэффициенты.

    Ñ(θ,φ) = A_L(θ,φ)

    Это самое грубое сглаживание -- теряет много деталей.
    Используется как нижняя граница качества в сравнении с BayesShrink.
    """
    import pywt
    T, P = grid.shape
    coeffs = pywt.wavedec2(grid.astype(float), wavelet=wavelet, level=levels)

    # Обнуляем все детали, оставляем только аппроксимацию
    coeffs_zero = [coeffs[0]] + [
        tuple(np.zeros_like(d) for d in level_detail)
        for level_detail in coeffs[1:]
    ]
    result = pywt.waverec2(coeffs_zero, wavelet=wavelet)
    return result[:T, :P]


# ─────────────────────────────────────────────────────────────────────────────
# УНИФИЦИРОВАННЫЙ ИНТЕРФЕЙС
# ─────────────────────────────────────────────────────────────────────────────

SmoothMethod = Literal["gaussian", "gradient", "wavelet", "wavelet_linear"]

def smooth(
    grid:   np.ndarray,
    method: SmoothMethod = "wavelet",
    **kwargs,
) -> np.ndarray:
    """
    Единая точка входа для всех методов сглаживания.

    :param grid: 2D-массив (T × P) одного детектора
    :param method: метод сглаживания:
                'gaussian'       -- гауссовский фильтр
                'gradient'       -- градиентно-взвешенный фильтр
                'wavelet'        -- BayesShrink (рекомендуется)
                'wavelet_linear' -- обнуление деталей (грубо)
    :param kwargs: параметры конкретного метода (sigma_theta, levels и т.д.)
    :return: Сглаженный 2D-массив
    """
    if method == "gaussian":
        return smooth_gaussian(grid, **kwargs)
    elif method == "gradient":
        return smooth_gradient(grid, **kwargs)
    elif method == "wavelet":
        return smooth_wavelet(grid, **kwargs)
    elif method == "wavelet_linear":
        return smooth_wavelet_linear(grid, **kwargs)
    else:
        raise ValueError(
            f"Неизвестный метод: '{method}'. "
            f"Доступны: gaussian, gradient, wavelet, wavelet_linear"
        )


def smooth_all_detectors(
    grids:  np.ndarray,
    method: SmoothMethod = "wavelet",
    **kwargs,
) -> np.ndarray:
    """
    Применяет сглаживание ко всем детекторам тензора.

    :param grids: тензор (D × T × P), D детекторов
    :param method: метод сглаживания
    :param kwargs: параметры метода
    :return: Сглаженный 2D-массив

    Возвращает тензор той же формы (D × T × P).
    """
    result = np.zeros_like(grids)
    for d in range(grids.shape[0]):
        result[d] = smooth(grids[d], method=method, **kwargs)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# МЕТРИКИ КАЧЕСТВА СГЛАЖИВАНИЯ
# ─────────────────────────────────────────────────────────────────────────────

def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Среднеквадратичная ошибка между двумя массивами."""
    return float(np.mean((a - b) ** 2))


def ssim(
    a: np.ndarray,
    b: np.ndarray,
    data_range: float | None = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Структурное сходство (SSIM) -- метрика сохранения структуры.

    SSIM ∈ [−1, 1], чем ближе к 1 -- тем лучше сохранена структура.

    :param a: первый сравниваемый массивы
    :param b: второй сравниваемый массивы
    :param data_range: диапазон значений (по умолчанию max−min по a)
    :param k1: первая стабилизирующая константы (стандарт: 0.01)
    :param k2: вторая стабилизирующая константы (стандарт: 0.03)
    :return: Значение метрики ∈ [−1, 1]
    """
    if data_range is None:
        data_range = float(a.max() - a.min()) or 1.0

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_a, mu_b = a.mean(), b.mean()
    sig_a  = a.std()
    sig_b  = b.std()
    sig_ab = float(np.mean((a - mu_a) * (b - mu_b)))

    num = (2 * mu_a * mu_b + c1) * (2 * sig_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (sig_a ** 2 + sig_b ** 2 + c2)
    return float(num / den) if den > 0 else 0.0


def compare_methods(
    grid_raw: np.ndarray,
    configs: list[dict] | None = None,
) -> dict:
    """
    Применяет все методы к одному детектору и считает метрики
    относительно сырого grid_raw (как «эталона» структуры).

    :param grid_raw: исходная сетка (T × P) до сглаживания
    :param configs: список словарей {'method': ..., **kwargs}.
               None -> используются рекомендованные параметры.
    :return: dict {name: {'grid': ..., 'mse': ..., 'ssim': ...}}

    """
    if configs is None:
        configs = [
            {"method": "gaussian",  "sigma_theta": 1.0, "sigma_phi": 0.5},
            {"method": "gaussian",  "sigma_theta": 0.5, "sigma_phi": 1.0,
             "_name": "gaussian_aniso"},
            {"method": "gradient",  "window_size": 3},
            {"method": "wavelet",   "levels": 4},
            {"method": "wavelet_linear", "levels": 1,
             "_name": "wavelet_linear"},
        ]

    results = {}
    for cfg in configs:
        cfg = cfg.copy()
        name = cfg.pop("_name", cfg["method"])
        method = cfg.pop("method")
        smoothed = smooth(grid_raw, method=method, **cfg)
        results[name] = {
            "grid":  smoothed,
            "mse":   mse(grid_raw, smoothed),
            "ssim":  ssim(grid_raw, smoothed),
            "method": method,
            "params": cfg,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def plot_smoothing_comparison(
    grid_raw: np.ndarray,
    theta:    np.ndarray,
    phi:      np.ndarray,
    det:      int,
    npl:      str = "npl4",
    binning:  str = "2.0Grad",
    save:     bool = True,
) -> plt.Figure:
    """
    Сравнительный дашборд: исходные данные + 4 метода сглаживания.

    Каждая панель -- тепловая карта N(θ, φ) + MSE и SSIM в подписи.
    """
    configs = [
        {"method": "gaussian",  "sigma_theta": 1.0, "sigma_phi": 0.5},
        {"method": "gradient",  "window_size": 3},
        {"method": "wavelet",   "levels": 4},
        {"method": "wavelet_linear", "levels": 1, "_name": "wavelet_linear"},
    ]
    results = compare_methods(grid_raw, configs)

    fig = plt.figure(figsize=(22, 15))
    fig.suptitle(
        f"Сравнение методов сглаживания -- дет.№{det}, {npl}/{binning}",
        fontsize=12, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4,
                            height_ratios=[1.2, 1, 1])

    panels = [("Исходные данные", grid_raw, None, None)] + [
        (name, v["grid"], v["mse"], v["ssim"])
        for name, v in results.items()
    ]

    labels = {
        "Исходные данные": "Исходные\nданные",
        "gaussian":        f"Гаусс\nσ_θ=1.0, σ_φ=0.5",
        "gradient":        "Градиентный\nфильтр (3×3)",
        "wavelet":         f"Вейвлет\nBayesShrink L=4",
        "wavelet_linear":  "Вейвлет\nлинейный L=1",
    }

    positions = [
        (0, slice(1, 3)),  # исходные данные
        (1, slice(0, 2)), (1, slice(2, 4)),  # метод 1, 2
        (2, slice(0, 2)), (2, slice(2, 4)),  # метод 3, 4
    ]

    for idx, (name, grid, err_mse, err_ssim) in enumerate(panels):
        ax = fig.add_subplot(gs[positions[idx]])

        # row, col = positions[idx]
        # ax = fig.add_subplot(gs[row, col])
        im = ax.pcolormesh(phi, theta, grid, cmap="inferno", shading="auto")
        cbar = plt.colorbar(im, ax=ax, shrink=0.78, label="N")
        cbar.ax.tick_params(labelsize=7)
        ax.set_xlabel("φ (°)", fontsize=8)
        ax.set_ylabel("θ (°)", fontsize=8)
        ax.tick_params(labelsize=7)

        title = labels.get(name, name)
        if err_mse is not None:
            title += f"\nMSE(10^8)={err_mse*10**8:.4f}  SSIM={err_ssim:.3f}"        # type:ignore
        ax.set_title(title, fontsize=8.5, fontweight="bold")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if save:
        out = OUTPUT_DIR / f"fig_smoothing_det{det}_{npl}_{binning}.png"
        fig.savefig(out, bbox_inches="tight", dpi=130)
        print(f"  Сохранён: {out.name}")
    return fig


def plot_metrics_summary(
    grids:    np.ndarray,
    dets:     np.ndarray,
    npl:      str = "npl4",
    binning:  str = "2.0Grad",
    save:     bool = True,
) -> plt.Figure:
    """
    Сводный график MSE и SSIM для всех детекторов и всех методов.

    Показывает насколько каждый метод изменяет структуру данных
    относительно исходного распределения.
    """
    method_names = ["gaussian", "gradient", "wavelet", "wavelet_linear"]
    method_labels = {
        "gaussian":       "Гаусс (σ=1.0/0.5)",
        "gradient":       "Градиент (3×3)",
        "wavelet":        "Wavelet BayesShrink L=4",
        "wavelet_linear": "Wavelet linear L=1",
    }
    colors = {"gaussian": "#2196F3", "gradient": "#4CAF50",
              "wavelet": "#FF9800", "wavelet_linear": "#9C27B0"}

    # Считаем метрики для каждого детектора и метода
    metrics = {m: {"mse": [], "ssim": []} for m in method_names}

    for d_idx in range(grids.shape[0]):
        grid = grids[d_idx]
        results = compare_methods(grid)
        for name in method_names:
            if name in results:
                metrics[name]["mse"].append(results[name]["mse"])
                metrics[name]["ssim"].append(results[name]["ssim"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Метрики сглаживания по детекторам -- {npl}/{binning}",
        fontsize=13, fontweight="bold",
    )
    x = np.arange(len(dets))

    for method in method_names:
        mse_vals  = metrics[method]["mse"]
        ssim_vals = metrics[method]["ssim"]
        if not mse_vals:
            continue
        lbl = method_labels[method]
        col = colors[method]
        axes[0].plot(x, mse_vals,  "o-", color=col, label=lbl, lw=1.5, ms=5)
        axes[1].plot(x, ssim_vals, "o-", color=col, label=lbl, lw=1.5, ms=5)

    for ax, title, ylabel in zip(
        axes,
        ["MSE (меньше = ближе к исходным)", "SSIM (больше = лучше сохранена структура)"],
        ["MSE", "SSIM"],
    ):
        ax.set_xticks(x)
        ax.set_xticklabels([f"#{d}" for d in dets], fontsize=8)
        ax.set_xlabel("Детектор")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save:
        out = OUTPUT_DIR / f"fig_smoothing_metrics_{npl}_{binning}.png"
        fig.savefig(out, bbox_inches="tight", dpi=130)
        print(f"  Сохранён: {out.name}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def main(
    npl:     str = "npl4",
    binning: str = "2.0Grad",
    method:  SmoothMethod | None = None,
    det_idx: int = 3,
    save_plots: bool = True,
    sigma_theta: float = GAUSS_SIGMA_THETA,
    sigma_phi:   float = GAUSS_SIGMA_PHI,
    wavelet_levels: int = WAVELET_LEVELS,
) -> np.ndarray:
    """
    Запуск сглаживания и построение графиков.

    :param npl:             уровень качества трека
    :param binning:         угловой биннинг
    :param method:          метод сглаживания (None = все + сравнение)
    :param det_idx:         индекс детектора для детального графика (0–14)
    :param save_plots:      сохранять ли графики
    :param sigma_theta:     σ по θ для гауссовского фильтра
    :param sigma_phi:       σ по φ для гауссовского фильтра
    :param wavelet_levels:  уровень разложения для BayesShrink
    :return: сглаженные данные
    """
    print("=" * 60)
    print("  Мюонография -- сглаживание данных")
    print("=" * 60)
    print(f"\n  Конфигурация : {npl} / {binning}")

    # Загружаем предобработанные данные
    try:
        data = load_preprocessed(npl, binning)
    except FileNotFoundError:
        print("  Предобработанные данные не найдены -- запускаем предобработку...")
        from muon_preprocessing import main as preproc_main
        preproc_main(npl=npl, binning=binning, save_plots=False)
        data = load_preprocessed(npl, binning)

    grids     = data["grids"]       # (D, T, P) -- в log1p пространстве
    theta     = data["theta"]
    phi       = data["phi"]
    detectors = data["detectors"]
    transform = str(data["transform_mode"])

    print(f"  Загружено    : {grids.shape} (детекторов × θ-бинов × φ-бинов)")
    print(f"  Трансформация: {transform}")

    # Для сглаживания работаем в исходном масштабе N_tracks
    grids_raw = inverse_transform(grids, mode=transform)

    if method is None:
        # ── Режим сравнения всех методов ────────────────────────────────────
        print(f"\n─── Сравнение методов (дет.№{detectors[det_idx]}) ───")
        det_grid = grids_raw[det_idx]
        results  = compare_methods(det_grid)

        print(f"\n{'Метод':22s}  {'MSE (10^8)':>10s}  {'SSIM':>8s}")
        print("─" * 46)
        for name, r in results.items():
            print(f"  {name:20s}  {r['mse']*100000000:10.5f}  {r['ssim']:8.4f}")

        if save_plots:
            print("\n─── Визуализация ───")
            plot_smoothing_comparison(
                det_grid, theta, phi,
                det=detectors[det_idx],
                npl=npl, binning=binning, save=True,
            )
            plot_metrics_summary(
                grids_raw, detectors,
                npl=npl, binning=binning, save=True,
            )

    else:
        # ── Режим одного метода ──────────────────────────────────────────────
        print(f"\n─── Метод: {method} ───")
        kwargs = {}
        if method == "gaussian":
            kwargs = {"sigma_theta": sigma_theta, "sigma_phi": sigma_phi}
        elif method == "wavelet":
            kwargs = {"levels": wavelet_levels}

        smoothed_grids = smooth_all_detectors(grids_raw, method=method, **kwargs)
        print(f"  Входной тензор  : {grids_raw.shape}")
        print(f"  Выходной тензор : {smoothed_grids.shape}")

        # Пример метрики
        for d in range(min(3, len(detectors))):
            m = mse(grids_raw[d], smoothed_grids[d])
            s = ssim(grids_raw[d], smoothed_grids[d])
            print(f"  дет.#{detectors[d]:2d}: MSE={m}  SSIM={s:.4f}")

        if save_plots:
            plot_smoothing_comparison(
                grids_raw[det_idx], theta, phi,
                det=detectors[det_idx],
                npl=npl, binning=binning, save=True,
            )

    print("\n✓ Готово!")
    print(f"  Графики: {OUTPUT_DIR}")

    plt.show()
    return grids_raw


if __name__ == "__main__":
    main(
        npl="npl4",
        binning="2.0Grad",
        method=None,       # None = сравнение всех методов
        det_idx=3,         # детектор №4 (индекс 3)
        save_plots=True,
    )


"""
=============================================================================
СПРАВОЧНИК ПО МЕТОДАМ СГЛАЖИВАНИЯ
=============================================================================

─────────────────────────────────────────────────────────────────────────────
ГАУССОВСКИЙ ФИЛЬТР
─────────────────────────────────────────────────────────────────────────────

  Линейный метод. Каждый выходной бин = взвешенная сумма соседей с весами
  по нормальному закону. Прост, быстр, но одинаково сглаживает везде --
  в том числе на физических границах.

  sigma (в единицах бинов):
    0.5  -- слабое сглаживание, мелкие скачки сохраняются
    1.0  -- умеренное, оптимально для биннинга 2.0°
    1.5  -- сильное, физические детали начинают теряться

  Анизотропный режим (sigma_theta ≠ sigma_phi):
    Рекомендуется sigma_theta > sigma_phi, если перепады сильнее по θ.
    Оптимум (Мартьянова О.А., биннинг 1.5°): sigma_theta=1.0, sigma_phi=0.5

─────────────────────────────────────────────────────────────────────────────
ГРАДИЕНТНО-ВЗВЕШЕННЫЙ ФИЛЬТР
─────────────────────────────────────────────────────────────────────────────

  Нелинейный адаптивный метод. Вес каждого соседнего бина обратно
  пропорционален градиенту данных: w = exp(−g²).
  Там, где данные резко меняются (граница полости), сглаживание
  автоматически отключается. В однородных зонах -- обычное усреднение.

  window_size:
    3 -- окно 3×3, стандарт. Локальное, быстрое.
    5 -- окно 5×5, более сильное сглаживание однородных зон.

  Физический смысл: аналог анизотропной диффузии Перона-Малика,
  но в дискретном нелинейном варианте.

─────────────────────────────────────────────────────────────────────────────
ВЕЙВЛЕТ BAYESSHRINK
─────────────────────────────────────────────────────────────────────────────

  Нелинейный многомасштабный метод. Разлагает данные на аппроксимацию
  и детали нескольких уровней; применяет адаптивный мягкий порог к деталям:
      T_bayes = σ²_noise / σ_signal
  Порог вычисляется автоматически для каждого уровня и поддиапазона.

  wavelet_levels:
    1 -- убирает только самый высокочастотный шум (мало эффекта)
    2 -- умеренное подавление
    4 -- рекомендуется: хорошо убирает шум, сохраняет структуру
    6+ -- может начать стирать физически значимые пики

  wavelet = 'db1' (Добеши-1 = Хаар): самый простой и быстрый.
  Можно попробовать 'db4', 'sym4' для более гладкого результата.

─────────────────────────────────────────────────────────────────────────────
РОЛЬ МЕТОДОВ В ПАЙПЛАЙНЕ
─────────────────────────────────────────────────────────────────────────────

  Предобработка -> Сглаживание (этот файл) -> U-Net

  Классические методы служат базовыми линиями (baseline):
  U-Net должна превзойти их по MSE и SSIM, чтобы работа имела смысл.

  Типичные ожидания:
    Гаусс      MSE ≈ средний,  SSIM ≈ средний  (теряет границы)
    Градиент   MSE ≈ низкий,   SSIM ≈ высокий  (сохраняет границы)
    BayesShrink MSE ≈ низкий,  SSIM ≈ высокий  (лучший классический)
    U-Net      ожидается лучший результат по обеим метрикам

=============================================================================
"""