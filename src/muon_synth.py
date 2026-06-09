"""
=============================================================
  Мюонография -- генератор синтетических данных
=============================================================

Генерирует синтетические пары (зашумлённый вход, чистый GT)
для предобучения U-Net до файнтюнинга на реальных данных.

Физическая модель:
  N*(θ, φ) = A · cos²θ · R(φ) + Σ_k G_k(θ, φ)

  где:
    cos²θ        -- теоретический профиль интенсивности мюонов
    R(φ)         -- медленная азимутальная вариация (геология)
    G_k(θ, φ)    -- гауссовские пики (полости, аномалии)
                   размещаются при θ < 70° (индексы 0–5),
                   где мюоны ещё почти горизонтальны

  Зашумлённый вход:
    N(θ, φ) ~ Poisson(N*(θ, φ) / scale) · scale

Зависимости:
  pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass

from config import (
    DATA_ROOT, OUTPUT_DIR,
    THETA_MIN, THETA_MAX,
    SYNTH_N_THETA, SYNTH_N_PHI,
    SYNTH_INTENSITY_RANGE, SYNTH_N_ANOMALIES_RANGE,
    SYNTH_NOISE_SCALE_RANGE, SYNTH_AZIMUTH_VAR_AMP,
    SYNTH_ANOMALY_AMP_RANGE, SYNTH_ANOMALY_SIGMA_THETA,
    SYNTH_ANOMALY_SIGMA_PHI, SYNTH_ANOMALY_THETA_IDX_RANGE,
)

# Псевдонимы для удобства внутри модуля
N_THETA = SYNTH_N_THETA
N_PHI = SYNTH_N_PHI
INTENSITY_RANGE = SYNTH_INTENSITY_RANGE
N_ANOMALIES_RANGE = SYNTH_N_ANOMALIES_RANGE
NOISE_SCALE_RANGE = SYNTH_NOISE_SCALE_RANGE
AZIMUTH_VAR_AMP = SYNTH_AZIMUTH_VAR_AMP
ANOMALY_AMP_RANGE = SYNTH_ANOMALY_AMP_RANGE
ANOMALY_SIGMA_THETA = SYNTH_ANOMALY_SIGMA_THETA
ANOMALY_SIGMA_PHI = SYNTH_ANOMALY_SIGMA_PHI


# Куда сохранять сгенерированные данные
SYNTH_DIR = DATA_ROOT / "synthetic"

# Папка для сохранения графиков
OUTPUT_DIR = OUTPUT_DIR / "muon_synthesis"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ФИЗИЧЕСКАЯ МОДЕЛЬ
# ─────────────────────────────────────────────────────────────────────────────

def make_theta_profile(
    theta_bins: np.ndarray,
    power: float = 2.0,
    jitter: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    θ-профиль интенсивности: cos²(θ) с нижним флором.

    Из реальных данных: максимум интенсивности при θ≈60°, спад к θ=90°.
    Это соответствует cos²(θ) из физики мюонного потока.

    Флор 0.2 гарантирует ненулевые значения даже при θ->90°,
    что предотвращает Пуассоновские нули внутри конуса.

    :param theta_bins: массив θ-значений в градусах, shape (T,)
    :param power: показатель степени (2.0 = стандартный)
    :param jitter: случайное отклонение power (вариация между образцами)
    :param rng: генератор случайных чисел
    :return: θ-профиль мюонного потока
    """
    if rng is None:
        rng = np.random.default_rng()

    p = power + rng.uniform(-jitter, jitter)
    profile = np.cos(np.radians(theta_bins)) ** p
    profile = np.clip(profile, 0, None)

    # Нижний флор: даже при θ=90° значение не падает к нулю
    profile = 0.2 + 0.8 * profile
    profile /= profile.max()
    return profile


def make_azimuth_profile(
    n_phi: int = N_PHI,
    amplitude: float = AZIMUTH_VAR_AMP,
    n_harmonics: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Генерирует медленную азимутальную вариацию R(φ).

    Реализовано как сумма низкочастотных гармоник со случайными фазами --
    имитирует плавную геологическую неоднородность над установкой.

    :param n_phi: число φ-бинов
    :param amplitude: амплитуда вариаций (0 = однородно, 0.2 = ±25%)
    :param n_harmonics: число гармоник (1–3: медленная вариация)
    :param rng: генератор случайных чисел
    :return: медленная азимутальную вариацию R(φ)
    """
    if rng is None:
        rng = np.random.default_rng()

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    R = np.ones(n_phi)

    for k in range(1, n_harmonics + 1):
        phase = rng.uniform(0, 2 * np.pi)
        amp_k = amplitude * rng.uniform(0.3, 1.0) / k
        R += amp_k * np.cos(k * phi + phase)

    # Нормируем к среднему = 1
    R = np.clip(R, 0.1, None)
    R /= R.mean()
    return R


def make_anomaly(
    theta_center: float,
    phi_center:   float,
    amplitude:    float,
    sigma_theta:  float,
    sigma_phi:    float,
    theta_bins:   np.ndarray,
    n_phi:        int = N_PHI,
) -> np.ndarray:
    """
    Генерирует один гауссовский пик -- имитацию аномалии плотности.

    Аномалия в угловом распределении соответствует полости или
    области пониженной плотности в направлении (theta_center, phi_center).

    :param theta_center: центр аномалии по θ (в единицах бинов)
    :param phi_center: центр аномалии по φ (в единицах бинов)
    :param amplitude: амплитуда (нормированные единицы)
    :param sigma_theta: ширина по θ (бинов)
    :param sigma_phi: ширина по φ (бинов)
    :param theta_bins: массив θ-значений в градусах, shape (T,)
    :param n_phi: число φ-бинов
    :return: один гауссовский пик
    """
    T = len(theta_bins)
    t_idx = np.arange(T)
    p_idx = np.arange(n_phi)
    TT, PP = np.meshgrid(t_idx, p_idx, indexing='ij')

    # Периодическое расстояние по φ
    dp = PP - phi_center
    dp = dp - n_phi * np.round(dp / n_phi)

    gauss = amplitude * np.exp(
        -0.5 * ((TT - theta_center) / sigma_theta) ** 2
        -0.5 * (dp / sigma_phi) ** 2
    )
    return gauss


def generate_clean(
    theta_bins: np.ndarray | None = None,
    n_phi:      int = N_PHI,
    intensity:  float = 0.7,
    n_anomalies: int | None = None,
    rng:        np.random.Generator | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Генерирует чистое (без шума) угловое распределение N*(θ, φ).

    Шаги:
      1. cos²θ × R(φ) -- базовое непрерывное поле
      2. Геометрическая маска -- обнуляет бины вне видимого конуса детектора
      3. Гауссовские аномалии -- добавляются только внутри видимой зоны

    :param theta_bins: массив θ-значений в градусах, shape (T,)
    :param n_phi: число φ-бинов
    :param intensity: суммарная статистика
    :param n_anomalies: количество анамилий
    :param rng: генератор случайных чисел
    :return: tuple:
        grid      -- чистая сетка shape (T, P), нормированная к [0, 1]
        anomalies -- список параметров аномалий (для отладки)
    """
    if rng is None:
        rng = np.random.default_rng()

    if theta_bins is None:
        step = (THETA_MAX - THETA_MIN) / N_THETA
        theta_bins = np.arange(THETA_MIN, THETA_MAX + step / 2, step)[:N_THETA]

    T = len(theta_bins)

    # Базовое распределение: cos²θ × R(φ)
    theta_profile = make_theta_profile(theta_bins, rng=rng)  # (T,)
    azimuth_profile = make_azimuth_profile(n_phi=n_phi, rng=rng)  # (P,)
    grid = intensity * np.outer(theta_profile, azimuth_profile)

    # Геометрическая маска: детектор видит только узкий φ-сектор
    geo_mask = make_geometric_mask(n_theta=T, n_phi=n_phi, rng=rng)
    grid = grid * geo_mask

    # Аномалии -- только внутри видимой зоны маски
    if n_anomalies is None:
        n_anomalies = rng.integers(
            N_ANOMALIES_RANGE[0], N_ANOMALIES_RANGE[1] + 1
        )

    # Видимые φ-позиции (где хотя бы один θ-бин ненулевой)
    visible_phi = np.where(geo_mask[0] > 0)[0]

    anomaly_params = []
    for _ in range(n_anomalies):
        if len(visible_phi) == 0:
            break
        phi_c = float(rng.choice(visible_phi))

        params = {
            "theta_center": rng.uniform(*SYNTH_ANOMALY_THETA_IDX_RANGE),
            "phi_center": phi_c,
            "amplitude": rng.uniform(*ANOMALY_AMP_RANGE) * intensity,
            "sigma_theta": rng.uniform(*ANOMALY_SIGMA_THETA),
            "sigma_phi": rng.uniform(*ANOMALY_SIGMA_PHI),
        }
        anomaly = make_anomaly(theta_bins=theta_bins, n_phi=n_phi, **params)
        grid += anomaly * geo_mask
        anomaly_params.append(params)

    grid = np.clip(grid, 0, None)
    return grid, anomaly_params


def make_geometric_mask(
    n_theta:    int = N_THETA,
    n_phi:      int = N_PHI,
    rng:        np.random.Generator | None = None,
) -> np.ndarray:
    """
    Генерирует геометрическую маску видимости детектора.

    Каждый детектор видит треки только в двух узких φ-секторах
    (прямое и обратное направление нормали к эмульсии).
    Остальные бины геометрически недоступны -- ноль.

    В реальных данных ~8% бинов ненулевые. Здесь мы имитируем
    это через два окна ±half_width бинов вокруг каждого направления.

    half_width = 12–22 бина -> ~13–25% охвата по φ для одного окна

    Из анализа реальных данных (npl4/2.0°):
      - Два симметричных прямоугольных окна с мягкими краями
      - Центры окон разделены ровно на n_phi/2 бинов (180°)
      - Ширина окна линейно растёт с θ:
          θ=58°: ~3 бина (6°) на пик
          θ=90°: ~33 бина (66°) на пик
      - Внутри каждого окна все бины ненулевые (нет пропусков)
      - Значения внутри окна плавно меняются (Гаусс по φ внутри окна)

    :param n_theta: кол-во θ-бинов
    :param n_phi: кол-вл φ-бинов
    :param rng: генератор случайных чисел
    :return: итоговая маска
    """
    if rng is None:
        rng = np.random.default_rng()

    phi_idx = np.arange(n_phi)
    phi_center = rng.uniform(0, n_phi)  # главное направление
    phi_back = (phi_center + n_phi // 2) % n_phi  # ровно через 180°

    # Интенсивность заднего пика: 40–70% (из реальных данных оба пика ≈ одинаковы,
    # но небольшая асимметрия возможна между детекторами)
    back_intensity = rng.uniform(0.85, 1.0)

    # Полуширина в бинах (full_width = 2 * half_w):
    # Реальные данные: full=12б при θ=58° -> half_w_min≈6
    #                  full=33б при θ=90° -> half_w_max≈16
    w_min = rng.uniform(5.0, 7.0)  # полуширина при θ=58°
    w_max = rng.uniform(14.0, 17.0)  # полуширина при θ=90°

    mask = np.zeros((n_theta, n_phi))
    for ti in range(n_theta):
        # Нелинейный (√t) рост ширины: быстрый в начале, медленный к θ=90°
        # Совпадает с реальными данными: 12б при θ=60° -> 33б при θ=90°
        t_frac = ti / max(n_theta - 1, 1)
        half_w = w_min + (w_max - w_min) * (t_frac ** 0.45)

        for center, intensity in [(phi_center, 1.0), (phi_back, back_intensity)]:
            dp = phi_idx - center
            dp = dp - n_phi * np.round(dp / n_phi)
            inside = np.abs(dp) <= half_w
            # Бинарная маска внутри окна: 1 везде где inside=True.
            # Это не даёт Пуассону обнулять края конуса (как в реальных данных,
            # где внутри конуса нет пустых бинов).
            mask[ti][inside] = np.maximum(mask[ti][inside], intensity)

    return mask


def add_poisson_noise(
    grid:  np.ndarray,
    scale: float | None = None,
    rng:   np.random.Generator | None = None,
) -> np.ndarray:
    """
    Добавляет пуассоновский шум к чистому распределению.

    Физический смысл: за конечное время экспозиции число зарегистрированных
    треков в каждом бине подчиняется распределению Пуассона с λ = N*(θ,φ).

    N(θ,φ) = Poisson(N*(θ,φ) / scale) × scale  только для N* > 0

    При малом scale (редкие события) шум сильнее; при большом -- слабее.

    :param grid: чистое распределение shape (T, P)
    :param scale: масштаб шума (None = случайный из NOISE_SCALE_RANGE)
    :param rng: генератор случайных чисел
    :return: пуассоновский шум
    """
    if rng is None:
        rng = np.random.default_rng()
    if scale is None:
        scale = rng.uniform(*NOISE_SCALE_RANGE)

    noisy = np.zeros_like(grid)
    nz_mask = grid > 0
    if nz_mask.any():
        lam = np.maximum(grid[nz_mask] / scale, 1e-9)
        noisy[nz_mask] = rng.poisson(lam).astype(float) * scale
    return noisy


# ─────────────────────────────────────────────────────────────────────────────
# ГЕНЕРАЦИЯ ДАТАСЕТА
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SynthSample:
    """Один синтетический образец: пара (зашумлённый вход, чистый GT)."""
    noisy:     np.ndarray   # shape (T, P)  -- вход в модель
    clean:     np.ndarray   # shape (T, P)  -- GT (эталон)
    anomalies: list[dict]   # параметры аномалий
    noise_scale: float      # использованный масштаб шума


def generate_sample(
    theta_bins:  np.ndarray | None = None,
    n_phi:       int   = N_PHI,
    seed:        int | None = None,
) -> SynthSample:
    """
    Генерирует один синтетический образец.

    :param theta_bins: θ-бины (None = стандартный диапазон из config)
    :param n_phi: число φ-бинов
    :param seed: фиксированный сид для воспроизводимости
    """
    rng = np.random.default_rng(seed)

    intensity = rng.uniform(*INTENSITY_RANGE)
    noise_scale = rng.uniform(*NOISE_SCALE_RANGE)

    clean, anomalies = generate_clean(
        theta_bins=theta_bins,
        n_phi=n_phi,
        intensity=intensity,
        rng=rng,
    )
    noisy = add_poisson_noise(clean, scale=noise_scale, rng=rng)

    return SynthSample(
        noisy=noisy,
        clean=clean,
        anomalies=anomalies,
        noise_scale=noise_scale,
    )


def generate_dataset(
    n_samples:   int  = 1000,
    theta_bins:  np.ndarray | None = None,
    n_phi:       int  = N_PHI,
    seed:        int  = 42,
    verbose:     bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Генерирует датасет из n_samples синтетических пар.

    :param n_samples: количество пар в датасете
    :param theta_bins: θ-бины (None = стандартный диапазон из config)
    :param n_phi: число φ-бинов
    :param seed: фиксированный сид для воспроизводимости
    :param verbose: режим вывода
    :return: Tuple
        X -- зашумлённые входы, shape (N, T, P)
        Y -- чистые GT,         shape (N, T, P)
    """
    if theta_bins is None:
        step = (THETA_MAX - THETA_MIN) / N_THETA
        theta_bins = np.arange(THETA_MIN, THETA_MAX + step / 2, step)[:N_THETA]

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10 ** 9, size=n_samples)

    X = np.zeros((n_samples, len(theta_bins), n_phi), dtype=np.float32)
    Y = np.zeros((n_samples, len(theta_bins), n_phi), dtype=np.float32)

    for i, s in enumerate(seeds):
        sample = generate_sample(theta_bins=theta_bins, n_phi=n_phi, seed=int(s))
        X[i] = sample.noisy.astype(np.float32)
        Y[i] = sample.clean.astype(np.float32)

        if verbose and (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_samples} образцов сгенерировано")

    return X, Y


def save_dataset(
    X:        np.ndarray,
    Y:        np.ndarray,
    filename: str = "synth_train.npz",
    out_dir:  Path = SYNTH_DIR,
) -> Path:
    """
    Сохраняет датасет в .npz файл.

    :param X: зашумлённые входы, shape (N, T, P)
    :param Y: чистые GT
    :param filename: название файла для сохранения
    :param out_dir: путь до сохранённого файла
    :return: Path -- путь до сохранённого файла
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    np.savez(path, X=X, Y=Y)
    size_mb = path.stat().st_size / 1e6
    print(f"  Сохранено: {path.name}  ({X.shape[0]} образцов, {size_mb:.1f} МБ)")
    return path


def load_dataset(
    filename: str = "synth_train.npz",
    out_dir:  Path = SYNTH_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Загружает датасет из .npz файла.

    :param filename: название файла для сохранения
    :param out_dir: путь до сохранённого файла

    :return: Tuple
        X -- зашумлённые входы, shape (N, T, P)
        Y -- чистые GT,         shape (N, T, P)
    """
    path = out_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {path}\n"
            f"Запустите: python muon_synth.py --n 5000"
        )
    data = np.load(path)
    X, Y = data["X"], data["Y"]
    print(f"  Загружено: {path.name}  ({X.shape[0]} образцов, shape={X.shape[1:]})")
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def plot_samples(
    n:          int  = 4,
    theta_bins: np.ndarray | None = None,
    seed:       int  = 0,
    save:       bool = True,
) -> plt.Figure:
    """
    Рисует n примеров: зашумлённый вход, чистый GT и разность.
    
    :param n: сколько примеров нарисовать
    :param theta_bins: θ-бины (None = стандартный диапазон из config)
    :param seed: для воспроизводимости
    :param save: сохранять ли график
    :return: рисунок plt.Figure
    """
    if theta_bins is None:
        step = (THETA_MAX - THETA_MIN) / N_THETA
        theta_bins = np.arange(THETA_MIN, THETA_MAX + step / 2, step)[:N_THETA]
    phi_bins = np.linspace(0, 360, N_PHI, endpoint=False)

    fig = plt.figure(figsize=(18, 4 * n))
    fig.suptitle("Синтетические образцы: зашумлённый вход / чистый GT / разность",
                 fontsize=13, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.45, wspace=0.3)

    for row in range(n):
        sample = generate_sample(theta_bins=theta_bins, seed=seed + row)
        titles = ["Зашумлённый вход  (X)", "Чистый GT  (Y)", "Разность  (Y − X)"]
        grids = [sample.noisy, sample.clean, sample.clean - sample.noisy]
        cmaps = ["inferno", "inferno", "RdBu_r"]

        for col, (title, grid, cmap) in enumerate(zip(titles, grids, cmaps)):
            ax = fig.add_subplot(gs[row, col])
            vmax = max(np.abs(grid).max(), 1e-6)
            kw = dict(vmin=-vmax, vmax=vmax) if col == 2 else {}
            im = ax.pcolormesh(phi_bins, theta_bins, grid,
                               cmap=cmap, shading="auto", **kw)
            plt.colorbar(im, ax=ax, shrink=0.85)
            ax.set_xlabel("φ (°)", fontsize=8)
            ax.set_ylabel("θ (°)", fontsize=8)
            n_anom = len(sample.anomalies)
            ax.set_title(
                f"{title}  (образец {row + 1}, "
                f"аномалий={n_anom}, σ_шума={sample.noise_scale:.3f})",
                fontsize=8.5, fontweight="bold",
            )

    plt.tight_layout()
    if save:
        out = OUTPUT_DIR / "fig_synth_samples.png"
        fig.savefig(out, bbox_inches="tight", dpi=130)
        print(f"  Сохранён: {out.name}")
    return fig


def plot_noise_levels(
    theta_bins: np.ndarray | None = None,
    save:       bool = True,
) -> plt.Figure:
    """
    Показывает влияние разных уровней шума на один и тот же образец.
    
    :param theta_bins: θ-бины (None = стандартный диапазон из config)
    :param save: сохранять ли график
    :return: рисунок plt.Figure
    """
    if theta_bins is None:
        step = (THETA_MAX - THETA_MIN) / N_THETA
        theta_bins = np.arange(THETA_MIN, THETA_MAX + step / 2, step)[:N_THETA]
    phi_bins = np.linspace(0, 360, N_PHI, endpoint=False)

    rng = np.random.default_rng(42)
    clean, anomalies = generate_clean(theta_bins=theta_bins, rng=rng, n_anomalies=2)

    scales = [0.005, 0.01, 0.03, 0.05]
    fig, axes = plt.subplots(1, len(scales) + 1, figsize=(22, 4))
    fig.suptitle("Влияние уровня шума на угловое распределение",
                 fontsize=12, fontweight="bold")

    axes[0].pcolormesh(phi_bins, theta_bins, clean, cmap="inferno", shading="auto")
    axes[0].set_title("Чистый GT", fontweight="bold", fontsize=9)
    axes[0].set_xlabel("φ (°)")
    axes[0].set_ylabel("θ (°)")

    rng2 = np.random.default_rng(42)
    for ax, scale in zip(axes[1:], scales):
        noisy = add_poisson_noise(clean, scale=scale, rng=rng2)
        ax.pcolormesh(phi_bins, theta_bins, noisy, cmap="inferno", shading="auto")
        nonzero_pct = (noisy > 0).mean() * 100
        ax.set_title(f"scale={scale}  ({nonzero_pct:.0f}% ненул.)",
                     fontweight="bold", fontsize=9)
        ax.set_xlabel("φ (°)")
        ax.set_ylabel("θ (°)")

    plt.tight_layout()
    if save:
        out = OUTPUT_DIR / "fig_synth_noise.png"
        fig.savefig(out, bbox_inches="tight", dpi=130)
        print(f"  Сохранён: {out.name}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def main(
    n_samples:    int  = 2000,
    n_train:      int | None = None,
    n_val:        int = 200,
    seed:         int = 42,
    show_samples: bool = True,
    show_noise:   bool = True,
    save:         bool = True,
):
    """
    Генерирует синтетический датасет и сохраняет его.

    :param n_samples: общее число образцов (train + val)
    :param n_train: число train образцов (None = n_samples - n_val)
    :param n_val: число val образцов
    :param seed: сид генератора
    :param show_*: строить ли соответствующие графики
    :param save: сохранять ли датасет в .npz
    """
    print("=" * 60)
    print("  Мюонография -- генерация синтетических данных")
    print("=" * 60)
    print(f"\n  Образцов всего : {n_samples}")
    print(f"  Размер сетки  : {N_THETA} × {N_PHI}")
    print(f"  Уровень шума  : {NOISE_SCALE_RANGE}")
    print(f"  Аномалий/обр  : {N_ANOMALIES_RANGE}")

    if n_train is None:
        n_train = n_samples - n_val

    step = (THETA_MAX - THETA_MIN) / N_THETA
    theta_bins = np.arange(THETA_MIN, THETA_MAX + step / 2, step)[:N_THETA]

    # Генерация
    print(f"\n─── Генерация train ({n_train} обр.) " + "─" * 22)
    X_train, Y_train = generate_dataset(n_train, theta_bins=theta_bins,
                                        seed=seed, verbose=True)
    print(f"\n─── Генерация val ({n_val} обр.) " + "─" * 25)
    X_val, Y_val = generate_dataset(n_val, theta_bins=theta_bins,
                                    seed=seed + 10000, verbose=False)

    print(f"\n  Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"  Val:   X={X_val.shape},   Y={Y_val.shape}")

    # Статистика
    nonzero_x = (X_train > 0).mean() * 100
    nonzero_y = (Y_train > 0).mean() * 100
    print(f"\n  Ненулевых бинов: X={nonzero_x:.1f}%, Y={nonzero_y:.1f}%")
    print(f"  X range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  Y range: [{Y_train.min():.4f}, {Y_train.max():.4f}]")

    # Сохранение
    if save:
        print("\n─── Сохранение " + "─" * 44)
        save_dataset(X_train, Y_train, "synth_train.npz")
        save_dataset(X_val, Y_val, "synth_val.npz")

    # Графики
    if show_samples:
        print("\n─── Примеры образцов  " + "─" * 38)
        plot_samples(n=3, theta_bins=theta_bins, seed=0, save=True)

    if show_noise:
        print("\n─── Уровни шума " + "─" * 43)
        plot_noise_levels(theta_bins=theta_bins, save=True)

    print("\nSUCCESS Готово!")
    print(f"  Данные: {SYNTH_DIR}")
    print(f"  Графики: {OUTPUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main(
        n_samples=5000,
        n_val=500,
        seed=42,
        show_samples=True,
        show_noise=True,
        save=True,
    )