"""
=============================================================
  Мюонография — предобработка данных
=============================================================

Этапы пайплайна:
  1. Фильтрация                 — рабочий θ-диапазон, исключение нулей
  2. Поправка на эффективность  — коррекция по EffCorFile_Tracks.dat
                                  (можно отключить если уже применена)
  3. Нормализация               — три режима: per-detector / global / angular
  4. Трансформация              — log(1+N) и/или винсоризация
  5. Построение сетки           — 2D-массив (θ × φ) для подачи в модель
  6. Сохранение                 — .npz файл, готовый к загрузке в ML

Зависимости:
  pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass, field
from scipy import stats as scipy_stats

from config import (
    DATA_ROOT, OUTPUT_DIR, PREPROC_DIR,
    GOOD_DETS, THETA_MIN, THETA_MAX,
    APPLY_EFF_CORRECTION, ZERO_THRESHOLD, EFF_CORR_MAX,
    WINSORIZE_PERCENTILE, NORMALIZATION, TRANSFORM,
    COLORS_NPL, load_tracks, load_efficiency, load_input,
    sum_all_detectors,
)

# Папка для сохранения графиков
OUTPUT_DIR = OUTPUT_DIR / "muon_preprocessing"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ПАРАМЕТРЫ ПРЕДОБРАБОТКИ — заданы в config.py, здесь только справка:
#
#   APPLY_EFF_CORRECTION  — применять поправку на эффективность (True/False)
#   ZERO_THRESHOLD        — порог для маскировки пустых бинов (0.0)
#   EFF_CORR_MAX          — макс. надёжная поправка (2.0 → исключает θ=58°)
#   WINSORIZE_PERCENTILE  — верхний перцентиль обрезки выбросов (99.5 / None)
#   NORMALIZATION         — режим нормализации (per_detector / global / angular / none)
#   TRANSFORM             — трансформация значений (log1p / sqrt / none)
#   PREPROC_DIR           — папка для сохранения .npz файлов
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASS: описывает один предобработанный детектор
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectorSample:
    """
    Один предобработанный детектор — готовая единица для ML.

    Атрибуты:
        det      — номер детектора
        npl      — уровень качества трека ("npl4", ...)
        binning  — биннинг ("2.0Grad", ...)
        theta    — массив θ-бинов (°), shape (T,)
        phi      — массив φ-бинов (°), shape (P,)
        grid_raw — сырые треки в рабочей зоне, shape (T, P)
        grid_eff — после поправки на эффективность, shape (T, P)
        grid_norm — после нормализации, shape (T, P)
        grid     — финальный массив после всех шагов, shape (T, P)
        meta     — словарь с метаданными (суммы, маски, параметры)
    """
    det:       int
    npl:       str
    binning:   str
    theta:     np.ndarray
    phi:       np.ndarray
    grid_raw:  np.ndarray
    grid_eff:  np.ndarray
    grid_norm: np.ndarray
    grid:      np.ndarray
    meta:      dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.grid.shape

    def __repr__(self):
        return (f"DetectorSample(det={self.det}, {self.npl}/{self.binning}, "
                f"shape={self.shape}, nonzero={self.meta.get('nonzero_frac', '?'):.1%})")


# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 1: ФИЛЬТРАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def filter_working_range(
    data: np.ndarray,
    theta_min: float = THETA_MIN,
    theta_max: float = THETA_MAX,
    zero_threshold: float = ZERO_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Выбирает рабочий угловой диапазон и строит 2D-сетку.

    Возвращает:
        theta_bins — уникальные θ-значения, shape (T,)
        phi_bins   — уникальные φ-значения, shape (P,)
        grid       — N_tracks на сетке (T × P), нули сохранены
        mask_zero  — булева маска нулевых ячеек, shape (T, P)
    """
    sub = data[(data[:, 0] >= theta_min) & (data[:, 0] <= theta_max)].copy()

    theta_bins = np.unique(sub[:, 0])
    phi_bins   = np.unique(sub[:, 1])
    grid = np.zeros((len(theta_bins), len(phi_bins)))

    t_idx = {v: i for i, v in enumerate(theta_bins)}
    p_idx = {v: i for i, v in enumerate(phi_bins)}
    for row in sub:
        ti = t_idx.get(row[0])
        pi = p_idx.get(row[1])
        if ti is not None and pi is not None:
            grid[ti, pi] = row[2]

    mask_zero = grid <= zero_threshold
    return theta_bins, phi_bins, grid, mask_zero


# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 2: ПОПРАВКА НА ЭФФЕКТИВНОСТЬ
# ─────────────────────────────────────────────────────────────────────────────

def build_eff_correction_map(
    eff: np.ndarray,
    theta_bins: np.ndarray,
    corr_max: float = EFF_CORR_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Строит вектор поправок для каждого θ-бина в земных координатах.

    Логика перевода:
        θ_D ≈ 90° − θ_Earth  (для вертикально установленного детектора)

    Поправка применяется как:
        N_corrected = N_raw / correction_coeff

    Ячейки с correction_coeff > corr_max помечаются как ненадёжные
    и маскируются (устанавливаются в 0 после коррекции).

    Возвращает:
        corr_vec  — вектор коэффициентов поправки, shape (T,)
        mask_bad  — булева маска ненадёжных θ-бинов, shape (T,)
    """
    # Строим словарь {theta_D_degrees -> correction_coeff}
    eff_valid = eff[eff[:, 3] > 0]  # только строки с ненулевой поправкой
    eff_dict  = dict(zip(eff_valid[:, 0], eff_valid[:, 3]))

    corr_vec = np.ones(len(theta_bins))
    mask_bad = np.zeros(len(theta_bins), dtype=bool)

    for i, theta_earth in enumerate(theta_bins):
        theta_d = 90.0 - theta_earth  # приближённый перевод

        if theta_d <= 0:
            mask_bad[i] = True
            continue

        # Находим ближайший бин в EffCorFile (шаг 0.25°)
        nearest_key = min(eff_dict.keys(), key=lambda x: abs(x - theta_d))
        if abs(nearest_key - theta_d) > 1.0:  # если ближайший далеко — не доверяем
            mask_bad[i] = True
            continue

        coeff = eff_dict[nearest_key]
        if coeff <= 0 or coeff > corr_max:
            mask_bad[i] = True
        else:
            corr_vec[i] = coeff

    return corr_vec, mask_bad


def apply_efficiency_correction(
    grid: np.ndarray,
    corr_vec: np.ndarray,
    mask_bad: np.ndarray,
) -> np.ndarray:
    """
    Применяет поправку на эффективность к сетке треков.
    Ненадёжные θ-строки обнуляются.

    grid     — shape (T, P)
    corr_vec — shape (T,)
    mask_bad — shape (T,)

    Возвращает grid_corrected shape (T, P).
    """
    grid_corr = grid.copy()

    for i, (coeff, is_bad) in enumerate(zip(corr_vec, mask_bad)):
        if is_bad:
            grid_corr[i, :] = 0.0
        else:
            grid_corr[i, :] = grid[i, :] / coeff

    return grid_corr


# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 3: НОРМАЛИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def normalize(
    grid: np.ndarray,
    mode: str = NORMALIZATION,
    theta_bins: np.ndarray | None = None,
    ref_total: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Нормализует сетку треков.

    Режимы:
        "none"         — без нормализации, возвращает копию
        "per_detector" — делит на суммарное число треков детектора
                         (убирает различия в статистике между детекторами)
        "global"       — делит на ref_total (глобальный max/sum по всем дет.)
                         (сохраняет относительные веса детекторов)
        "angular"      — нормировка на θ-профиль (theta marginal)
                         (убирает cos²θ-тренд, оставляет φ-аномалии)

    Возвращает:
        grid_norm — нормированный массив shape (T, P)
        norm_info — словарь с параметрами нормализации (для инверсии)
    """
    grid_norm = grid.copy().astype(float)
    norm_info = {"mode": mode}

    if mode == "none":
        pass

    elif mode == "per_detector":
        total = grid.sum()
        norm_info["total"] = total
        if total > 0:
            grid_norm = grid_norm / total

    elif mode == "global":
        if ref_total is None:
            raise ValueError("mode='global' требует ref_total")
        norm_info["ref_total"] = ref_total
        if ref_total > 0:
            grid_norm = grid_norm / ref_total

    elif mode == "angular":
        # Θ-профиль: среднее по φ для каждого θ-бина (только ненулевые)
        theta_profile = np.array([
            row[row > 0].mean() if (row > 0).any() else 1.0
            for row in grid
        ])
        norm_info["theta_profile"] = theta_profile
        for i, prof_val in enumerate(theta_profile):
            if prof_val > 0:
                grid_norm[i, :] = grid_norm[i, :] / prof_val

    else:
        raise ValueError(f"Неизвестный режим нормализации: '{mode}'")

    return grid_norm, norm_info


# ─────────────────────────────────────────────────────────────────────────────
# ШАГ 4: ТРАНСФОРМАЦИЯ И ВИНСОРИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def winsorize(
    grid: np.ndarray,
    percentile: float = WINSORIZE_PERCENTILE,
    mask_zero: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Обрезает выбросы сверху на уровне заданного перцентиля.
    Считается только по ненулевым ячейкам.

    Возвращает:
        grid_wins — обрезанный массив
        threshold — применённый порог
    """
    nonzero_vals = grid[grid > 0] if mask_zero is None else grid[~mask_zero]
    if len(nonzero_vals) == 0:
        return grid.copy(), 0.0

    threshold = np.percentile(nonzero_vals, percentile)
    grid_wins = np.clip(grid, 0, threshold)
    return grid_wins, threshold


def transform(
    grid: np.ndarray,
    mode: str = TRANSFORM,
) -> np.ndarray:
    """
    Применяет монотонную трансформацию к значениям N_tracks.

    Режимы:
        "none"  — без трансформации
        "log1p" — log(1 + N), приближает к нормальному распределению
        "sqrt"  — √N, мягче log1p, полезно при умеренной асимметрии
    """
    if mode == "none":
        return grid.copy()
    elif mode == "log1p":
        return np.log1p(grid)
    elif mode == "sqrt":
        return np.sqrt(grid)
    else:
        raise ValueError(f"Неизвестная трансформация: '{mode}'")


def inverse_transform(
    grid: np.ndarray,
    mode: str = TRANSFORM,
) -> np.ndarray:
    """
    Обратная трансформация (для восстановления предсказаний модели).

    "log1p" → expm1
    "sqrt"  → квадрат
    "none"  → без изменений
    """
    if mode == "none":
        return grid.copy()
    elif mode == "log1p":
        return np.expm1(grid)
    elif mode == "sqrt":
        return np.square(grid)
    else:
        raise ValueError(f"Неизвестная трансформация: '{mode}'")


# ─────────────────────────────────────────────────────────────────────────────
# ПОЛНЫЙ ПАЙПЛАЙН ДЛЯ ОДНОГО ДЕТЕКТОРА
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_detector(
    npl: str,
    binning: str,
    det: int,
    normalization: str = NORMALIZATION,
    transform_mode: str = TRANSFORM,
    winsorize_pct: float | None = WINSORIZE_PERCENTILE,
    eff_corr_max: float = EFF_CORR_MAX,
    apply_eff_correction: bool = APPLY_EFF_CORRECTION,
    ref_total: float | None = None,
    verbose: bool = False,
) -> DetectorSample | None:
    """
    Полный пайплайн предобработки для одного детектора.

    Шаги:
      1. Загрузка сырых данных
      2. Фильтрация по θ-диапазону
      3. Поправка на эффективность (если apply_eff_correction=True)
      4. Нормализация
      5. Винсоризация (опционально)
      6. Трансформация

    Параметры:
      apply_eff_correction — применять ли поправку из EffCorFile_Tracks.dat.
        True  (по умолчанию) — применять. Используй если данные сырые.
        False — пропустить. Используй если поправка уже была применена
                при сборке Tracks_DistrOutput (уточни у научного руководителя).

    Возвращает DetectorSample или None если данные недоступны.
    """
    # Загрузка
    raw = load_tracks(npl, binning, det)
    if raw is None:
        if verbose:
            print(f"  det{det:2d}: нет данных")
        return None

    # Шаг 1: фильтрация
    theta_bins, phi_bins, grid_raw, mask_zero = filter_working_range(raw)

    # Шаг 2: поправка на эффективность (опционально)
    if apply_eff_correction:
        eff = load_efficiency(npl, binning)
        corr_vec, mask_bad = build_eff_correction_map(eff, theta_bins, eff_corr_max)
        grid_eff = apply_efficiency_correction(grid_raw, corr_vec, mask_bad)
    else:
        # Поправка пропущена: grid_eff = grid_raw, нет исключённых бинов
        corr_vec = np.ones(len(theta_bins))
        mask_bad = np.zeros(len(theta_bins), dtype=bool)
        grid_eff = grid_raw.copy()

    # Шаг 3: нормализация
    grid_norm, norm_info = normalize(
        grid_eff, mode=normalization,
        theta_bins=theta_bins, ref_total=ref_total,
    )

    # Шаг 4: винсоризация (до трансформации, на нормированных данных)
    wins_threshold = None
    grid_final = grid_norm.copy()
    if winsorize_pct is not None:
        grid_final, wins_threshold = winsorize(grid_final, winsorize_pct)

    # Шаг 5: трансформация
    grid_final = transform(grid_final, mode=transform_mode)

    # Метаданные
    nonzero = (grid_raw > 0).sum()
    total   = grid_raw.size
    meta = {
        "raw_sum":              float(grid_raw.sum()),
        "eff_sum":              float(grid_eff.sum()),
        "nonzero_count":        int(nonzero),
        "total_bins":           int(total),
        "nonzero_frac":         float(nonzero / total),
        "bad_theta_bins":       int(mask_bad.sum()),
        "apply_eff_correction": apply_eff_correction,
        "corr_vec":             corr_vec,
        "mask_bad":             mask_bad,
        "mask_zero":            mask_zero,
        "norm_info":            norm_info,
        "wins_threshold":       wins_threshold,
        "transform":            transform_mode,
        "normalization":        normalization,
        "eff_corr_max":         eff_corr_max,
    }

    if verbose:
        nonz_pct = nonzero / total * 100
        bad_pct  = mask_bad.sum() / len(theta_bins) * 100
        eff_flag = "" if apply_eff_correction else " [поправка отключена]"
        print(f"  det{det:2d}: shape={grid_final.shape}, "
              f"ненулевых={nonz_pct:.1f}%, "
              f"исключено θ-бинов={mask_bad.sum()} ({bad_pct:.0f}%){eff_flag}")

    return DetectorSample(
        det=det, npl=npl, binning=binning,
        theta=theta_bins, phi=phi_bins,
        grid_raw=grid_raw,
        grid_eff=grid_eff,
        grid_norm=grid_norm,
        grid=grid_final,
        meta=meta,
    )


def preprocess_all_detectors(
    npl: str,
    binning: str,
    dets: list[int] | None = None,
    normalization: str = NORMALIZATION,
    transform_mode: str = TRANSFORM,
    winsorize_pct: float | None = WINSORIZE_PERCENTILE,
    apply_eff_correction: bool = APPLY_EFF_CORRECTION,
    verbose: bool = True,
) -> list[DetectorSample]:
    """
    Предобрабатывает все рабочие детекторы.

    При normalization='global' ref_total вычисляется автоматически
    как максимум суммарных треков среди всех детекторов.
    """
    if dets is None:
        dets = GOOD_DETS

    if verbose:
        print(f"Предобработка: {npl}/{binning}, {len(dets)} детекторов")
        print(f"  Нормализация: {normalization} | Трансформация: {transform_mode}")
        print(f"  Поправка на эффективность: {'да' if apply_eff_correction else 'нет (отключена)'}")

    # Для global normalization — предварительно считаем ref_total
    ref_total = None
    if normalization == "global":
        totals = []
        for det in dets:
            raw = load_tracks(npl, binning, det)
            if raw is None:
                continue
            theta_bins, _, grid_raw, _ = filter_working_range(raw)
            if apply_eff_correction:
                eff = load_efficiency(npl, binning)
                corr_vec, mask_bad = build_eff_correction_map(eff, theta_bins)
                grid_for_total = apply_efficiency_correction(grid_raw, corr_vec, mask_bad)
            else:
                grid_for_total = grid_raw
            totals.append(grid_for_total.sum())
        ref_total = max(totals) if totals else 1.0
        if verbose:
            print(f"  Global ref_total = {ref_total:,.1f}")

    samples = []
    for det in dets:
        sample = preprocess_detector(
            npl, binning, det,
            normalization=normalization,
            transform_mode=transform_mode,
            winsorize_pct=winsorize_pct,
            apply_eff_correction=apply_eff_correction,
            ref_total=ref_total,
            verbose=verbose,
        )
        if sample is not None:
            samples.append(sample)

    if verbose:
        print(f"\n  Готово: {len(samples)} детекторов")

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# СОХРАНЕНИЕ И ЗАГРУЗКА
# ─────────────────────────────────────────────────────────────────────────────

def save_preprocessed(
    samples: list[DetectorSample],
    npl: str,
    binning: str,
    out_dir: Path = PREPROC_DIR,
) -> Path:
    """
    Сохраняет предобработанные данные в .npz файл.

    Структура файла:
        grids       — shape (N_dets, T, P) — финальные массивы
        grids_raw   — shape (N_dets, T, P) — сырые данные
        grids_eff   — shape (N_dets, T, P) — после поправки
        detectors   — shape (N_dets,)      — номера детекторов
        theta       — shape (T,)
        phi         — shape (P,)
        meta_*      — метаданные

    Возвращает путь к сохранённому файлу.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = out_dir / f"{npl}_{binning}_preprocessed.npz"

    grids     = np.stack([s.grid for s in samples])
    grids_raw = np.stack([s.grid_raw for s in samples])
    grids_eff = np.stack([s.grid_eff for s in samples])

    np.savez(
        fname,
        grids       = grids,
        grids_raw   = grids_raw,
        grids_eff   = grids_eff,
        detectors   = np.array([s.det for s in samples]),
        theta       = samples[0].theta,
        phi         = samples[0].phi,
        raw_sums    = np.array([s.meta["raw_sum"] for s in samples]),
        nonzero_fracs = np.array([s.meta["nonzero_frac"] for s in samples]),
        bad_theta_bins = np.array([s.meta["bad_theta_bins"] for s in samples]),
        normalization  = samples[0].meta["normalization"],
        transform_mode = samples[0].meta["transform"],
    )

    size_mb = fname.stat().st_size / 1e6
    print(f"  Сохранено: {fname.name}  ({size_mb:.1f} МБ)")
    print(f"  grids shape: {grids.shape}")
    return fname


def load_preprocessed(
    npl: str,
    binning: str,
    out_dir: Path = PREPROC_DIR,
) -> dict:
    """
    Загружает предобработанные данные из .npz файла.

    Возвращает словарь с ключами:
        grids, grids_raw, grids_eff, detectors, theta, phi,
        raw_sums, nonzero_fracs, normalization, transform_mode
    """
    fname = out_dir / f"{npl}_{binning}_preprocessed.npz"
    if not fname.exists():
        raise FileNotFoundError(
            f"Предобработанные данные не найдены: {fname}\n"
            f"Запустите: python muon_preprocessing.py --npl {npl} --binning {binning}"
        )
    data = dict(np.load(fname, allow_pickle=True))
    print(f"  Загружено: {fname.name}")
    print(f"  grids shape: {data['grids'].shape}  "
          f"(детекторов × θ-бинов × φ-бинов)")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ПРЕДОБРАБОТКИ
# ─────────────────────────────────────────────────────────────────────────────

def plot_preprocessing_steps(
    sample: DetectorSample,
    save: bool = True,
) -> plt.Figure:
    """
    Для одного детектора показывает все шаги пайплайна:
    сырые данные → после поправки → после нормализации → финал.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Шаги предобработки — дет.№{sample.det}, "
        f"{sample.npl}/{sample.binning}",
        fontsize=13, fontweight="bold",
    )

    panels = [
        (sample.grid_raw,  "1. Сырые данные (N_tracks)",         "inferno"),
        (sample.grid_eff,  "2. После поправки на эффективность", "inferno"),
        (sample.grid_norm, "3. После нормализации",              "viridis"),
        (sample.grid,      f"4. Финал ({sample.meta['transform']})", "viridis"),
    ]

    for ax, (grid, title, cmap) in zip(axes.flat, panels):
        im = ax.pcolormesh(
            sample.phi, sample.theta, grid,
            cmap=cmap, shading="auto",
        )
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("φ (°)")
        ax.set_ylabel("θ (°)")
        ax.set_title(title, fontweight="bold")

        # Подсветить исключённые θ-строки
        if "mask_bad" in sample.meta:
            bad_thetas = sample.theta[sample.meta["mask_bad"]]
            for bt in bad_thetas:
                ax.axhline(bt, color="cyan", lw=0.5, alpha=0.4)

    plt.tight_layout()
    if save:
        out = OUTPUT_DIR / f"fig_preproc_det{sample.det}_{sample.npl}.png"
        fig.savefig(out, bbox_inches="tight", dpi=130)
        print(f"  Сохранён: {out.name}")
    return fig


def plot_preprocessing_summary(
    samples: list[DetectorSample],
    save: bool = True,
) -> plt.Figure:
    """
    Сводный дашборд по всем детекторам:
    - Суммарные треки до/после поправки
    - Распределение N_tracks до/после трансформации
    - Суммарная карта после предобработки
    - Статистика по детекторам
    """
    npl     = samples[0].npl
    binning = samples[0].binning

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Итоги предобработки — {npl}/{binning}\n"
        f"Нормализация: {samples[0].meta['normalization']} | "
        f"Трансформация: {samples[0].meta['transform']}",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    det_ids  = [s.det for s in samples]
    raw_sums = [s.meta["raw_sum"] for s in samples]
    eff_sums = [s.meta["eff_sum"] for s in samples]
    bad_bins = [s.meta["bad_theta_bins"] for s in samples]

    # ── (0,0) Суммы треков до/после поправки ────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(det_ids))
    ax.bar(x - 0.2, [v / 1e3 for v in raw_sums], 0.4,
           label="Сырые", color="#90CAF9", edgecolor="k", linewidth=0.4)
    ax.bar(x + 0.2, [v / 1e3 for v in eff_sums], 0.4,
           label="После поправки", color="#1565C0", edgecolor="k", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([f"#{d}" for d in det_ids], fontsize=7)
    ax.set_ylabel("Треков (тыс.)")
    ax.set_title("Треки до/после поправки\nна эффективность", fontweight="bold")
    ax.legend(fontsize=9)

    # ── (0,1) Исключённые θ-бины ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x, bad_bins, color="#EF9A9A", edgecolor="k", linewidth=0.4)
    ax2.set_xticks(x); ax2.set_xticklabels([f"#{d}" for d in det_ids], fontsize=7)
    ax2.set_ylabel("Исключено θ-бинов")
    ax2.set_title("Ненадёжные θ-бины\n(поправка > {:.0f}×)".format(
        samples[0].meta["eff_corr_max"]), fontweight="bold")

    # ── (0,2) Распределение N_tracks: raw vs final ──────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    all_raw   = np.concatenate([s.grid_raw[s.grid_raw > 0].ravel() for s in samples])
    all_final = np.concatenate([s.grid[s.grid > 0].ravel() for s in samples])
    ax3.hist(all_raw, bins=50, alpha=0.5, density=True,
             color="#FF9800", label="Сырые", edgecolor="k", linewidth=0.2)
    ax3.hist(all_final, bins=50, alpha=0.5, density=True,
             color="#4CAF50", label="Финал", edgecolor="k", linewidth=0.2)
    ax3.set_xlabel("N_tracks / трансформ. значение")
    ax3.set_ylabel("Плотность")
    ax3.set_title("Распределение N_tracks\n(ненулевые бины)", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.set_yscale("log")

    # ── (1,0–1) Суммарная карта после предобработки ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 0:2])
    total_grid = sum(s.grid for s in samples)
    theta = samples[0].theta
    phi   = samples[0].phi
    im = ax4.pcolormesh(phi, theta, total_grid, cmap="plasma", shading="auto")
    plt.colorbar(im, ax=ax4, label="Сумма по детекторам")
    ax4.set_xlabel("φ (°)"); ax4.set_ylabel("θ (°)")
    ax4.set_title(
        f"Суммарная карта после предобработки ({len(samples)} дет.)",
        fontweight="bold",
    )

    # ── (1,2) Сводная статистика ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    step = float(binning.replace("Grad", ""))
    n_theta = len(samples[0].theta)
    n_phi   = len(samples[0].phi)
    table_data = [
        ["Детекторов",      str(len(samples))],
        ["Форма сетки",     f"{n_theta} × {n_phi}"],
        ["θ-бинов рабочих", f"{n_theta - int(np.mean(bad_bins))}/{n_theta}"],
        ["Нормализация",    samples[0].meta["normalization"]],
        ["Трансформация",   samples[0].meta["transform"]],
        ["Нулевых бинов",   f"{(total_grid==0).mean()*100:.1f}%"],
        ["Max финал",       f"{total_grid.max():.3f}"],
        ["Сохранено в",     f"data/preprocessed/"],
    ]
    tbl = ax5.table(
        cellText=table_data,
        colLabels=["Параметр", "Значение"],
        cellLoc="left", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ECEFF1")
    ax5.set_title("Параметры предобработки", fontweight="bold", pad=10)

    if save:
        out = OUTPUT_DIR / f"fig_preproc_summary_{npl}_{binning}.png"
        fig.savefig(out, bbox_inches="tight", dpi=130)
        print(f"  Сохранён: {out.name}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ДИАГНОСТИКА: выбор оптимальных параметров
# ─────────────────────────────────────────────────────────────────────────────

def analyze_parameters(
    npl: str = "npl4",
    binning: str = "2.0Grad",
    save: bool = True,
) -> plt.Figure:
    """
    Анализирует данные и строит диагностический дашборд для обоснованного
    выбора нормализации, трансформации и настройки поправки на эффективность.

    Что показывает каждая панель:
      (0,0) Гистограммы raw / log1p / sqrt + нормальность (Q-Q)
      (0,1) Q-Q plot для трёх трансформаций
      (0,2) Поправка на эффективность по θ — насколько она значима
      (1,0) Вариация суммарных треков между детекторами
      (1,1) θ-профиль: насколько силён cos²θ тренд (нужна ли angular norm)
      (1,2) Итоговые рекомендации в виде таблицы
    """
    print(f"Диагностика параметров: {npl}/{binning}")

    # ── Сбор данных ──────────────────────────────────────────────────────────
    raw_vals, eff_vals, sums_raw, sums_eff = [], [], [], []
    theta_profile_raw = None
    theta_profile_eff = None

    eff_data = load_efficiency(npl, binning)

    for det in GOOD_DETS:
        raw = load_tracks(npl, binning, det)
        if raw is None:
            continue
        theta_bins, _, grid_raw, _ = filter_working_range(raw)
        corr_vec, mask_bad = build_eff_correction_map(eff_data, theta_bins)
        grid_eff = apply_efficiency_correction(grid_raw, corr_vec, mask_bad)

        nz_raw = grid_raw[grid_raw > 0].ravel()
        nz_eff = grid_eff[grid_eff > 0].ravel()
        raw_vals.extend(nz_raw.tolist())
        eff_vals.extend(nz_eff.tolist())
        sums_raw.append(grid_raw.sum())
        sums_eff.append(grid_eff.sum())

        if theta_profile_raw is None:
            theta_profile_raw = grid_raw.copy()
            theta_profile_eff = grid_eff.copy()
        else:
            theta_profile_raw += grid_raw
            theta_profile_eff += grid_eff

    raw_vals = np.array(raw_vals)
    eff_vals = np.array(eff_vals)
    sums_raw = np.array(sums_raw)
    sums_eff = np.array(sums_eff)

    # Θ-профиль: среднее по φ для каждого θ-бина
    t_profile_r = theta_profile_raw.mean(axis=1)
    t_profile_e = theta_profile_eff.mean(axis=1)

    # ── Статистики для трансформаций ─────────────────────────────────────────
    transforms = {
        "raw":   eff_vals,
        "log1p": np.log1p(eff_vals),
        "sqrt":  np.sqrt(eff_vals),
    }
    stats_table = {}
    for name, arr in transforms.items():
        idx = np.random.choice(len(arr), min(5000, len(arr)), replace=False)
        sw_p = scipy_stats.shapiro(arr[idx]).pvalue
        stats_table[name] = {
            "skew":  scipy_stats.skew(arr),
            "kurt":  scipy_stats.kurtosis(arr),
            "cv":    arr.std() / arr.mean() * 100,
            "sw_p":  sw_p,
        }

    # ── Вариация между детекторами ────────────────────────────────────────────
    cv_raw = sums_raw.std() / sums_raw.mean() * 100
    cv_eff = sums_eff.std() / sums_eff.mean() * 100
    ratio  = sums_eff.max() / sums_eff.min()

    # ── Насколько силён θ-тренд (нужна ли angular norm) ─────────────────────
    # Если max/min профиля > 3 — тренд сильный, angular помогает
    theta_trend_ratio = t_profile_e.max() / max(t_profile_e[t_profile_e > 0].min(), 1e-9)

    # ── Строим рисунок ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Диагностика параметров предобработки — {npl}/{binning}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    colors_tr = {"raw": "#EF9A9A", "log1p": "#A5D6A7", "sqrt": "#90CAF9"}

    # ── (0,0) Гистограммы трёх трансформаций ─────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    for name, arr in transforms.items():
        ax00.hist(arr, bins=60, alpha=0.55, density=True,
                  color=colors_tr[name], label=name, edgecolor="k", linewidth=0.2)
    ax00.set_xlabel("Значение")
    ax00.set_ylabel("Плотность (лог.)")
    ax00.set_title("Распределение N_tracks\n(ненулевые бины, после поправки)",
                   fontweight="bold")
    ax00.legend(fontsize=9)
    ax00.set_yscale("log")

    # ── (0,1) Q-Q plot для каждой трансформации ──────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    for name, arr in transforms.items():
        idx = np.random.choice(len(arr), min(3000, len(arr)), replace=False)
        qq = scipy_stats.probplot(arr[idx], dist="norm")
        ax01.scatter(qq[0][0], qq[0][1], s=4, alpha=0.4,
                     color=colors_tr[name], label=name)
    # Референсная прямая для log1p
    arr_log = transforms["log1p"]
    idx_l   = np.random.choice(len(arr_log), min(3000, len(arr_log)), replace=False)
    qq_l    = scipy_stats.probplot(arr_log[idx_l], dist="norm")
    x_ref   = np.array([qq_l[0][0].min(), qq_l[0][0].max()])
    ax01.plot(x_ref, qq_l[1][0] * x_ref + qq_l[1][1],
              color="gray", lw=1.5, ls="--", label="Нормаль")
    ax01.set_xlabel("Теор. квантили")
    ax01.set_ylabel("Выб. квантили")
    ax01.set_title("Q-Q plot — насколько\nблизко к нормальному?", fontweight="bold")
    ax01.legend(fontsize=9, markerscale=3)

    # ── (0,2) Поправка на эффективность ──────────────────────────────────────
    ax02 = fig.add_subplot(gs[0, 2])
    corr_vec_full, mask_bad_full = build_eff_correction_map(eff_data, theta_bins)
    theta_arr = theta_bins

    colors_eff = ["tomato" if (c > EFF_CORR_MAX or c == 0) else "#1565C0"
                  for c in corr_vec_full]
    ax02.bar(range(len(theta_arr)), corr_vec_full,
             color=colors_eff, edgecolor="k", linewidth=0.3)
    ax02.axhline(1.0,          color="gray", ls="--", lw=1.2, label="Поправка = 1")
    ax02.axhline(EFF_CORR_MAX, color="tomato", ls=":", lw=1.5,
                 label=f"Порог = {EFF_CORR_MAX}×")
    ax02.set_xticks(range(len(theta_arr)))
    ax02.set_xticklabels([f"{t:.0f}°" for t in theta_arr], rotation=45, fontsize=7)
    ax02.set_xlabel("θ (°)")
    ax02.set_ylabel("Коэффициент поправки")
    ax02.set_title("Поправка на эффективность\nКрасный = ненадёжный бин",
                   fontweight="bold")
    ax02.legend(fontsize=9)
    n_bad = mask_bad_full.sum()
    ax02.text(0.02, 0.97,
              f"Ненадёжных θ-бинов: {n_bad}/{len(theta_arr)}",
              transform=ax02.transAxes, va="top", fontsize=9,
              color="tomato" if n_bad > 0 else "green")

    # ── (1,0) Вариация между детекторами ─────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(GOOD_DETS))
    ax10.bar(x - 0.2, sums_raw / 1e3, 0.38, label="Сырые",
             color="#90CAF9", edgecolor="k", linewidth=0.3)
    ax10.bar(x + 0.2, sums_eff / 1e3, 0.38, label="После поправки",
             color="#1565C0", edgecolor="k", linewidth=0.3)
    ax10.set_xticks(x)
    ax10.set_xticklabels([f"#{d}" for d in GOOD_DETS], rotation=45, fontsize=7)
    ax10.set_ylabel("Треков (тыс.)")
    ax10.set_title(
        f"Статистика треков по детекторам\nCV={cv_eff:.1f}%, ratio={ratio:.1f}×",
        fontweight="bold",
    )
    ax10.legend(fontsize=9)
    ax10.text(
        0.02, 0.97,
        "→ per_detector norm" if cv_eff > 20 else "→ global norm достаточно",
        transform=ax10.transAxes, va="top", fontsize=9, color="#1B5E20",
    )

    # ── (1,1) Θ-профиль ──────────────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(theta_bins, t_profile_r / t_profile_r.max(),
              "o--", color="#EF9A9A", lw=1.5, ms=5, label="Сырые (норм.)")
    ax11.plot(theta_bins, t_profile_e / t_profile_e.max(),
              "s-",  color="#1565C0", lw=2,   ms=5, label="После поправки (норм.)")
    cos2 = np.cos(np.radians(theta_bins)) ** 2
    ax11.plot(theta_bins, cos2 / cos2.max(),
              "k:", lw=1.5, label="cos²θ (теория)")
    ax11.set_xlabel("θ (°)")
    ax11.set_ylabel("Нормированный профиль")
    ax11.set_title(
        f"θ-профиль: тренд max/min = {theta_trend_ratio:.1f}×\n"
        f"{'→ angular norm рекомендуется' if theta_trend_ratio > 3 else '→ angular norm опциональна'}",
        fontweight="bold",
    )
    ax11.legend(fontsize=9)

    # ── (1,2) Итоговые рекомендации ──────────────────────────────────────────
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.axis("off")

    # Логика рекомендаций на основе данных
    best_transform = min(
        stats_table,
        key=lambda k: abs(stats_table[k]["skew"]) + abs(stats_table[k]["kurt"]) / 3
    )
    norm_rec = "per_detector" if cv_eff > 20 else "global"
    angular_rec = "да" if theta_trend_ratio > 3 else "нет"

    def sw_verdict(p):
        if p > 0.05:   return "нормальное ✓"
        if p > 0.001:  return "близко к норм."
        return "не норм."

    rows = [
        ["Параметр", "Значение", "Вывод"],
        ["Трансформация",
         f"skew raw={stats_table['raw']['skew']:.2f}",
         f"→ {best_transform}"],
        ["  log1p skew",  f"{stats_table['log1p']['skew']:.2f}",
         sw_verdict(stats_table["log1p"]["sw_p"])],
        ["  sqrt  skew",  f"{stats_table['sqrt']['skew']:.2f}",
         sw_verdict(stats_table["sqrt"]["sw_p"])],
        ["Нормализация",
         f"CV={cv_eff:.1f}%, ratio={ratio:.1f}×",
         f"→ {norm_rec}"],
        ["Angular norm",
         f"тренд={theta_trend_ratio:.1f}×",
         f"→ {angular_rec}"],
        ["Поправка эфф.",
         f"{n_bad} ненадёжных θ",
         "проверь у руковод." if n_bad == 0 else f"→ исключить {n_bad} θ-бин(а)"],
        ["Винсоризация",
         f"p99.5={np.percentile(eff_vals, 99.5):.1f}",
         "→ рекомендуется"],
    ]

    tbl = ax12.table(
        cellText=[r[1:] for r in rows[1:]],
        colLabels=rows[0][1:],
        rowLabels=[r[0] for r in rows[1:]],
        cellLoc="left", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ECEFF1")
        # Подсветить рекомендованную трансформацию
        txt = str(cell.get_text().get_text())
        if f"→ {best_transform}" in txt or "✓" in txt:
            cell.set_facecolor("#C8E6C9")
    ax12.set_title("Рекомендации по параметрам", fontweight="bold", pad=10)

    plt.tight_layout()
    if save:
        out = OUTPUT_DIR / f"fig_param_analysis_{npl}_{binning}.png"
        fig.savefig(out, bbox_inches="tight", dpi=130)
        print(f"  Сохранён: {out.name}")

    # Вывод текстового резюме
    print("\n─── Рекомендации " + "─" * 42)
    print(f"  Трансформация    : {best_transform}  "
          f"(skew={stats_table[best_transform]['skew']:.2f}, "
          f"cv={stats_table[best_transform]['cv']:.1f}%)")
    print(f"  Нормализация     : {norm_rec}  "
          f"(CV детекторов={cv_eff:.1f}%, ratio={ratio:.1f}×)")
    print(f"  Angular norm     : {'рекомендуется' if theta_trend_ratio > 3 else 'опциональна'}  "
          f"(тренд={theta_trend_ratio:.1f}×)")
    print(f"  Ненадёжных θ-бин : {n_bad}  "
          f"({'проверь у руководителя — возможно поправка уже применена' if n_bad == 0 else 'будут исключены'})")
    print(f"  Винсоризация     : p99.5 = {np.percentile(eff_vals, 99.5):.1f}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def main(
    npl: str = "npl4",
    binning: str = "2.0Grad",
    normalization: str = NORMALIZATION,
    transform_mode: str = TRANSFORM,
    apply_eff_correction: bool = APPLY_EFF_CORRECTION,
    winsorize_pct: float | None = WINSORIZE_PERCENTILE,
    analyze: bool = False,
    save_npz: bool = True,
    save_plots: bool = True,
):
    """
    Точка входа для запуска предобработки или диагностики.

    Параметры:
        npl               — уровень качества трека: "npl3", "npl4", "npl5", "npl6"
        binning           — угловой биннинг: "1.0Grad", "1.5Grad", "2.0Grad", "2.5Grad"
        normalization     — режим нормализации (см. константу NORMALIZATION)
        transform_mode    — трансформация значений (см. константу TRANSFORM)
        apply_eff_correction — применять ли поправку на эффективность
        winsorize_pct     — перцентиль для обрезки выбросов, None = отключить
        analyze           — если True, запускает диагностику вместо предобработки
        save_npz          — сохранять ли предобработанные данные в .npz
        save_plots        — сохранять ли графики в figures/
    """
    print("=" * 60)
    print("  Мюонография — предобработка данных")
    print("=" * 60)

    # ── Режим диагностики ────────────────────────────────────────────────────
    if analyze:
        print(f"\n  Режим: ДИАГНОСТИКА ПАРАМЕТРОВ")
        print(f"  Данные: {npl} / {binning}\n")
        analyze_parameters(npl, binning, save=save_plots)
        plt.show()
        return

    # ── Режим предобработки ──────────────────────────────────────────────────
    print(f"\n  Конфигурация     : {npl} / {binning}")
    print(f"  Нормализация     : {normalization}")
    print(f"  Трансформация    : {transform_mode}")
    print(f"  Поправка эфф.    : {'да' if apply_eff_correction else 'нет'}")
    print(f"  Винсоризация     : {f'{winsorize_pct}%' if winsorize_pct else 'отключена'}")
    print(f"  Детекторы        : {GOOD_DETS}\n")

    print("─── Предобработка " + "─" * 41)
    samples = preprocess_all_detectors(
        npl=npl,
        binning=binning,
        normalization=normalization,
        transform_mode=transform_mode,
        winsorize_pct=winsorize_pct,
        apply_eff_correction=apply_eff_correction,
        verbose=True,
    )

    if not samples:
        print("⚠  Нет данных. Проверь DATA_ROOT в config.py")
        return

    if save_npz:
        print("\n─── Сохранение " + "─" * 44)
        save_preprocessed(samples, npl, binning)

    if save_plots:
        print("\n─── Визуализация " + "─" * 42)
        sample4 = next((s for s in samples if s.det == 4), samples[0])
        plot_preprocessing_steps(sample4, save=True)
        plot_preprocessing_summary(samples, save=True)

    print("\n✓ Готово!")
    print(f"  Данные: {PREPROC_DIR}")
    print(f"  Графики: {OUTPUT_DIR}")

    print("\n─── Пример использования в следующем скрипте " + "─" * 14)
    print("""
  from muon_preprocessing import load_preprocessed, inverse_transform

  data      = load_preprocessed("npl4", "2.0Grad")
  grids     = data["grids"]        # shape (15, 17, 180)  — вход в модель
  grids_raw = data["grids_raw"]    # shape (15, 17, 180)  — сырые данные
  theta     = data["theta"]        # shape (17,)  — ось θ
  phi       = data["phi"]          # shape (180,) — ось φ
  detectors = data["detectors"]    # shape (15,)  — номера детекторов

  # Восстановить исходный масштаб из предсказания модели:
  # n_tracks = inverse_transform(prediction, mode=str(data["transform_mode"]))
    """)

    plt.show()
    return samples


if __name__ == "__main__":
    main(
        npl="npl4",
        binning="2.0Grad",
        normalization=NORMALIZATION,
        transform_mode=TRANSFORM,
        apply_eff_correction=APPLY_EFF_CORRECTION,
        winsorize_pct=WINSORIZE_PERCENTILE,
        analyze=False,
        save_npz=True,
        save_plots=True,
    )

"""
=============================================================================
СПРАВОЧНИК ПО ПАРАМЕТРАМ ПРЕДОБРАБОТКИ
=============================================================================

Все константы задаются в config.py и автоматически импортируются сюда.
Менять значения нужно только там — не в этом файле.

─────────────────────────────────────────────────────────────────────────────
npl  — уровень качества трека
─────────────────────────────────────────────────────────────────────────────

Параметр задаётся при вызове main() или preprocess_all_detectors().
Определяет, из какой папки берутся данные (npl3/, npl4/ и т.д.).

  "npl4"  →  1.76M треков, рекомендуется как основной датасет.
             Баланс между статистикой и качеством трека.

  "npl5"  →  1.24M треков (70% от npl4).
             Треки чище — прошли минимум через 5 пластин.

  "npl6"  →  0.61M треков (34% от npl4).
             Максимальная чистота. Используй как эталон для валидации.

  "npl3"  →  Неполный архив. Не использовать в ML.

─────────────────────────────────────────────────────────────────────────────
binning  — угловой биннинг
─────────────────────────────────────────────────────────────────────────────

Шаг разбивки сферического пространства по θ и φ.
Влияет на размер выходного массива grids и статистику в каждом бине.

  "1.0Grad"  →  shape (17, 360). ~1 трек/бин. Слишком разрежено для ML.

  "1.5Grad"  →  shape (17, 240). ~4 трека/бин. Пограничный вариант.

  "2.0Grad"  →  shape (17, 180). ~16 треков/бин. Оптимально.
                Рекомендуется как основной.

  "2.5Grad"  →  shape (17, 144). ~25 треков/бин. Хуже разрешение,
                лучше статистика. Использовать как альтернативу.

Примечание: 17 θ-бинов — это рабочий диапазон [58°, 90°].
            При APPLY_EFF_CORRECTION=True остаётся 15 надёжных бинов
            (θ=58° исключается — поправка 3.4× слишком велика).

─────────────────────────────────────────────────────────────────────────────
APPLY_EFF_CORRECTION  — поправка на эффективность детектора
─────────────────────────────────────────────────────────────────────────────

Детектор регистрирует треки под разными углами с разной эффективностью.
EffCorFile_Tracks.dat содержит поправочный коэффициент k(θ_D):

    N_corrected = N_raw / k(θ_D)

где θ_D — угол в системе координат детектора, θ_D ≈ 90° − θ_Земля.

Поправка надёжна только при θ_D ≤ 30° (θ_Земля ≥ 60°).
За пределами этого диапазона коэффициент слишком большой или равен нулю.

  True   →  применить. Используй если данные сырые (по умолчанию).

  False  →  пропустить. Используй если поправка уже была применена
            при сборке Tracks_DistrOutput (уточнить у руководителя).
            При False исключённых θ-бинов нет — все 17 остаются.

Управляется константой в config.py:  APPLY_EFF_CORRECTION = True

─────────────────────────────────────────────────────────────────────────────
NORMALIZATION  — нормализация N_tracks
─────────────────────────────────────────────────────────────────────────────

Зачем нужна: детекторы регистрируют разное суммарное число треков —
разброс достигает 5.9×, CV = 39%. Без нормализации модель видит детекторы
с разной «яркостью» без физической причины.

  "per_detector"  →  делит каждый детектор на его суммарные треки.
                     Все детекторы имеют одинаковый общий вес.
                     Рекомендуется при CV > 20%.

  "global"        →  делит на максимум среди всех детекторов.
                     Сохраняет относительные веса детекторов.
                     Использовать если важна абсолютная разница.

  "angular"       →  делит каждую θ-строку на её среднее по φ.
                     Убирает cos²θ тренд, оставляет только φ-аномалии.
                     Рекомендуется если задача — поиск аномалий по φ.
                     Диагностика показала тренд 4.7× → применимо.

  "none"          →  без нормализации. Только для отладки.

Управляется константой в config.py:  NORMALIZATION = "per_detector"

─────────────────────────────────────────────────────────────────────────────
TRANSFORM  — трансформация значений N_tracks
─────────────────────────────────────────────────────────────────────────────

Зачем нужна: сырое распределение N_tracks сильно скошено вправо (skew=1.53,
CV=94.6%). Большинство нейронных сетей работают лучше с симметричным
распределением. Трансформация применяется последней — после нормализации
и винсоризации.

  "log1p"  →  log(1 + N).
              skew падает до −0.51, CV до 20.3%.
              Близко к нормальному. Рекомендуется.
              Обратное: expm1(x) = e^x − 1.

  "sqrt"   →  √N.
              skew падает до 0.68, CV до 48.9%.
              Мягче log1p — меньше сжимает крупные значения.
              Использовать если log1p кажется слишком агрессивным.
              Обратное: x².

  "none"   →  без трансформации.
              Подходит для градиентного бустинга и Random Forest,
              которые нечувствительны к масштабу значений.

Управляется константой в config.py:  TRANSFORM = "log1p"

Обратное преобразование (для восстановления предсказаний модели):

    from muon_preprocessing import inverse_transform
    n_tracks = inverse_transform(model_output, mode="log1p")

─────────────────────────────────────────────────────────────────────────────
WINSORIZE_PERCENTILE  — обрезка выбросов
─────────────────────────────────────────────────────────────────────────────

Заменяет все значения выше p-го перцентиля на пороговое значение.
Применяется ПОСЛЕ нормализации и ДО трансформации.
Считается только по ненулевым бинам.

По данным npl4/2.0°: p99.5 ≈ 455 треков.
Единичные выбросы детекторов 12, 14, 17 (z > 4) попадают выше этого порога.

  99.5   →  обрезает верхние 0.5% (рекомендуется по умолчанию).
  99.0   →  агрессивнее, если выбросов много.
  None   →  отключить винсоризацию.

Управляется константой в config.py:  WINSORIZE_PERCENTILE = 99.5

─────────────────────────────────────────────────────────────────────────────
EFF_CORR_MAX  — порог надёжности поправки
─────────────────────────────────────────────────────────────────────────────

θ-бины, для которых коэффициент поправки превышает EFF_CORR_MAX,
исключаются из анализа (устанавливаются в 0).

По данным npl4/2.0°:
  θ=58° → θ_D=32° → коэфф=3.40 → исключается (> 2.0)
  θ=60° → θ_D=30° → коэфф=1.65 → остаётся

  2.0  →  исключает θ-бины с поправкой >2× (рекомендуется).
  1.5  →  строже, исключит θ=60° тоже (коэфф=1.65).
  3.0  →  мягче, оставит θ=58° (ненадёжно).

Управляется константой в config.py:  EFF_CORR_MAX = 2.0

=============================================================================
СПРАВОЧНИК ПО ПАРАМЕТРАМ ПРЕДОБРАБОТКИ (подробно)
=============================================================================

Этот блок описывает все параметры, их физический смысл и рекомендации
по выбору. Константы по умолчанию хранятся в config.py.

─────────────────────────────────────────────────────────────────────────────
npl — уровень качества трека (number of plates)
─────────────────────────────────────────────────────────────────────────────

  Определяет минимальное число эмульсионных пластин, в которых одновременно
  зарегистрирован трек. Чем выше npl — тем чище треки (меньше фоновых
  событий), но тем меньше общая статистика.

  Аналог в электронных детекторах: кратность срабатывания сцинтилляторов.

  Варианты и их характеристики (биннинг 2.0°, рабочий диапазон):

    "npl4"  —  1.764M треков  —  рекомендуется как основной датасет
    "npl5"  —  1.238M треков  —  70% от npl4, меньше шума
    "npl6"  —  0.606M треков  —  35% от npl4, максимальная чистота;
                                  использовать как эталон при валидации
    "npl3"  —  неполный архив —  не использовать в ML

─────────────────────────────────────────────────────────────────────────────
binning — угловой биннинг
─────────────────────────────────────────────────────────────────────────────

  Шаг разбивки сферического пространства по θ и φ. Определяет размер
  выходного массива (N_theta × N_phi) и компромисс между угловым
  разрешением и статистической наполненностью каждого бина.

    "1.0Grad"  →  17 × 360 = 6120 бинов,  ~1 трек/бин  —  слишком разрежено
    "1.5Grad"  →  17 × 240 = 4080 бинов,  ~4 трека/бин —  пограничный вариант
    "2.0Grad"  →  17 × 180 = 3060 бинов,  ~16 треков/бин — оптимально ✓
    "2.5Grad"  →  17 × 144 = 2448 бинов,  ~25 треков/бин — меньше разрешение

  Рекомендуется: "2.0Grad". Форма выходного массива grids: (15, 17, 180).

─────────────────────────────────────────────────────────────────────────────
apply_eff_correction — поправка на эффективность детектора
─────────────────────────────────────────────────────────────────────────────

  Детектор регистрирует треки с разной эффективностью в зависимости от угла
  θ_D (угол трека в системе координат самого детектора). При больших θ_D
  трек проходит пластины под наклоном, часть треков «вылезает» за границы
  пластины и не набирает нужного числа npl — таким образом детектор
  «недосчитывает» реальные события.

  Поправочный коэффициент из EffCorFile_Tracks.dat устраняет это смещение:
      N_corrected = N_raw / correction_coeff

  Надёжность поправки по диапазонам (для нашей установки):

    θ_D ≤ 22°  →  поправка в пределах ±15%  →  вполне надёжно
    θ_D ≤ 30°  →  поправка ≤ 2.0×           →  допустимо (EFF_CORR_MAX)
    θ_D > 30°  →  поправка > 2.0× или = 0   →  ненадёжно, бин исключается

  Связь с земными координатами (θ_D ≈ 90° − θ_Earth):

    θ_Earth = 58°  →  θ_D ≈ 32°  →  поправка 3.4×  →  ИСКЛЮЧАЕТСЯ
    θ_Earth = 60°  →  θ_D ≈ 30°  →  поправка 1.65× →  допустимо
    θ_Earth ≥ 62°  →  θ_D ≤ 28°  →  поправка ≤ 1.4× →  надёжно ✓

  True  — применять (по умолчанию; используй если данные сырые)
  False — пропустить (если поправка уже была применена при сборке данных;
          уточни у научного руководителя)

─────────────────────────────────────────────────────────────────────────────
normalization — нормализация
─────────────────────────────────────────────────────────────────────────────

  Детекторы регистрируют разное суммарное число треков из-за разного
  расположения и ориентации. Разброс в наших данных: 5.9×, CV = 39%.
  Без нормализации модель видит детекторы с разной «яркостью» без какой-
  либо физической причины — это мешает обобщению.

  "per_detector"  — делит каждый детектор на его собственную сумму треков:
                      grid_norm = grid / grid.sum()
                    Все детекторы получают одинаковый «вес». Устраняет
                    разброс в абсолютных значениях. Рекомендуется при CV > 20%.

  "global"        — делит на максимум суммы треков среди всех детекторов:
                      grid_norm = grid / max(det_sums)
                    Сохраняет относительные веса детекторов. Применять когда
                    важно, что один детектор видит больше треков, чем другой.

  "angular"       — делит каждую строку θ на её среднее по φ:
                      grid_norm[i, :] = grid[i, :] / mean(grid[i, :])
                    Убирает cos²θ-тренд (систематическое убывание треков
                    с ростом θ), оставляя только φ-аномалии. Применять когда
                    задача — поиск аномалий именно в азимутальном распределении.
                    По нашим данным: тренд 4.7× → рекомендуется как второй шаг
                    после per_detector.

  "none"          — без нормализации. Только для отладки или если нормализация
                    выполняется снаружи этого скрипта.

─────────────────────────────────────────────────────────────────────────────
transform_mode — трансформация значений
─────────────────────────────────────────────────────────────────────────────

  Сырые значения N_tracks имеют сильно скошенное распределение:

    raw:   skew = +1.53,  CV = 94.6%  — правосторонняя асимметрия
    log1p: skew = −0.51,  CV = 20.3%  — близко к нормальному ✓
    sqrt:  skew = +0.68,  CV = 48.9%  — промежуточный результат

  Большинство нейронных сетей работают лучше с симметричным распределением
  и меньшей дисперсией. Трансформация применяется после нормализации
  и после винсоризации.

  "log1p"  —  log(1 + N)
              Наиболее агрессивное сжатие хвоста. Рекомендуется.
              Обратное преобразование: expm1(x) = e^x − 1

  "sqrt"   —  √N
              Мягче log1p. Подходит если нужно сохранить более
              широкий динамический диапазон выходных значений.
              Обратное преобразование: x²

  "none"   —  без трансформации.
              Подходит для ансамблевых методов (Random Forest, XGBoost),
              нечувствительных к масштабу входных данных.
              Обратное преобразование: не требуется

  Обратное преобразование применяется автоматически:
      from muon_preprocessing import inverse_transform
      n_tracks = inverse_transform(model_output, mode="log1p")

─────────────────────────────────────────────────────────────────────────────
winsorize_pct — винсоризация (обрезка выбросов)
─────────────────────────────────────────────────────────────────────────────

  В анализе аномалий выявлены единичные бины с аномально высокими
  значениями (z > 4σ) у детекторов 12, 14, 17. Винсоризация заменяет
  все значения выше заданного перцентиля на пороговое значение.

  Применяется ПОСЛЕ нормализации и ДО трансформации — это важно,
  потому что порог определяется в нормированном пространстве.

  99.5  — обрезать верхние 0.5% значений (по умолчанию)
          При per_detector нормализации пороговое значение ≈ 0.0033
          (соответствует ~454 сырым трекам)
  99.9  — обрезать только самые экстремальные выбросы (мягче)
  None  — не применять (для ансамблевых методов, устойчивых к выбросам)

─────────────────────────────────────────────────────────────────────────────
EFF_CORR_MAX — порог надёжности поправки (в config.py)
─────────────────────────────────────────────────────────────────────────────

  Если поправочный коэффициент для θ-бина превышает EFF_CORR_MAX,
  этот бин считается ненадёжным и обнуляется.

  2.0  — стандартный порог (по умолчанию)
         При нашей установке исключает θ = 58° (поправка 3.4×)
  1.5  — строже (исключит также θ = 60°, поправка 1.65×)
  3.0  — мягче (оставит θ = 58° с поправкой 3.4×, не рекомендуется)

─────────────────────────────────────────────────────────────────────────────
ИТОГОВЫЙ ПАЙПЛАЙН
─────────────────────────────────────────────────────────────────────────────

  Tracks_DistrOutput_N.dat
          │
          ▼  filter_working_range()        θ ∈ [58°, 90°] → сетка (17×180)
     grid_raw
          │
          ▼  apply_efficiency_correction() ÷ поправка; θ=58° → 0
     grid_eff
          │
          ▼  normalize()                   ÷ сумма детектора (per_detector)
     grid_norm
          │
          ▼  winsorize()                   обрезка верхних 0.5%
          │
          ▼  transform()                   log(1 + N)
       grid                           ← финальный вход в модель (17×180)
          │
          ▼  save_preprocessed()
  npl4_2.0Grad_preprocessed.npz
      ├── grids       (15, 17, 180)   финал
      ├── grids_raw   (15, 17, 180)   сырые (для сравнения)
      ├── grids_eff   (15, 17, 180)   после поправки
      ├── theta       (17,)
      ├── phi         (180,)
      └── detectors   (15,)

─────────────────────────────────────────────────────────────────────────────
РЕКОМЕНДУЕМЫЕ КОНФИГУРАЦИИ
─────────────────────────────────────────────────────────────────────────────

  Основной эксперимент (ML-обучение):
    npl="npl4", binning="2.0Grad", normalization="per_detector",
    transform_mode="log1p", apply_eff_correction=True, winsorize_pct=99.5

  Эталон для валидации (чистые данные):
    npl="npl6", binning="2.0Grad" — остальные параметры те же

  Поиск φ-аномалий (убрать θ-тренд):
    normalization="angular" — применить после per_detector нормализации

  Ансамблевые методы (деревья, RF):
    transform_mode="none", winsorize_pct=None — не нужны

=============================================================================
"""