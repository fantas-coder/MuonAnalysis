"""
config.py -- Центральная конфигурация проекта мюонографии
=========================================================

Все пути, константы и общие вспомогательные функции хранятся здесь.
Остальные скрипты импортируют нужное из этого файла.

Быстрый старт:
    1. Укажи DATA_ROOT -- путь к папке с данными (там должны лежать npl3/, npl4/ и т.д.)
    2. При необходимости поменяй OUTPUT_DIR -- куда сохранять графики
    3. В скриптах: from config import *
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# ПУТИ
# ─────────────────────────────────────────────────────────────────────────────

# Корневая папка с данными: внутри должны лежать npl3/, npl4/, npl5/, npl6/
DATA_ROOT = Path("../data")

# Куда сохранять графики
OUTPUT_DIR = Path("../figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ПАРАМЕТРЫ ДАННЫХ
# ─────────────────────────────────────────────────────────────────────────────

# Все доступные варианты биннинга
VALID_BINNINGS = ["1.0Grad", "1.5Grad", "2.0Grad", "2.5Grad"]

# Валидация: Ожидаемое число строк в Tracks_DistrOutput для каждого биннинга
# (180/step) × (360/step)
EXPECTED_ROWS: dict[str, int] = {
    "1.0Grad": 64800,
    "1.5Grad": 28800,
    "2.0Grad": 16200,
    "2.5Grad": 10368,
}

# Детекторы
ALL_DETS  = list(range(1, 19))          # все 18 детекторов
BAD_DETS  = {8, 15, 16}                 # исключены: №8 (нет треков), №15 и №16 (слишком мало)
GOOD_DETS = [d for d in ALL_DETS if d not in BAD_DETS]

# Рабочий диапазон зенитного угла (геометрическое ограничение установки)
THETA_MIN = 58.0   # °  -- ниже этого угла треков нет
THETA_MAX = 90.0   # °  -- выше также треков нет

# ─────────────────────────────────────────────────────────────────────────────
# ПАРАМЕТРЫ АНАЛИЗА АНОМАЛИЙ
# ─────────────────────────────────────────────────────────────────────────────

IQR_K     = 3.0   # выброс если N > Q3 + IQR_K × IQR (коэффициент Тьюки)
ZSCORE_TH = 4.0   # выброс если |z| > ZSCORE_TH (количество сигма для выброса)
CROSS_TH  = 5.0   # кросс-детекторный порог: отклонение > CROSS_TH × медиана

# ─────────────────────────────────────────────────────────────────────────────
# ПАРАМЕТРЫ ПРЕДОБРАБОТКИ  (используются в muon_preprocessing.py)
# ─────────────────────────────────────────────────────────────────────────────

# Применять ли поправку на эффективность из EffCorFile_Tracks.dat.
# N_corr = ε × N_raw,  где ε = N_ожид / N_зарег ≥ 1 при больших θ_D
# (детектор недосчитывает треки при наклонном прохождении через пластины)
# True  — применять (по умолчанию, данные не скорректированы).
# False — пропустить (если поправка уже применена при сборке Tracks_DistrOutput).
APPLY_EFF_CORRECTION: bool = True

# Максимальный надёжный коэффициент поправки.
# θ-бины с коэффициентом выше порога исключаются как ненадёжные.
# По данным npl4/2.0°: θ=58° даёт коэффициент 3.4× → исключается.
EFF_CORR_MAX: float = 1.5

# Режим нормализации N_tracks.
# "per_detector" — делит на сумму треков детектора (рекомендуется, CV=39%)
# "global"       — делит на максимум среди всех детекторов
# "angular"      — делит на θ-профиль, убирает cos²θ тренд
# "none"         — без нормализации
NORMALIZATION: str = "per_detector"

# Трансформация значений N_tracks.
# "log1p" — log(1 + N), приближает к нормальному (skew: 1.53 → -0.51)
# "sqrt"  — √N, мягче log1p (skew: 1.53 → 0.68)
# "none"  — без трансформации
TRANSFORM: str = "log1p"

# Верхний перцентиль для винсоризации (обрезка выбросов).
# Применяется после нормализации и до трансформации.
# None — отключить. По данным npl4/2.0°: p99.5 ≈ 455 треков.
WINSORIZE_PERCENTILE: float | None = 99.5

# Нижняя граница N_tracks: значения ≤ порога считаются нулём.
# 0.0 — сохраняем все ненулевые значения как есть.
ZERO_THRESHOLD: float = 0.0

# Папка для сохранения предобработанных .npz файлов
PREPROC_DIR: "Path" = DATA_ROOT / "preprocessed"

# ─────────────────────────────────────────────────────────────────────────────
# СТИЛЬ ГРАФИКОВ
# ─────────────────────────────────────────────────────────────────────────────

COLORS_NPL: dict[str, str] = {
    "npl3": "#9C27B0",   # фиолетовый
    "npl4": "#2196F3",   # синий
    "npl5": "#FF9800",   # оранжевый
    "npl6": "#4CAF50",   # зелёный
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":     120,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
})

# ─────────────────────────────────────────────────────────────────────────────
# ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ
# ─────────────────────────────────────────────────────────────────────────────

def data_path(npl: str, binning: str, filename: str) -> Path:
    """Возвращает полный путь к файлу данных."""
    return DATA_ROOT / npl / binning / filename


def load_detectors(npl: str = "npl4", binning: str = "1.0Grad") -> np.ndarray:
    """
    Загружает Detectors.txt.
    Столбцы: [номер, X (м), Y (м), высота (м н.у.м.)]
    Начало координат -- детектор №9.
    """
    return np.loadtxt(data_path(npl, binning, "Detectors.txt"))


def load_input(npl: str = "npl4", binning: str = "1.0Grad") -> np.ndarray:
    """
    Загружает Input.txt.
    Столбцы: [номер, X (м), Y (м), высота (м н.у.м.), угол_поворота (°)]
    Пропускает строки-заголовки.
    """
    rows = []
    with open(data_path(npl, binning, "Input.txt")) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    pass
    return np.array(rows)


def load_efficiency(npl: str = "npl4", binning: str = "1.0Grad") -> np.ndarray:
    """
    Загружает EffCorFile_Tracks.dat.
    Столбцы:
        [0] θ_D (градусы)  -- угол в СК детектора
        [1] θ_D (радианы)
        [2] N треков в калибровочном детекторе
        [3] поправочный коэффициент эффективности
    Надёжна только при θ_D ≤ 30° (θ_Земля ≥ 60°).
    """
    return np.loadtxt(data_path(npl, binning, "EffCorFile_Tracks.dat"))


def load_tracks(npl: str, binning: str, det: int) -> np.ndarray | None:
    """
    Загружает Tracks_DistrOutput_N.dat для детектора det.
    Столбцы: [θ (°), φ (°), N_tracks]  -- уже в земных координатах.

    Возвращает None если:
        - файл не существует
        - формат аномальный (< 10 уникальных значений φ -- признак npl6/1.0Grad)
    """
    path = data_path(npl, binning, f"Tracks_DistrOutput_{det}.dat")
    if not path.exists():
        return None
    d = np.loadtxt(path)
    if len(np.unique(d[:, 1])) < 10:
        return None
    return d


def sum_all_detectors(
    npl: str,
    binning: str,
    dets: list[int] | None = None,
) -> np.ndarray | None:
    """
    Суммирует N_tracks по всем (или указанным) детекторам в единую (θ,φ)-таблицу.
    По умолчанию использует GOOD_DETS из конфига.
    """
    if dets is None:
        dets = GOOD_DETS
    acc = None
    for det in dets:
        d = load_tracks(npl, binning, det)
        if d is None:
            continue
        if acc is None:
            acc = d.copy()
        else:
            acc[:, 2] += d[:, 2]
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОБРАБОТКИ
# ─────────────────────────────────────────────────────────────────────────────

def make_grid(
    data: np.ndarray,
    theta_min: float = THETA_MIN,
    theta_max: float = THETA_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Преобразует табличные данные (θ, φ, значение) в регулярную 2D-сетку.

    :param data: массив shape (N, 3) с колонками [θ, φ, значение]
    :param theta_min: минимальный угол θ для фильтрации
    :param theta_max: максимальный угол θ для фильтрации
    :return: кортеж (phi_bins, theta_bins, grid), где:
             phi_bins - уникальные значения φ (по оси X)
             theta_bins - уникальные значения θ (по оси Y)
             grid - матрица значений shape (len(theta_bins), len(phi_bins))
    """
    sub   = data[(data[:, 0] >= theta_min) & (data[:, 0] <= theta_max)]
    t_bins = np.unique(sub[:, 0])
    p_bins = np.unique(sub[:, 1])
    grid   = np.zeros((len(t_bins), len(p_bins)))
    t_idx  = {v: i for i, v in enumerate(t_bins)}
    p_idx  = {v: i for i, v in enumerate(p_bins)}
    for row in sub:
        ti = t_idx.get(row[0])
        pi = p_idx.get(row[1])
        if ti is not None and pi is not None:
            grid[ti, pi] = row[2]
    return p_bins, t_bins, grid


def theta_marginal(
    data: np.ndarray,
    theta_min: float = THETA_MIN,
    theta_max: float = THETA_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет интегральное распределение по углу θ (суммируя значения по φ).

    :param data: массив shape (N, 3) с колонками [θ, φ, значение]
    :param theta_min: минимальный угол θ для фильтрации
    :param theta_max: максимальный угол θ для фильтрации
    :return: кортеж (theta_bins, counts), где:
             theta_bins - уникальные значения θ
             counts - сумма значений по φ для каждого θ
    """
    sub    = data[(data[:, 0] >= theta_min) & (data[:, 0] <= theta_max)]
    t_bins = np.unique(sub[:, 0])
    counts = np.array([sub[sub[:, 0] == t, 2].sum() for t in t_bins])
    return t_bins, counts


def phi_marginal(
    data: np.ndarray,
    theta_min: float = THETA_MIN,
    theta_max: float = THETA_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет интегральное распределение по углу φ (суммируя значения по θ).

    :param data: массив shape (N, 3) с колонками [θ, φ, значение]
    :param theta_min: минимальный угол θ для фильтрации
    :param theta_max: максимальный угол θ для фильтрации
    :return: кортеж (phi_bins, counts), где:
             phi_bins - уникальные значения φ
             counts - сумма значений по θ для каждого φ
    """
    sub    = data[(data[:, 0] >= theta_min) & (data[:, 0] <= theta_max)]
    p_bins = np.unique(sub[:, 1])
    counts = np.array([sub[sub[:, 1] == p, 2].sum() for p in p_bins])
    return p_bins, counts


def check_data_integrity(verbose: bool = True) -> dict:
    """
    Быстрая проверка наличия и формата всех файлов.
    Выводит таблицу статусов и возвращает словарь {(npl, binning): статус}.
    """
    status = {}
    for npl in ["npl3", "npl4", "npl5", "npl6"]:
        for binning in VALID_BINNINGS:
            folder = DATA_ROOT / npl / binning
            if not folder.exists():
                status[(npl, binning)] = "MISSING"
                continue
            files = list(folder.glob("Tracks_DistrOutput_*.dat"))
            if not files:
                status[(npl, binning)] = "NO_FILES"
                continue
            d = np.loadtxt(files[0])
            n_phi = len(np.unique(d[:, 1]))
            status[(npl, binning)] = (
                f"ANOMAL (φ={n_phi})" if n_phi < 10
                else f"OK ({len(files)} дет.)"
            )

    if verbose:
        header = f"{'':16}" + "".join(f"{b:>14}" for b in VALID_BINNINGS)
        print(header)
        for npl in ["npl3", "npl4", "npl5", "npl6"]:
            row = f"{npl:16}"
            for b in VALID_BINNINGS:
                row += f"{status.get((npl, b), '?'):>14}"
            print(row)
    return status


# ─────────────────────────────────────────────────────────────────────────────
# БЫСТРАЯ САМОДИАГНОСТИКА  (python config.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Проверка конфигурации и доступности данных")
    print("=" * 55)
    print(f"\n  DATA_ROOT  : {DATA_ROOT.resolve()}")
    print(f"  OUTPUT_DIR : {OUTPUT_DIR.resolve()}")
    print(f"  GOOD_DETS  : {GOOD_DETS}")
    print(f"  BAD_DETS   : {sorted(BAD_DETS)}")
    print(f"  θ диапазон : {THETA_MIN}° – {THETA_MAX}°\n")

    if not DATA_ROOT.exists():
        print(f"    ERROR! Папка {DATA_ROOT} не найдена!")
        print("     Укажи правильный путь в переменной DATA_ROOT.")
    else:
        print("─ Проверка файлов " + "─" * 36)
        check_data_integrity(verbose=True)
        print("\nSUCCESS! Конфигурация загружена успешно.")