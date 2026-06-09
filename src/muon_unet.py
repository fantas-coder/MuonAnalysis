"""
=============================================================
  Мюонография -- U-Net: архитектура и обучение
=============================================================

Двухэтапный пайплайн обучения:
  Этап 1 -- Pretraining на синтетических данных (muon_synth.py)
  Этап 2 -- Fine-tuning на реальных парах npl4 -> npl6

Архитектурные особенности под наши данные:
  - Асимметричный пулинг (сетка 17×180, не квадратная)
  - Circular padding по оси φ (азимут периодичен)
  - Взвешенная MSE (92% нулевых бинов)
  - Skip connections на каждом уровне

Зависимости:
  pip install torch numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_ROOT, OUTPUT_DIR, PREPROC_DIR, GOOD_DETS,
    THETA_MIN, THETA_MAX, BASE_CHANNELS, DEPTH,
    PRETRAIN_EPOCHS, FINETUNE_EPOCHS, PRETRAIN_LR, FINETUNE_LR,
    BATCH_SIZE, WEIGHT_DECAY, NONZERO_WEIGHT, TRAIN_DETS, VAL_DETS, TEST_DETS
)
from muon_preprocessing import load_preprocessed, inverse_transform
from muon_synth import load_dataset, SYNTH_DIR
from muon_smoothing import smooth, mse as np_mse, ssim as np_ssim

# Куда сохранять модели
MODELS_DIR = DATA_ROOT / "models"

# Папка для сохранения графиков
OUTPUT_DIR = OUTPUT_DIR / "muon_unet"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# АРХИТЕКТУРА U-NET
# ─────────────────────────────────────────────────────────────────────────────

class CircularPad2d(nn.Module):
    """
    Круговое дополнение по оси φ (dim=3) и нулевое по оси θ (dim=2).

    Необходимо потому что φ ∈ [0°, 360°) является периодическим:
    левый и правый края сетки физически соединены. Стандартный zero
    padding создал бы ложный перепад на границе.
    """
    def __init__(self, pad: int = 1):
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Круговое дополнение по φ (ось 3)
        x = torch.cat([x[:, :, :, -self.pad:], x, x[:, :, :, :self.pad]], dim=3)
        # Нулевое дополнение по θ (ось 2)
        x = nn.functional.pad(x, (0, 0, self.pad, self.pad))
        return x


class ConvBlock(nn.Module):
    """
    Базовый блок: CircularPad -> Conv2d -> BatchNorm -> ReLU (×2).

    Два последовательных свёрточных слоя с нормализацией -- стандарт U-Net.
    Circular padding гарантирует физически корректную обработку φ-края.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            CircularPad2d(1),
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CircularPad2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MuonUNet(nn.Module):
    """
    U-Net для восстановления угловых распределений мюонного потока.

    Вход:  (B, 1, T, P) = (batch, 1, 17, 180)
    Выход: (B, 1, T, P) = (batch, 1, 17, 180)

    Архитектурные решения:
      - CircularPad2d вместо zero padding по φ
      - Асимметричный пулинг (1×2) на нижних уровнях чтобы не потерять
        θ-размерность (всего 17 бинов)
      - Skip connections через конкатенацию (не сложение)
      - Финальный sigmoid для нормировки выхода в [0, 1]

    Параметры:
        base_ch -- число каналов на первом уровне (удваивается с глубиной)
        depth   -- число уровней энкодера/декодера (1–4)
    """
    def __init__(self, base_ch: int = BASE_CHANNELS, depth: int = DEPTH):
        super().__init__()
        self.depth = depth
        chs = [base_ch * (2 ** i) for i in range(depth + 1)]

        # Энкодер
        self.enc_blocks = nn.ModuleList()
        self.pools      = nn.ModuleList()

        in_ch = 1
        for i, out_ch in enumerate(chs[:-1]):
            self.enc_blocks.append(ConvBlock(in_ch, out_ch))
            # Асимметричный пулинг на нижних уровнях:
            # θ-ось мала (17 бинов), не сжимаем её после первого уровня
            pool_kernel = (2, 2) if i == 0 else (1, 2)
            self.pools.append(nn.MaxPool2d(kernel_size=pool_kernel))
            in_ch = out_ch

        # Бутылочное горлышко
        self.bottleneck = ConvBlock(chs[-2], chs[-1])

        # Декодер: используем F.interpolate до точного размера skip,
        # затем Conv2d для смены числа каналов.
        # Это решает проблему нечётных размерностей (17 -> 8 -> 17, а не 16).
        self.up_convs   = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in range(depth - 1, -1, -1):
            # 1×1 свёртка для уменьшения каналов после интерполяции
            self.up_convs.append(
                nn.Conv2d(chs[i + 1], chs[i], kernel_size=1)
            )
            self.dec_blocks.append(ConvBlock(chs[i] * 2, chs[i]))

        # Финальная свёртка 1×1
        self.final_conv = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor, mask_output: bool = True) -> torch.Tensor:
        """
        Энкодер: сохраняем карты признаков для skip connections
        Запоминаем маску ненулевых бинов входа (видимый конус)

        :param x: вход (B, 1, T, P)
        :param mask_output: если True, обнуляет выход там, где вход нулевой.
                            Это сохраняет геометрическую маску детектора:
                            модель не предсказывает треки вне видимого конуса.
        """
        # Запоминаем маску ненулевых бинов входа (видимый конус)
        input_mask = (x > 0).float()

        # Энкодер: сохраняем карты признаков для skip connections
        skips = []
        h = x
        for enc, pool in zip(self.enc_blocks, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)

        # Бутылочное горлышко
        h = self.bottleneck(h)

        # Декодер: интерполяция до точного размера skip + conv + concat
        for up_conv, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(skips)):
            h = nn.functional.interpolate(
                h, size=skip.shape[2:], mode='bilinear', align_corners=False
            )
            h = up_conv(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        out = self.final_conv(h)

        # Маскируем выход: вне видимого конуса (где вход = 0) выход тоже = 0.
        # Это ключевое ограничение — модель сглаживает ВНУТРИ конуса,
        # но не «размазывает» сигнал в геометрически недоступные зоны.
        if mask_output:
            out = out * input_mask

        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────────────────────────────
def normalize_per_sample(grid: np.ndarray) -> np.ndarray:
    """
    Нормирует образец на свой максимум → значения в [0, 1].

    Это КЛЮЧЕВОЙ шаг для совместимости синтетики и реальных данных:
    синтетика генерируется в диапазоне ~[0, 1.5], а реальные данные после
    per_detector нормализации имеют значения ~[0, 0.004]. Без приведения
    к общему масштабу модель, обученная на синтетике, видит реальные
    данные как сплошной ноль.

    Нормировка на максимум делает обе области данных сравнимыми и не
    меняет структуру (форму конуса, относительные интенсивности).
    """
    g = grid.astype(np.float32)
    gmax = g.max()
    return g / gmax if gmax > 0 else g

class SynthDataset(Dataset):
    """
    PyTorch Dataset для синтетических пар (X, Y).
    Загружает данные из .npz файла, созданного muon_synth.py.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        # Нормируем каждый образец на максимум входа X (и тем же делителем Y),
        # чтобы сохранить относительный масштаб X↔Y внутри пары
        X_norm = np.zeros_like(X, dtype=np.float32)
        Y_norm = np.zeros_like(Y, dtype=np.float32)
        for i in range(len(X)):
            xmax = X[i].max()
            if xmax > 0:
                X_norm[i] = X[i] / xmax
                Y_norm[i] = Y[i] / xmax
        self.X = torch.from_numpy(X_norm).unsqueeze(1)
        self.Y = torch.from_numpy(Y_norm).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class RealDataset(Dataset):
    """
    PyTorch Dataset для реальных пар (npl4, npl6).

    Каждый образец -- один детектор. Если npl6 недоступен,
    датасет использует npl4 -> npl4 (автоэнкодер).
    """
    def __init__(
        self,
        det_indices: list[int],
        npl_input:   str = "npl4",
        npl_target:  str = "npl6",
        binning:     str = "2.0Grad",
        augment:     bool = False,
    ):
        self.augment = augment

        try:
            data_in  = load_preprocessed(npl_input,  binning)
            data_tgt = load_preprocessed(npl_target, binning)
            use_real_gt = True
        except FileNotFoundError:
            print(f"  WARNING!  {npl_target} не найден -- используем {npl_input} -> {npl_input}")
            data_in  = load_preprocessed(npl_input, binning)
            data_tgt = data_in
            use_real_gt = False

        X_all = data_in["grids"]
        Y_all = data_tgt["grids"]
        dets  = list(data_in["detectors"])

        # Отбираем нужные детекторы и нормируем каждый на максимум входа
        X_list, Y_list = [], []
        for det in det_indices:
            if det in dets:
                idx  = dets.index(det)
                xmax = X_all[idx].max()
                if xmax > 0:
                    X_list.append(X_all[idx] / xmax)
                    Y_list.append(Y_all[idx] / xmax)

        self.X = torch.from_numpy(
            np.stack(X_list).astype(np.float32)
        ).unsqueeze(1)
        self.Y = torch.from_numpy(
            np.stack(Y_list).astype(np.float32)
        ).unsqueeze(1)
        self.use_real_gt = use_real_gt

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.augment:
            # 1. Случайный сдвиг по φ (циклический) -- φ периодичен
            shift = torch.randint(0, x.shape[-1], (1,)).item()
            x = torch.roll(x, shift, dims=-1)
            y = torch.roll(y, shift, dims=-1)

            # 2. Случайное масштабирование интенсивности (±15%).
            # Искусственно расширяет крошечную реальную выборку (10 детекторов).
            # Один множитель на x и y -- сохраняет соотношение вход/эталон.
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.3   # [0.85, 1.15]
            x = x * scale
            y = y * scale
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# ФУНКЦИЯ ПОТЕРЬ
# ─────────────────────────────────────────────────────────────────────────────

class WeightedMSELoss(nn.Module):
    """
    Взвешенная MSE: ненулевые бины получают больший вес.

    Мотивация: при ~92% нулевых бинов стандартная MSE доминируется
    нулями и не обучает модель правильно предсказывать ненулевые
    значения, несущие весь физический сигнал.

    L = (1/N) Σ w_ij · (ŷ_ij - y_ij)²
    w_ij = nonzero_weight если y_ij > 0, иначе 1.0
    """
    def __init__(self, nonzero_weight: float = NONZERO_WEIGHT):
        super().__init__()
        self.nonzero_weight = nonzero_weight

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.where(
            target > 0,
            torch.full_like(target, self.nonzero_weight),
            torch.ones_like(target),
        )
        return (weights * (pred - target) ** 2).mean()


class SSIMLoss(nn.Module):
    """
    Дифференцируемая функция потерь на основе SSIM: L = 1 − SSIM.

    Использует глобальные статистики по изображению (как в numpy-метрике
    ssim из muon_smoothing), что согласуется с метрикой оценки.

    Минимизация 1−SSIM напрямую оптимизирует структурное сходство —
    ту самую метрику, по которой мы сравниваемся с классическими методами.
    """

    def __init__(self, k1: float = 0.01, k2: float = 0.03):
        super().__init__()
        self.k1 = k1
        self.k2 = k2

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
    ) -> torch.Tensor:
        # Считаем SSIM по каждому образцу в батче, затем усредняем
        b = pred.shape[0]
        p = pred.view(b, -1)
        t = target.view(b, -1)

        # Диапазон данных по каждому образцу
        data_range = t.max(dim=1).values - t.min(dim=1).values
        data_range = torch.clamp(data_range, min=1e-6)
        c1 = (self.k1 * data_range) ** 2
        c2 = (self.k2 * data_range) ** 2

        mu_p = p.mean(dim=1)
        mu_t = t.mean(dim=1)
        var_p = p.var(dim=1, unbiased=False)
        var_t = t.var(dim=1, unbiased=False)
        cov = ((p - mu_p.unsqueeze(1)) * (t - mu_t.unsqueeze(1))).mean(dim=1)

        ssim =  (
                    (2 * mu_p * mu_t + c1) * (2 * cov + c2)
                ) / (
                    (mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2)
                )
        return (1 - ssim).mean()


class CombinedLoss(nn.Module):
    """
    Комбинированная функция потерь: взвешенная MSE + α·(1−SSIM).

    MSE обеспечивает точность по значениям, SSIM-член — сохранение
    структуры. Параметр alpha балансирует вклад.

    L = WeightedMSE + alpha · SSIMLoss
    """
    def __init__(
        self,
        nonzero_weight: float = NONZERO_WEIGHT,
        alpha:          float = 0.5,
    ):
        super().__init__()
        self.mse  = WeightedMSELoss(nonzero_weight)
        self.ssim = SSIMLoss()
        self.alpha = alpha

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.mse(pred, target) + self.alpha * self.ssim(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# ЦИКЛ ОБУЧЕНИЯ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainHistory:
    """История обучения: потери и метрики по эпохам."""
    train_loss: list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    val_mse:    list[float] = field(default_factory=list)
    val_ssim:   list[float] = field(default_factory=list)
    stage:      str = ""


def train_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
) -> float:
    """Один эпоха обучения. Возвращает среднюю потерю."""
    model.train()
    total_loss = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, float]:
    """
    Оценка на val. Возвращает (val_loss, val_mse, val_ssim).
    MSE и SSIM считаются в numpy на оригинальном масштабе.
    """
    model.eval()
    total_loss, mse_sum, ssim_sum, n = 0.0, 0.0, 0.0, 0

    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        total_loss += criterion(pred, Y).item()

        pred_np = pred.squeeze(1).cpu().numpy()
        y_np    = Y.squeeze(1).cpu().numpy()
        for p, t in zip(pred_np, y_np):
            mse_sum  += np_mse(p, t)
            ssim_sum += np_ssim(p, t)
            n += 1

    return (
        total_loss / len(loader),
        mse_sum / max(n, 1),
        ssim_sum / max(n, 1),
    )

def save_checkpoint(
    path:      Path,
    model:     nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch:     int,
    history:   "TrainHistory",
    best_loss: float,
) -> None:
    """
    Сохраняет полное состояние обучения для возобновления:
    веса модели, состояние оптимайзера, шедулера, историю метрик.
    """
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history":         history.__dict__,
        "best_loss":       best_loss,
    }, path)


def load_checkpoint(
    path:      Path,
    model:     nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device:    torch.device,
) -> tuple[int, "TrainHistory", float]:
    """
    Загружает чекпоинт. Возвращает (start_epoch, history, best_loss).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    history = TrainHistory(**ckpt["history"])
    print(f"  Чекпоинт: эпоха {ckpt['epoch']}, best_loss={ckpt['best_loss']:.5f}")
    return ckpt["epoch"] + 1, history, ckpt["best_loss"]


def train(
    model:            nn.Module,
    train_loader:     DataLoader,
    val_loader:       DataLoader,
    epochs:           int,
    lr:               float,
    device:           torch.device,
    stage:            str  = "train",
    save_best:        bool = True,
    model_path:       Path | None = None,
    checkpoint_every: int  = 10,
    resume_from:      Path | None = None,
    use_ssim_loss:    bool = False,
    ssim_alpha:       float = 0.2,
) -> TrainHistory:
    """
    Полный цикл обучения с сохранением лучшей модели по val_loss.

    :param model: наша модель U-Net
    :param train_loader: загрузчик train батчей
    :param val_loader: загрузчик val батчей
    :param epochs: число эпох
    :param lr: learning rate
    :param device: cpu или cuda
    :param stage: метка для логирования ('pretrain' / 'finetune')
    :param save_best: сохранять ли лучшую модель по val_loss
    :param model_path: путь для сохранения модели
    :param checkpoint_every: чекпоинт каждые N эпох (полное состояние)
    :param resume_from: Path к чекпоинту для возобновления (None = с нуля)
    :param use_ssim_loss: использовать ли комбинированный лосс mse + ssim
    :param ssim_alpha: параметр для комбинированного лосса
    """
    if use_ssim_loss:
        criterion = CombinedLoss(alpha=ssim_alpha)
        print(f"  Функция потерь: CombinedLoss (MSE + {ssim_alpha}*SSIM)")
    else:
        criterion = WeightedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    # При файнтюнинге замораживаем энкодер на первые FINETUNE_FREEZE_EPOCHS эпох.
    # Это защищает представления, выученные при предобучении на синтетике,
    # от разрушения малой реальной выборкой (катастрофическое забывание).
    from config import FINETUNE_FREEZE_EPOCHS
    freeze_epochs = FINETUNE_FREEZE_EPOCHS if stage == "finetune" else 0

    def set_encoder_frozen(frozen: bool):
        for p in model.enc_blocks.parameters():
            p.requires_grad = not frozen
        for p in model.bottleneck.parameters():
            p.requires_grad = not frozen

    if freeze_epochs > 0:
        set_encoder_frozen(True)
        print(f"  Энкодер заморожен на первые {freeze_epochs} эпох")

    start_epoch = 1
    best_loss = float("inf")
    history = TrainHistory(stage=stage)

    ckpt_path = (
        model_path.parent / f"ckpt_{stage}.pt"
        if model_path else None
    )

    # Возобновление
    if resume_from and resume_from.exists():
        print(f"  Возобновление: {resume_from.name}")
        start_epoch, history, best_loss = load_checkpoint(
            resume_from, model, optimizer, scheduler, device
        )
        print(f"  Продолжаем с эпохи {start_epoch}/{epochs}")
    elif resume_from:
        print(f"  Чекпоинт не найден — стартуем с нуля")

    for epoch in range(start_epoch, epochs + 1):
        # Размораживаем энкодер после freeze_epochs эпох
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            set_encoder_frozen(False)
            print(f"  [{stage}] ep {epoch}: энкодер разморожен")

        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_mse, v_ssim = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(v_loss)

        history.train_loss.append(t_loss)
        history.val_loss.append(v_loss)
        history.val_mse.append(v_mse)
        history.val_ssim.append(v_ssim)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [{stage}] ep {epoch:3d}/{epochs} │ "
                  f"train={t_loss:.5f}  val={v_loss:.5f}  "
                  f"SSIM={v_ssim:.4f}  lr={lr_now:.2e}")

        if save_best and v_loss < best_loss:
            best_loss = v_loss
            if model_path:
                torch.save(model.state_dict(), model_path)

        if ckpt_path and epoch % checkpoint_every == 0:
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler,
                epoch, history, best_loss,
            )
            print(f"  Чекпоинт сохранён: ep {epoch}  ({ckpt_path.name})")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_test(
    model:    nn.Module,
    npl:      str = "npl4",
    binning:  str = "2.0Grad",
    device:   torch.device | None = None,
) -> dict:
    """
    Оценивает модель на тестовых детекторах и сравнивает
    с классическими методами сглаживания.

    ВАЖНО: метрики считаются относительно ЧИСТОГО ЭТАЛОНА (npl6),
    а не относительно входа (npl4). Это измеряет качество восстановления
    истинного сигнала, а не степень "неизменности" входа. Метод, который
    ничего не делает, получил бы SSIM=1 против входа, но низкий против эталона.

    Возвращает dict с MSE и SSIM для каждого метода и детектора.
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    data = load_preprocessed(npl, binning)
    grids = data["grids"]
    dets = list(data["detectors"])

    # Загружаем чистый эталон npl6 для честного сравнения
    try:
        data_gt = load_preprocessed("npl6", binning)
        grids_gt = data_gt["grids"]
        dets_gt = list(data_gt["detectors"])
        has_gt = True
    except FileNotFoundError:
        print("  WARNING! npl6 не найден — сравнение идёт против входа npl4")
        grids_gt = grids
        dets_gt = dets
        has_gt = False

    results = {m: {"mse": [], "ssim": []} for m in
               ["unet", "gaussian", "gradient", "wavelet"]}

    for det in TEST_DETS:
        if det not in dets:
            continue
        idx = dets.index(det)
        gmax = grids[idx].max()
        x_np = (grids[idx] / gmax if gmax > 0 else grids[idx]).astype(np.float32)

        # Эталон: тот же детектор из npl6, нормированный на свой максимум
        if has_gt and det in dets_gt:
            idx_gt = dets_gt.index(det)
            gt_max = grids_gt[idx_gt].max()
            gt = (grids_gt[idx_gt] / gt_max if gt_max > 0
                  else grids_gt[idx_gt]).astype(np.float32)
        else:
            gt = x_np  # fallback

        # U-Net: сравниваем выход с эталоном GT
        x_t = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(x_t).squeeze().cpu().numpy()
        results["unet"]["mse"].append(np_mse(gt, pred))
        results["unet"]["ssim"].append(np_ssim(gt, pred))

        # Классические методы: сравниваем выход с эталоном GT
        for method, kwargs in [
            ("gaussian", {"sigma_theta": 1.0, "sigma_phi": 0.5}),
            ("gradient", {"window_size": 3}),
            ("wavelet", {"levels": 4}),
        ]:
            sm = smooth(x_np, method=method, **kwargs)
            results[method]["mse"].append(np_mse(gt, sm))
            results[method]["ssim"].append(np_ssim(gt, sm))

    # Средние значения
    print(f"\n{'Метод':20s}  {'MSE (ср.)':>12s}  {'SSIM (ср.)':>12s}")
    print("─" * 48)
    for method, vals in results.items():
        if vals["mse"]:
            m = np.mean(vals["mse"])
            s = np.mean(vals["ssim"])
            marker = " ←" if method == "unet" else ""
            print(f"  {method:18s}  {m:12.6f}  {s:12.4f}{marker}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(
    histories: list[TrainHistory],
    save: bool = True,
) -> plt.Figure:
    """График потерь и метрик по эпохам для обоих этапов."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("История обучения U-Net", fontsize=13, fontweight="bold")

    colors = {"pretrain": "#2196F3", "finetune": "#FF9800"}
    offset = 0

    for hist in histories:
        col = colors.get(hist.stage, "gray")
        ep  = range(offset + 1, offset + len(hist.train_loss) + 1)
        axes[0].plot(ep, hist.train_loss, color=col, label=f"{hist.stage} train")
        axes[0].plot(ep, hist.val_loss,   color=col, ls="--",
                     label=f"{hist.stage} val")
        axes[1].plot(ep, hist.val_mse,    color=col, label=hist.stage)
        axes[2].plot(ep, hist.val_ssim,   color=col, label=hist.stage)
        offset += len(hist.train_loss)

    # Граница между этапами
    if len(histories) == 2:
        n1 = len(histories[0].train_loss)
        for ax in axes:
            ax.axvline(n1, color="gray", ls=":", lw=1.5, label="fine-tune ->")

    axes[0].set_title("Функция потерь (WeightedMSE)", fontweight="bold")
    axes[1].set_title("Val MSE", fontweight="bold")
    axes[2].set_title("Val SSIM", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Эпоха"); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        out = OUTPUT_DIR / "fig_unet_history.png"
        fig.savefig(out, bbox_inches="tight", dpi=130)
        print(f"  Сохранён: {out.name}")
        plt.close(fig)
    return fig


@torch.no_grad()
def plot_predictions(
    model:   nn.Module,
    npl:     str = "npl4",
    binning: str = "2.0Grad",
    device:  torch.device | None = None,
    save:    bool = True,
) -> plt.Figure | None:
    """
    Для каждого тестового детектора: вход / предсказание U-Net / GT.
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    data    = load_preprocessed(npl, binning)
    grids   = data["grids"]
    dets    = list(data["detectors"])
    theta   = data["theta"]
    phi     = data["phi"]

    test_dets = [d for d in TEST_DETS if d in dets]
    n = len(test_dets)
    if n == 0:
        print("Нет тестовых детекторов в данных")
        return None

    fig = plt.figure(figsize=(18, 5 * n))
    fig.suptitle(f"Предсказания U-Net -- {npl}/{binning}",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.45, wspace=0.3)

    for row, det in enumerate(test_dets):
        idx  = dets.index(det)
        gmax = grids[idx].max()
        x_np = (grids[idx] / gmax if gmax > 0 else grids[idx]).astype(np.float32)

        x_t  = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(x_t).squeeze().cpu().numpy()

        for col, (title, grid) in enumerate([
            ("Вход (npl4)",      x_np),
            ("U-Net (выход)",    pred),
            ("Разность",         pred - x_np),
        ]):
            ax = fig.add_subplot(gs[row, col])
            cmap = "inferno" if col < 2 else "RdBu_r"
            vmax = np.abs(grid).max() or 1e-6
            kw = dict(vmin=-vmax, vmax=vmax) if col == 2 else {}
            im = ax.pcolormesh(phi, theta, grid, cmap=cmap,
                               shading="auto", **kw)
            plt.colorbar(im, ax=ax, shrink=0.85)
            ax.set_xlabel("φ (°)"); ax.set_ylabel("θ (°)")
            m = np_mse(x_np, pred); s = np_ssim(x_np, pred)
            ax.set_title(
                f"дет.#{det} -- {title}" +
                (f"\nMSE={m:.5f}  SSIM={s:.4f}" if col == 1 else ""),
                fontsize=9, fontweight="bold",
            )

    fig.subplots_adjust(hspace=0.45, wspace=0.3)
    if save:
        out = OUTPUT_DIR / f"fig_unet_predictions_{npl}.png"
        fig.savefig(out, bbox_inches="tight", dpi=130)
        print(f"  Сохранён: {out.name}")
        plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def main(
    stage:            str   = "all",
    npl:              str   = "npl4",
    binning:          str   = "2.0Grad",
    pretrain_epochs:  int   = PRETRAIN_EPOCHS,
    finetune_epochs:  int   = FINETUNE_EPOCHS,
    batch_size:       int   = BATCH_SIZE,
    base_channels:    int   = BASE_CHANNELS,
    depth:            int   = DEPTH,
    checkpoint_every: int   = 10,
    resume_pretrain:  bool  = False,
    resume_finetune:  bool  = False,
    save_plots:       bool  = True,
    use_ssim_loss:    bool  = False,
):
    """
    Двухэтапный пайплайн обучения U-Net.

    :param stage: какой этап запустить "pretrain" | "finetune" | "all" | "eval"
    :param npl: уровень качества трека для реальных данных
    :param binning: биннинг
    :param pretrain_epochs: эпохи предобучения на синтетике
    :param finetune_epochs: эпохи файнтюнинга на реальных данных
    :param batch_size: размер батча
    :param base_channels: базовое число каналов U-Net
    :param depth: глубина U-Net (число уровней)
    :param checkpoint_every: сохранять чекпоинт каждые N эпох
    :param resume_pretrain: True = продолжить pretrain с чекпоинта
    :param resume_finetune: True = продолжить finetune с чекпоинта
    :param save_plots: сохранять ли графики
    :param use_ssim_loss: использовать ли комбинированный loss: Mse + SSIM
    """
    print("=" * 60)
    print("  Мюонография -- обучение U-Net")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Устройство : {device}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pretrain_path = MODELS_DIR / f"unet_pretrained_{base_channels}ch.pt"
    finetune_path = MODELS_DIR / f"unet_finetuned_{npl}_{binning}.pt"
    ckpt_pretrain = MODELS_DIR / "ckpt_pretrain.pt"
    ckpt_finetune = MODELS_DIR / "ckpt_finetune.pt"

    model = MuonUNet(base_ch=base_channels, depth=depth).to(device)
    n_params = model.count_parameters()
    print(f"  Параметров : {n_params:,}")
    print(f"  Архитектура: depth={depth}, base_ch={base_channels}")

    histories = []

    # ── ЭТАП 1: Предобучение на синтетике ────────────────────────────────────
    if stage in ("pretrain", "all"):
        print("\n─── Этап 1: Предобучение на синтетических данных ───────")

        try:
            X_train, Y_train = load_dataset("synth_train.npz")
            X_val, Y_val = load_dataset("synth_val.npz")
        except FileNotFoundError:
            print("  Синтетика не найдена -- генерируем...")
            from muon_synth import main as synth_main
            synth_main(n_samples=2000, n_val=200, show_samples=False,
                       show_noise=False, save=True)
            X_train, Y_train = load_dataset("synth_train.npz")
            X_val, Y_val = load_dataset("synth_val.npz")

        train_ds = SynthDataset(X_train, Y_train)
        val_ds = SynthDataset(X_val, Y_val)

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                shuffle=False, num_workers=0)

        print(f"  Train: {len(train_ds)} обр.  Val: {len(val_ds)} обр.")

        hist1 = train(
            model, train_loader, val_loader,
            epochs=pretrain_epochs, lr=PRETRAIN_LR,
            device=device, stage="pretrain",
            save_best=True, model_path=pretrain_path,
            checkpoint_every=checkpoint_every,
            resume_from=ckpt_pretrain if resume_pretrain else None,
            use_ssim_loss=use_ssim_loss,
        )
        histories.append(hist1)
        print(f"  Лучший val SSIM: {max(hist1.val_ssim):.4f}")

    # ── ЭТАП 2: Файнтюнинг на реальных данных ────────────────────────────────
    if stage in ("finetune", "all"):
        print("\n─── Этап 2: Файнтюнинг на реальных данных ─────────────")

        # Проверяем наличие предобработанного npl6 (целевые данные).
        # Если нет — запускаем препроцессинг автоматически.
        npl6_path = PREPROC_DIR / "npl6_2.0Grad_preprocessed.npz"
        if not npl6_path.exists():
            print("  npl6 не предобработан — запускаем препроцессинг...")
            try:
                from muon_preprocessing import main as preproc_main
                preproc_main(npl="npl6", binning=binning,
                             save_npz=True, save_plots=False)
            except Exception as e:
                print(f"  WARNING! Не удалось предобработать npl6: {e}")
                print("  Файнтюнинг будет npl4→npl4 (бессмысленно).")
                print("  Убедись что данные npl6 доступны в DATA_ROOT.")

        if pretrain_path.exists() and stage != "finetune" and not resume_finetune:
            pass  # веса уже в модели из этапа 1
        elif ckpt_finetune.exists() and resume_finetune:
            pass  # восстановим внутри train()
        elif pretrain_path.exists():
            model.load_state_dict(torch.load(pretrain_path, map_location=device))
            print(f"  Загружены веса: {pretrain_path.name}")
        else:
            print("  WARNING!  Предобученные веса не найдены -- файнтюнинг с нуля")

        train_ds = RealDataset(
            TRAIN_DETS, npl_input=npl, npl_target="npl6",
            binning=binning, augment=True,
        )
        val_ds = RealDataset(
            VAL_DETS, npl_input=npl, npl_target="npl6",
            binning=binning, augment=False,
        )

        if len(train_ds) == 0:
            print(f"  ERROR!  Нет данных для train детекторов {TRAIN_DETS}")
            print("     Проверь что данные предобработаны: python muon_preprocessing.py")
            return None

        train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)),
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)),
                                shuffle=False, num_workers=0)

        gt_label = "npl6" if train_ds.use_real_gt else "npl4 (авт.)"
        print(f"  Train детекторы: {TRAIN_DETS} ({len(train_ds)} обр.)")
        print(f"  Val детекторы  : {VAL_DETS}   ({len(val_ds)} обр.)")
        print(f"  GT (эталон)    : {gt_label}")

        hist2 = train(
            model, train_loader, val_loader,
            epochs=finetune_epochs, lr=FINETUNE_LR,
            device=device, stage="finetune",
            save_best=True, model_path=finetune_path,
            checkpoint_every=checkpoint_every,
            resume_from=ckpt_finetune if resume_finetune else None,
            use_ssim_loss=use_ssim_loss,
        )
        histories.append(hist2)
        print(f"  Лучший val SSIM: {max(hist2.val_ssim):.4f}")

    # ── Оценка ────────────────────────────────────────────────────────────────
    if stage in ("eval", "all"):
        print("\n─── Оценка на тестовых детекторах ─────────────────────")

        best_path = finetune_path if finetune_path.exists() else pretrain_path
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
            print(f"  Модель: {best_path.name}")

        evaluate_on_test(model, npl=npl, binning=binning, device=device)

        if save_plots:
            plot_predictions(model, npl=npl, binning=binning,
                             device=device, save=True)

    # ── Графики обучения ───────────────────────────────────────────────────
    if histories and save_plots:
        print("\n─── Графики ────────────────────────────────────────────")
        plot_history(histories, save=True)

    print("\nSUCCESS! Готово!")
    print(f"  Модели  : {MODELS_DIR}")
    print(f"  Графики : {OUTPUT_DIR}")

    plt.show()
    return model


if __name__ == "__main__":
    main(
        stage="all",
        npl="npl4",
        binning="2.0Grad",
        pretrain_epochs=PRETRAIN_EPOCHS,
        finetune_epochs=FINETUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        save_plots=True,
        use_ssim_loss=True,
    )