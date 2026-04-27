import datetime
import copy
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

INPUT_SIZE = 17
OUTPUT_SIZE = 13
NUM_EPOCHS = 100
BATCH_SIZE = 64
NUM_LAYERS = 4
NUM_WORKERS = 12
PREFETCH_FACTOR = 2
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-5
D_MODEL = 128
NHEAD = 8
DROPOUT = 0.1
PRED_LEN = 1
USE_AMP = True

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
if device == "cuda":
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
BASE_DIR = Path(__file__).resolve().parent


class StandardNormalizer:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> None:
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={data.shape}")
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std = (data.std(axis=0) + 1e-8).astype(np.float32)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer is not fitted")
        return ((data - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer is not fitted")
        return (data * self.std + self.mean).astype(np.float32)


def save_normalizers(path: Path, inp_norm: StandardNormalizer, out_norm: StandardNormalizer) -> None:
    if inp_norm.mean is None or inp_norm.std is None or out_norm.mean is None or out_norm.std is None:
        raise RuntimeError("Normalizers must be fitted before saving")
    np.savez(
        path,
        inp_mean=inp_norm.mean,
        inp_std=inp_norm.std,
        out_mean=out_norm.mean,
        out_std=out_norm.std,
    )


def load_normalizers(path: Path) -> Tuple[StandardNormalizer, StandardNormalizer]:
    d = np.load(path)
    inp_norm = StandardNormalizer()
    out_norm = StandardNormalizer()
    inp_norm.mean = d["inp_mean"].astype(np.float32)
    inp_norm.std = d["inp_std"].astype(np.float32)
    out_norm.mean = d["out_mean"].astype(np.float32)
    out_norm.std = d["out_std"].astype(np.float32)
    return inp_norm, out_norm


class SimulLearn_Transformer(nn.Module):
    def __init__(self):
        super(SimulLearn_Transformer, self).__init__()
        # iTransformer-style embedding:
        # [B, T, N] -> [B, N, T] -> Linear(T -> D_MODEL) -> [B, N, D_MODEL]
        self.variate_embedding = nn.Linear(SEQUENCE_LENGTH, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        encode_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encode_layer,
            num_layers=NUM_LAYERS,
            norm=nn.LayerNorm(D_MODEL),
        )
        # Shared projection on each variate token, same style as official iTransformer.
        self.projector = nn.Linear(D_MODEL, PRED_LEN, bias=True)

    def forward(self, x):
        _, T, _ = x.shape
        if T != SEQUENCE_LENGTH:
            raise ValueError(f"Expected sequence length {SEQUENCE_LENGTH}, got {T}")

        # Invert dimensions so attention is over variates (tokens), not time steps.
        x = x.permute(0, 2, 1)  # [B, N, T]
        x = self.dropout(self.variate_embedding(x))  # [B, N, D_MODEL]
        x = self.transformer(x)  # [B, N, D_MODEL]
        x = self.projector(x).permute(0, 2, 1)  # [B, PRED_LEN, N]
        return x[:, -1, :OUTPUT_SIZE]  # [B, 13]
    

class CustomDroneDatasetNormalized(Dataset):
    def __init__(
        self,
        data_file: Path,
        inp_norm: StandardNormalizer,
        out_norm: StandardNormalizer,
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        df = pd.read_csv(data_file, header=None)
        self.sequence_length = sequence_length

        inp = df.iloc[:-1, :].values.astype(np.float32)  # (N-1, 17)
        outp = df.iloc[1:, :-4].values.astype(np.float32)  # (N-1, 13)

        if inp.shape[1] != INPUT_SIZE:
            raise ValueError(f"Expected {INPUT_SIZE} input features, got {inp.shape[1]}")
        if outp.shape[1] != OUTPUT_SIZE:
            raise ValueError(f"Expected {OUTPUT_SIZE} output features, got {outp.shape[1]}")

        # Normalize once up front to avoid per-sample CPU bottlenecks in __getitem__.
        self.inp = torch.from_numpy(inp_norm.transform(inp))
        self.outp = torch.from_numpy(out_norm.transform(outp))

    def __len__(self):
        return len(self.inp) - self.sequence_length

    def __getitem__(self, idx):
        x = self.inp[idx : idx + self.sequence_length]  # (seq, 17)
        y = self.outp[idx + self.sequence_length - 1]  # (13,)
        return x, y


def fit_normalizers_from_training_split(data_file: Path) -> Tuple[StandardNormalizer, StandardNormalizer, int]:
    df = pd.read_csv(data_file, header=None)
    inp_all = df.iloc[:-1, :].values.astype(np.float32)
    out_all = df.iloc[1:, :-4].values.astype(np.float32)

    dataset_len = len(inp_all) - SEQUENCE_LENGTH
    train_size = int(0.8 * dataset_len)

    inp_train_rows = inp_all[: SEQUENCE_LENGTH + train_size - 1]
    out_train_rows = out_all[SEQUENCE_LENGTH - 1 : SEQUENCE_LENGTH - 1 + train_size]

    inp_norm = StandardNormalizer()
    out_norm = StandardNormalizer()
    inp_norm.fit(inp_train_rows)
    out_norm.fit(out_train_rows)

    return inp_norm, out_norm, train_size


def normalize_input_sequence(raw_sequence: np.ndarray, inp_norm: StandardNormalizer) -> np.ndarray:
    raw_sequence = np.asarray(raw_sequence, dtype=np.float32)
    if raw_sequence.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
        raise ValueError(f"Expected shape {(SEQUENCE_LENGTH, INPUT_SIZE)}, got {raw_sequence.shape}")
    return inp_norm.transform(raw_sequence)


def denormalize_output(pred_norm: np.ndarray, out_norm: StandardNormalizer) -> np.ndarray:
    pred_norm = np.asarray(pred_norm, dtype=np.float32)
    if pred_norm.shape != (OUTPUT_SIZE,):
        raise ValueError(f"Expected shape {(OUTPUT_SIZE,)}, got {pred_norm.shape}")
    return out_norm.inverse_transform(pred_norm)


def predict_one_step(
    model: nn.Module,
    raw_sequence: np.ndarray,
    inp_norm: StandardNormalizer,
    out_norm: StandardNormalizer,
) -> np.ndarray:
    x_norm = normalize_input_sequence(raw_sequence, inp_norm)
    x_t = torch.from_numpy(x_norm).unsqueeze(0).to(device)  # (1, seq, 17)
    with torch.no_grad():
        y_norm_t = model(x_t)[0]
    y_norm = y_norm_t.detach().cpu().numpy().astype(np.float32)
    return denormalize_output(y_norm, out_norm)


def main_train() -> None:
    data_file = BASE_DIR / "position_diffed_no_timestamp.csv" # position(3) + attitude(4) + linear_vel(3) + angular_vel(3) + motor_value(4) = 17

    inp_norm, out_norm, _ = fit_normalizers_from_training_split(data_file)
    normalizers_path = BASE_DIR / f"normalizers_{date}.npz"
    save_normalizers(normalizers_path, inp_norm, out_norm)
    print(f"Saved normalizers to {normalizers_path}")

    full_dataset = CustomDroneDatasetNormalized(data_file, inp_norm, out_norm, sequence_length=SEQUENCE_LENGTH)

    train_size = int(0.8 * len(full_dataset))
    indices = list(range(len(full_dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    use_cuda = device == "cuda"
    dataloader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "persistent_workers": NUM_WORKERS > 0,
        "pin_memory": use_cuda,
    }
    if NUM_WORKERS > 0:
        dataloader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)

    model = SimulLearn_Transformer().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    use_amp = USE_AMP and use_cuda
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    warmup_epochs = min(WARMUP_EPOCHS, max(NUM_EPOCHS - 1, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(NUM_EPOCHS - warmup_epochs, 1),
                eta_min=MIN_LEARNING_RATE,
            ),
        ],
        milestones=[warmup_epochs],
    )
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    log_path = BASE_DIR / f"SimulLearn_Transformer_Training_Log{date}.txt"
    with open(log_path, "w") as f:
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_dataloader:
                inputs = inputs.to(device, non_blocking=use_cuda)
                targets = targets.to(device, non_blocking=use_cuda)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_dataloader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs = inputs.to(device, non_blocking=use_cuda)
                    targets = targets.to(device, non_blocking=use_cuda)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            line = (
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | Train Loss: {avg_loss:.8f} | Val Loss: {avg_val_loss:.8f} | Learning Rate: {current_lr:.6f}"
            )
            print(line)
            f.write(line + "\n")

            if avg_val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                early_stop_line = f"Early stopping triggered at epoch {epoch + 1}"
                print(early_stop_line)
                f.write(early_stop_line + "\n")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model_path = BASE_DIR / f"SimulLearn_Transformer_Normalized_{date}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main_train()
