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
NUM_EPOCHS = 20
BATCH_SIZE = 64
HIDDEN_SIZE = 256
NUM_LAYERS = 1
NUM_WORKERS = 12
PREFETCH_FACTOR = 2
SEQUENCE_LENGTH = 20
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


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


class SimulLearn_LSTM(nn.Module):
    def __init__(self):
        super(SimulLearn_LSTM, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.num_layers = NUM_LAYERS
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


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

        inp = df.iloc[:-1, 1:].values.astype(np.float32)  # (N-1, 17)
        outp = df.iloc[1:, 1:-4].values.astype(np.float32)  # (N-1, 13)

        if inp.shape[1] != INPUT_SIZE:
            raise ValueError(f"Expected {INPUT_SIZE} input features, got {inp.shape[1]}")
        if outp.shape[1] != OUTPUT_SIZE:
            raise ValueError(f"Expected {OUTPUT_SIZE} output features, got {outp.shape[1]}")

        self.inp = inp
        self.outp = outp
        self.inp_norm = inp_norm
        self.out_norm = out_norm

    def __len__(self):
        return len(self.inp) - self.sequence_length

    def __getitem__(self, idx):
        x = self.inp[idx : idx + self.sequence_length]  # (seq, 17)
        y = self.outp[idx + self.sequence_length]  # (13,)

        x_norm = self.inp_norm.transform(x)
        y_norm = self.out_norm.transform(y)

        return torch.from_numpy(x_norm), torch.from_numpy(y_norm)


output_weights = torch.tensor(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ],
    dtype=torch.float32,
).to(device)


class WeightedMSELoss(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weights = weights

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        return (mse * self.weights).mean()


def fit_normalizers_from_training_split(data_file: Path) -> Tuple[StandardNormalizer, StandardNormalizer, int]:
    df = pd.read_csv(data_file, header=None)
    inp_all = df.iloc[:-1, 1:].values.astype(np.float32)
    out_all = df.iloc[1:, 1:-4].values.astype(np.float32)

    dataset_len = len(inp_all) - SEQUENCE_LENGTH
    train_size = int(0.8 * dataset_len)

    inp_train_rows = inp_all[: SEQUENCE_LENGTH + train_size]
    out_train_rows = out_all[SEQUENCE_LENGTH : SEQUENCE_LENGTH + train_size]

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
    data_file = BASE_DIR / "position_diffed.csv"

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

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=True,
    )

    model = SimulLearn_LSTM().to(device)
    criterion = WeightedMSELoss(output_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    log_path = BASE_DIR / f"SimulLearn_LSTM_Training_Log{date}.txt"
    with open(log_path, "w") as f:
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_dataloader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            line = (
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | Train Loss: {avg_loss:.8f} | Val Loss: {avg_val_loss:.8f}"
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

    model_path = BASE_DIR / f"SimulLearn_LSTM_Normalized_{date}.pth"
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main_train()
