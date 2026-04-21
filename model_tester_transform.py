import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

INPUT_SIZE = 17
OUTPUT_SIZE = 13
NUM_LAYERS = 2
SEQUENCE_LENGTH = 128
D_MODEL = 128
NHEAD = 8
DROPOUT = 0.1

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class SimulLearn_Transformer(nn.Module):
    def __init__(self):
        super(SimulLearn_Transformer, self).__init__()
        self.input_projection = nn.Linear(INPUT_SIZE, D_MODEL)
        self.pos_embedding = nn.Embedding(SEQUENCE_LENGTH, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

        encode_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encode_layer, num_layers=NUM_LAYERS)
        self.fc = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, OUTPUT_SIZE),
        )

    def forward(self, x):
        _, T, _ = x.shape

        if T != SEQUENCE_LENGTH:
            raise ValueError(
                f"Expected sequence length {SEQUENCE_LENGTH}, got {T}"
            )

        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x = self.dropout(self.input_projection(x) + self.pos_embedding(pos))
        x = self.transformer(x)
        x = x.mean(dim=1)  # (B, D_MODEL)
        return self.fc(x)


def _latest_matching(base_dir: Path, pattern: str) -> Path:
    matches = list(base_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files match {pattern} under {base_dir}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def _load_normalizers(path: Path):
    d = np.load(path)
    inp_mean = d["inp_mean"].astype(np.float32)
    inp_std = d["inp_std"].astype(np.float32)
    out_mean = d["out_mean"].astype(np.float32)
    out_std = d["out_std"].astype(np.float32)
    return inp_mean, inp_std, out_mean, out_std


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    # Usage:
    #   python3 model_tester.py [MODEL_PATH] [NORMALIZERS_PATH] [OUT_SUBDIR]
    # If omitted, uses latest Transformer model and matching normalizers.
    model_path = "SimulLearn_Transformer_Normalized_2026-04-01_15-29-09.pth"
    normalizers_path = "normalizers_2026-04-01_15-29-09.npz"

    out_subdir = "04-01"

    print(f"Model: {model_path}")
    print(f"Normalizers: {normalizers_path}")

    inp_mean, inp_std, out_mean, out_std = _load_normalizers(normalizers_path)

    # 모델 전체를 torch.save(model, path)로 저장했으므로
    # 같은 클래스 정의가 이 파일 안에 있어야 torch.load가 됩니다.
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()

    df = pd.read_csv(base_dir / "small_log_diffed_mission.csv", header=None)
    inputs = df.iloc[:-1, 1:].values.astype(np.float32)  # (N, 17)

    if inputs.shape[1] != INPUT_SIZE:
        raise ValueError(
            f"Expected {INPUT_SIZE} input features, got {inputs.shape[1]}"
        )

    if len(inputs) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least {SEQUENCE_LENGTH} rows, got {len(inputs)}"
        )

    sequence = inputs[:SEQUENCE_LENGTH].tolist()  # raw scale
    predictions = []

    num_steps = len(inputs) - SEQUENCE_LENGTH

    for i in range(num_steps):
        raw_seq = np.asarray(sequence[-SEQUENCE_LENGTH:], dtype=np.float32)  # (128, 17)
        x_norm = (raw_seq - inp_mean) / inp_std
        x_t = torch.from_numpy(x_norm).unsqueeze(0).to(device)  # (1, 128, 17)

        with torch.no_grad():
            y_norm_t = model(x_t)[0]  # (13,)

        y_norm = y_norm_t.detach().cpu().numpy().astype(np.float32)
        pred = y_norm * out_std + out_mean  # back to raw(diffed) scale

        predictions.append(pred)

        if (i + 1) % 1000 == 0:
            print(f"Step {i + 1}")

        next_input = np.zeros(INPUT_SIZE, dtype=np.float32)
        next_input[:OUTPUT_SIZE] = pred
        next_input[OUTPUT_SIZE:] = inputs[SEQUENCE_LENGTH + i, OUTPUT_SIZE:]
        sequence.append(next_input.tolist())

    out_path = base_dir / out_subdir / "predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(out_path, index=False, header=False)
    print(f"Saved: {out_path} ({len(predictions)} rows)")


if __name__ == "__main__":
    main()