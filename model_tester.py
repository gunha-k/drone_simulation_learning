import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

INPUT_SIZE = 17
OUTPUT_SIZE = 13
HIDDEN_SIZE = 256
NUM_LAYERS = 1
SEQUENCE_LENGTH = 20

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


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
    # If omitted, uses the latest `SimulLearn_LSTM_Normalized_*.pth` and matching `normalizers_*.npz`.
    model_path = Path(sys.argv[1]).expanduser() if len(sys.argv) >= 2 else _latest_matching(base_dir, "SimulLearn_LSTM_Normalized_*.pth")

    if len(sys.argv) >= 3:
        normalizers_path = Path(sys.argv[2]).expanduser()
    else:
        timestamp = model_path.name.replace("SimulLearn_LSTM_Normalized_", "").replace(".pth", "")
        normalizers_path = base_dir / f"normalizers_{timestamp}.npz"
        if not normalizers_path.exists():
            normalizers_path = _latest_matching(base_dir, "normalizers_*.npz")

    out_subdir = sys.argv[3] if len(sys.argv) >= 4 else "03-23"

    print(f"Model: {model_path}")
    print(f"Normalizers: {normalizers_path}")

    inp_mean, inp_std, out_mean, out_std = _load_normalizers(normalizers_path)

    model = torch.load(model_path, weights_only=False).to(device)
    model.eval()

    df = pd.read_csv(base_dir / "small_log_diffed.csv", header=None)
    inputs = df.iloc[:-1, 1:].values.astype(np.float32)  # (N, 17)

    sequence = inputs[:SEQUENCE_LENGTH].tolist()  # raw scale
    predictions = []

    num_steps = len(inputs) - SEQUENCE_LENGTH

    for i in range(num_steps):
        raw_seq = np.asarray(sequence[-SEQUENCE_LENGTH:], dtype=np.float32)  # (20, 17)
        x_norm = (raw_seq - inp_mean) / inp_std
        x_t = torch.from_numpy(x_norm).unsqueeze(0).to(device)  # (1, 20, 17)

        with torch.no_grad():
            y_norm_t = model(x_t)[0]  # (13,) normalized

        y_norm = y_norm_t.detach().cpu().numpy().astype(np.float32)
        pred = y_norm * out_std + out_mean  # back to raw(diffed) scale

        predictions.append(pred)
        if (i + 1) % 1000 == 0:
            print(f"Step {i+1}")

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
