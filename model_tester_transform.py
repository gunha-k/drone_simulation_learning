import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ===================== User Input (edit here) =====================
MODEL_FILE = "SimulLearn_Transformer_Normalized_2026-04-23_15-58-58.pth"
NORMALIZERS_FILE = "normalizers_2026-04-23_15-58-58.npz"
INPUT_CSV_FILE = "small_log_diffed_mission.csv"
OUTPUT_TXT_FILE = "transform_eval_result.txt"
# ================================================================

INPUT_SIZE = 17
OUTPUT_SIZE = 13
NUM_LAYERS = 4
LEGACY_NUM_LAYERS = 2
SEQUENCE_LENGTH = 128
D_MODEL = 128
NHEAD = 8
DROPOUT = 0.1
PRED_LEN = 1

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class SimulLearn_Transformer(nn.Module):
    # Same architecture as SimulLearn_Transformer_Normalized.py
    def __init__(self):
        super(SimulLearn_Transformer, self).__init__()
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
        self.projector = nn.Linear(D_MODEL, PRED_LEN, bias=True)

    def forward(self, x):
        _, T, _ = x.shape
        if T != SEQUENCE_LENGTH:
            raise ValueError(f"Expected sequence length {SEQUENCE_LENGTH}, got {T}")

        x = x.permute(0, 2, 1)  # [B, N, T]
        x = self.dropout(self.variate_embedding(x))  # [B, N, D]
        x = self.transformer(x)  # [B, N, D]
        x = self.projector(x).permute(0, 2, 1)  # [B, PRED_LEN, N]
        return x[:, -1, :OUTPUT_SIZE]  # [B, 13]


class SimulLearn_TransformerLegacy(nn.Module):
    # Backward compatibility for older checkpoints.
    def __init__(self):
        super(SimulLearn_TransformerLegacy, self).__init__()
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
        self.transformer = nn.TransformerEncoder(
            encode_layer, num_layers=LEGACY_NUM_LAYERS
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Linear(D_MODEL, OUTPUT_SIZE),
        )

    def forward(self, x):
        _, T, _ = x.shape
        if T != SEQUENCE_LENGTH:
            raise ValueError(f"Expected sequence length {SEQUENCE_LENGTH}, got {T}")

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.dropout(self.input_projection(x) + self.pos_embedding(pos))
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


def _strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def _load_model() -> nn.Module:
    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        state_dict = _strip_module_prefix(state_dict)
        if "variate_embedding.weight" in state_dict:
            model = SimulLearn_Transformer()
        elif "input_projection.weight" in state_dict:
            model = SimulLearn_TransformerLegacy()
        else:
            raise RuntimeError("Unsupported checkpoint keys.")
        model.load_state_dict(state_dict)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    model = model.to(device)
    model.eval()
    return model


def _load_normalizers():
    d = np.load(NORMALIZERS_FILE)
    inp_mean = d["inp_mean"].astype(np.float32)
    inp_std = d["inp_std"].astype(np.float32)
    out_mean = d["out_mean"].astype(np.float32)
    out_std = d["out_std"].astype(np.float32)
    return inp_mean, inp_std, out_mean, out_std


def _format_report(
    mse_norm: float,
    mae_norm: float,
    rmse_norm: float,
    mse_norm_by_feature: np.ndarray,
    mae_norm_by_feature: np.ndarray,
    rmse_norm_by_feature: np.ndarray,
    mse_raw: float,
    mae_raw: float,
    rmse_raw: float,
    mse_by_feature: np.ndarray,
    mae_by_feature: np.ndarray,
    rmse_by_feature: np.ndarray,
    num_steps: int,
) -> str:
    lines = []
    lines.append("=== Transformer One-Step Evaluation Report ===")
    lines.append(f"Model file       : {MODEL_FILE}")
    lines.append(f"Normalizers file : {NORMALIZERS_FILE}")
    lines.append(f"Input csv file   : {INPUT_CSV_FILE}")
    lines.append(f"Evaluated steps  : {num_steps}")
    lines.append("")
    lines.append("[Existing metrics: normalized-output test-loss style]")
    lines.append(
        "Same formula as train/valid loss (criterion=nn.MSELoss on normalized outputs)"
    )
    lines.append(f"Overall MSE  : {mse_norm:.8f}")
    lines.append(f"Overall MAE  : {mae_norm:.8f}")
    lines.append(f"Overall RMSE : {rmse_norm:.8f}")
    lines.append("")
    lines.append("[Per-feature error (normalized scale)]")
    for i in range(OUTPUT_SIZE):
        lines.append(
            f"y{i:02d} -> MSE: {mse_norm_by_feature[i]:.8f}, MAE: {mae_norm_by_feature[i]:.8f}, RMSE: {rmse_norm_by_feature[i]:.8f}"
        )
    lines.append("")
    lines.append("[Requested metrics: original csv target vs denormalized prediction]")
    lines.append("Target basis: next-state first 13 columns from original csv")
    lines.append(f"Overall MSE  : {mse_raw:.8f}")
    lines.append(f"Overall MAE  : {mae_raw:.8f}")
    lines.append(f"Overall RMSE : {rmse_raw:.8f}")
    lines.append("")
    lines.append("[Per-feature error (raw scale)]")
    for i in range(OUTPUT_SIZE):
        lines.append(
            f"y{i:02d} -> MSE: {mse_by_feature[i]:.8f}, MAE: {mae_by_feature[i]:.8f}, RMSE: {rmse_by_feature[i]:.8f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    print(f"Model file: {MODEL_FILE}")
    print(f"Normalizers file: {NORMALIZERS_FILE}")
    print(f"Input csv file: {INPUT_CSV_FILE}")
    print("Mode: one-step evaluation (no autoregressive rollout)")

    inp_mean, inp_std, out_mean, out_std = _load_normalizers()
    inp_std = np.where(inp_std == 0, 1.0, inp_std)
    out_std = np.where(out_std == 0, 1.0, out_std)

    model = _load_model()

    df = pd.read_csv(INPUT_CSV_FILE, header=None)
    if df.shape[1] == INPUT_SIZE + 1:
        # Timestamp + 17 features
        inputs = df.iloc[:-1, 1:].values.astype(np.float32)
    elif df.shape[1] == INPUT_SIZE:
        # 17 features only
        inputs = df.iloc[:-1, :].values.astype(np.float32)
    else:
        raise ValueError(
            f"CSV must have {INPUT_SIZE} or {INPUT_SIZE + 1} columns, got {df.shape[1]}"
        )

    if len(inputs) <= SEQUENCE_LENGTH:
        raise ValueError(f"Need at least {SEQUENCE_LENGTH + 1} rows, got {len(inputs)}")

    num_steps = len(inputs) - SEQUENCE_LENGTH
    targets = inputs[SEQUENCE_LENGTH:, :OUTPUT_SIZE]
    predictions = np.zeros((num_steps, OUTPUT_SIZE), dtype=np.float32)
    predictions_norm = np.zeros((num_steps, OUTPUT_SIZE), dtype=np.float32)

    for i in range(num_steps):
        # Use only original data window.
        raw_seq = inputs[i : i + SEQUENCE_LENGTH]
        x_norm = (raw_seq - inp_mean) / inp_std
        x_t = torch.from_numpy(x_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            y_norm_t = model(x_t)[0]

        y_norm = y_norm_t.detach().cpu().numpy().astype(np.float32)
        predictions_norm[i] = y_norm
        predictions[i] = y_norm * out_std + out_mean

        if (i + 1) % 1000 == 0:
            print(f"Step {i + 1}")

    targets_norm = (targets - out_mean) / out_std

    diff_raw = predictions - targets
    mse_by_feature = np.mean(diff_raw**2, axis=0)
    mae_by_feature = np.mean(np.abs(diff_raw), axis=0)
    rmse_by_feature = np.sqrt(np.mean(diff_raw**2, axis=0))
    overall_mse = float(np.mean(diff_raw**2))
    overall_mae = float(np.mean(np.abs(diff_raw)))
    overall_rmse = float(np.sqrt(np.mean(diff_raw**2)))

    diff_norm = predictions_norm - targets_norm
    mse_norm_by_feature = np.mean(diff_norm**2, axis=0)
    mae_norm_by_feature = np.mean(np.abs(diff_norm), axis=0)
    rmse_norm_by_feature = np.sqrt(mse_norm_by_feature)
    mse_norm = float(np.mean(diff_norm**2))
    mae_norm = float(np.mean(np.abs(diff_norm)))
    rmse_norm = float(np.sqrt(mse_norm))

    report_text = _format_report(
        mse_norm=mse_norm,
        mae_norm=mae_norm,
        rmse_norm=rmse_norm,
        mse_norm_by_feature=mse_norm_by_feature,
        mae_norm_by_feature=mae_norm_by_feature,
        rmse_norm_by_feature=rmse_norm_by_feature,
        mse_raw=overall_mse,
        mae_raw=overall_mae,
        rmse_raw=overall_rmse,
        mse_by_feature=mse_by_feature,
        mae_by_feature=mae_by_feature,
        rmse_by_feature=rmse_by_feature,
        num_steps=num_steps,
    )

    with open(OUTPUT_TXT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text, end="")
    print(f"Saved report: {OUTPUT_TXT_FILE}")


if __name__ == "__main__":
    main()
