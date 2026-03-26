import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import datetime

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

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class OutputNormalizer:
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0)        # (13,)
        self.std  = data.std(axis=0) + 1e-8  # (13,)

    def transform(self, data: np.ndarray):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray):
        return data * self.std + self.mean

    def save(self, path="normalizer.npz"):
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path="normalizer.npz"):
        d = np.load(path)
        self.mean, self.std = d["mean"], d["std"]

_df_all   = pd.read_csv("position_diffed.csv", header=None)
_outp_all = _df_all.iloc[1:, 1:-4].values.astype(np.float32)  # (N-1, 13)
normalizer = OutputNormalizer()
normalizer.fit(_outp_all)
normalizer.save(f"normalizer_{date}.npz")  # 추론 시 역변환용
print("Output mean:", normalizer.mean)
print("Output std: ", normalizer.std)

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
    
class CustomDroneDataset(Dataset):
    def __init__(self, data_file, sequence_length=SEQUENCE_LENGTH):
        df = pd.read_csv(data_file, header=None)
        self.sequence_length = sequence_length
        self.inp = df.iloc[:-1, 1:].values
        self.outp = df.iloc[1:, 1:-4].values
        
    def __len__(self):
        return len(self.inp) - self.sequence_length
    
    def __getitem__(self, idx):
        input_data = torch.FloatTensor(self.inp[idx:idx+self.sequence_length])
        output_data = torch.FloatTensor(self.outp[idx + self.sequence_length])
        return input_data, output_data

output_weights = torch.tensor([
    1.0,  # position
    1.0,  # 
    1.0,  # 
    1.0,  # attitude
    1.0,  # 
    1.0,  # 
    1.0,  # 
    1.0,  # linear velocity
    1.0,  #
    1.0,  #
    1.0,  # angular velocity
    1.0,  #
    1.0,  #
], dtype=torch.float32).to(device)

class WeightedMSELoss(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.weights = weights

    def forward(self, pred, target):
        mse = (pred - target) ** 2       # (batch, 13)
        return (mse * self.weights).mean()
    
model = SimulLearn_LSTM()
criterion = WeightedMSELoss(output_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

full_dataset = CustomDroneDataset("position_diffed.csv", sequence_length=SEQUENCE_LENGTH)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

indices = list(range(len(full_dataset)))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

train_dataload = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)
val_dataload = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)

model = model.to(device)

f = open(f"SimulLearn_LSTM_Training_Log{date}.txt", "w")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_dataload:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_dataload)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_dataload:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_dataload)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_loss:.8f} | Val Loss: {avg_val_loss:.8f}")
    f.write(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_loss:.8f} | Val Loss: {avg_val_loss:.8f}\n")
print("Training complete.")
f.close()

torch.save(model, f"SimulLearn_LSTM_{date}.pth")
print(f"Model saved to SimulLearn_LSTM_{date}.pth")
