import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import datetime

INPUT_SIZE = 23
OUTPUT_SIZE = 19
NUM_EPOCHS = 100
BATCH_SIZE = 2048

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class SimulLearn_DNN(nn.Module):
    def __init__(self):
        super(SimulLearn_DNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, OUTPUT_SIZE),
        )

    def forward(self, x):
        return self.model(x)
    
class CustomDroneDataset(Dataset):
    def __init__(self, data_file):
        df = pd.read_csv(data_file, header=None)
        self.inp = df.iloc[:-1, 1:].values
        self.outp = df.iloc[1:, 1:20].values
        
    def __len__(self):
        return len(self.inp)
    
    def __getitem__(self, idx):
        input_data = torch.FloatTensor(self.inp[idx])
        output_data = torch.FloatTensor(self.outp[idx])
        return input_data, output_data
    
model = SimulLearn_DNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

full_dataset = CustomDroneDataset("final_merged_result.csv")
train_size = int(0.8*len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_dataload = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataload = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = model.to(device)
model.train()
for epoch in range(NUM_EPOCHS):
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
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
print("Training complete.")
model.eval()
val_loss = 0.0
with torch.no_grad():
    for inputs, targets in val_dataload:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()
avg_val_loss = val_loss / len(val_dataload)
print(f"Validation : {avg_val_loss}")
dummy_input = torch.randn(1, INPUT_SIZE).to(device)
date = datetime.datetime.now().time().strftime("%H-%M-%S")

torch.onnx.export(
    model,
    dummy_input,
    f"SimulLearn_DNN_{date}.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print(f"Model saved to SimulLearn_DNN_{date}.onnx")
