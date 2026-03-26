import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

SEQUENCE_LENGTH = 20

date = "03-23"

BASE_DIR = Path(__file__).resolve().parent

original = pd.read_csv(BASE_DIR / "small_log.csv", header=None)
predicted = pd.read_csv(BASE_DIR / date / "predictions.csv", header=None)

predicted.iloc[0, 0:3] += original.iloc[SEQUENCE_LENGTH - 1, 0:3]
for i in range(1, len(predicted)):
    predicted.iloc[i, 0:3] += predicted.iloc[i - 1, 0:3]
predicted_points = predicted.iloc[:, 0:3].values
original_points = original.iloc[SEQUENCE_LENGTH:, 0:3].values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(original_points[:, 0], original_points[:, 1], original_points[:, 2], label='Original')
ax.plot(predicted_points[:, 0], predicted_points[:, 1], predicted_points[:, 2], label='Predicted')
ax.legend()

output_dir = BASE_DIR / date
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "path_predicted_2026_03_23.png"
plt.savefig(output_path)
print(f"Saved plot to: {output_path}")
plt.show()
