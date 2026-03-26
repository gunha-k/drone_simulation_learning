import pandas as pd
import numpy as np

df = pd.read_csv("small_log.csv", header=None)
previous_x = 0
previous_y = 0
previous_z = 0
for i in range(len(df[1])): 
    pre_x = df[1][i]
    pre_y = df[2][i]
    pre_z = df[3][i]
    df[1][i] = df[1][i] - previous_x
    df[2][i] = df[2][i] - previous_y
    df[3][i] = df[3][i] - previous_z
    previous_x = pre_x
    previous_y = pre_y
    previous_z = pre_z
print(df.head())
df.to_csv("small_log_diffed.csv", header=None, index=False)