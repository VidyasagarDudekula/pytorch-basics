from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch

X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
Y = [0,1,0, 1]
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.int32)


dataset = TensorDataset(X_tensor, Y_tensor)

for i in range(len(dataset)):
    X_sample, Y_sample = dataset[i]
    print(f"X[{i}]: {X_sample}, Y[{i}]: {Y_sample}")

dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

for batch_x, batch_y in dataloader:
    print("Batch X:- ", batch_x)
    print("Batch Y:- ", batch_y)

