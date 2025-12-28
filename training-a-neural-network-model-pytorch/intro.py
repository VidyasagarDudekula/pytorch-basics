import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([
    [3.0, 0.5], [1.0, 1.0], [0.5, 2.0], [2.0, 1.5],
    [3.5, 3.0], [2.0, 2.5], [1.5, 1.0], [0.5, 0.5],
    [2.5, 0.8], [2.1, 2.0], [1.2, 0.5], [0.7, 1.5]
], dtype=torch.float32)

Y = torch.tensor([[1], [0], [0], [1], [1], [0], [1], [0], [1], [0], [1], [0]], dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

critetion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)


for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = critetion(outputs, Y)
    loss.backward()
    optimizer.step()
    if epoch%10 == 0:
        print(f"Epoch:- {epoch}, loss:- {loss.item()}")

