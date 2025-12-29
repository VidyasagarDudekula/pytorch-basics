from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim


X_train = torch.tensor([
    [3.0, 0.5], [1.0, 1.0], [0.5, 2.0], [2.0, 1.5],
    [3.5, 3.0], [2.0, 2.5], [1.5, 1.0], [0.5, 0.5],
    [2.5, 0.8], [2.1, 2.0], [1.2, 0.5], [0.7, 1.5]
], dtype=torch.float32)
y_train = torch.tensor([[1], [0], [0], [1], [1], [0], [1], [0], [1], [0], [1], [0]], dtype=torch.float32)
X_test = torch.tensor([[2.5, 1.0], [0.8, 0.8], [1.0, 2.0], [3.0, 2.5]], dtype=torch.float32)
y_test = torch.tensor([[1], [0], [0], [1]], dtype=torch.float32)


model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch%10 == 0:
        print(loss)
        print(f"Epoch :- {epoch}, loss:- {loss.item()}")

model.eval()

with torch.no_grad():
    predictions = model(X_test)
    predictions_class = (predictions>0.5).int()
    loss = criterion(predictions, y_test)
    accuracy = accuracy_score(y_test.numpy(), predictions_class.numpy())

print(f"Raw predictions:- {predictions}")
print(f"Loss:- {loss.item()}, accuracy:- {accuracy}")
    