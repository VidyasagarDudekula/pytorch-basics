import torch

tensor_example = torch.tensor([[1, 2, 3], [4, 5, 6]])

print(tensor_example.view(3, 2))
print(tensor_example.view(1, 6), tensor_example.view(6, 1))

print(tensor_example.view(3, -1))
print(tensor_example.view(1, -1))
print(tensor_example.view(-1))

tensor_a = tensor_example.view(1, -1)
tensor_b = tensor_example.view(-1)

print(tensor_a.shape)
print(tensor_b.shape)