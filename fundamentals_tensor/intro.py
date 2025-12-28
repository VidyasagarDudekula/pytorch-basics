import torch

tensor_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
tensor_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)

tensor_sum = torch.add(tensor_a, tensor_b)

print(tensor_sum)

tensor_sum = tensor_a + tensor_b
print(tensor_sum)

tensor_product = torch.mul(tensor_a, tensor_b)
print(tensor_product)

tensor_product = tensor_a * tensor_b
print(tensor_product)

tensor_dot_product = torch.matmul(tensor_a, tensor_b)
print(tensor_dot_product)
print("Broadcasting\n")
tensor_c = torch.tensor([[1], [2]])

print(tensor_a + tensor_c)