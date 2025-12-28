import torch
import torch.nn as nn

input_tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

layer = nn.Linear(in_features=2, out_features=3)

output_tensor = layer(input_tensor)

print(f"Input Tensor:- {input_tensor}")

print(f"Output tesnor before activation:- {output_tensor}")

relu = nn.ReLU()

activated_output_tesnro = relu(output_tensor)

print(f"Output after activation function:- {activated_output_tesnro}")

sigmoid = nn.Sigmoid()

activated_output_sigmoid = sigmoid(output_tensor)

print(f"Output tesnro after sigmoid activation function:- {activated_output_sigmoid}")
