import torch
import torch.nn as nn

input_tensor = torch.tensor([[2.0, 3.0, 1.0, 5.0]], dtype=torch.float32)

layer = nn.Linear(in_features=4, out_features=3)

output_tensor = layer(input_tensor)

print(f"Output tensor before activation function:- {output_tensor}")
print(f"laer weights:- {layer.weight}")
print(f"layer bias:- {layer.bias}")