import torch
import numpy as np
# Create differentiable tensor
x = torch.tensor(torch.arange(0,4), requires_grad=False)
print(x.dtype)
# Calculate y=sum(x**2)
y = x**2
# Calculate gradient (dy/dx=2x)
y.sum().backward()
# Print values
print(x)
print(y)
print(x.grad)