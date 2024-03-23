import torch
import timeit

device = torch.device("cuda")
# Defining the functions
def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return x + (y - x).detach()

def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    quant = (w * scale).round().clamp(-1, 1) / scale
    w_quant = w + (quant - w).detach()
    scale = abs(w_quant).max().detach()
    w_quant = w_quant / scale
    return w_quant, scale

# Scripting the functions
activation_quant_scripted = torch.jit.script(activation_quant)
weight_quant_scripted = torch.jit.script(weight_quant)

# Setup for benchmark
x = torch.randn(1024, 1024).to(device)
w = torch.randn(1024, 1024).to(device)

iterations = 10000


# Time activation quant function
activation_quant_time = timeit.timeit(lambda: activation_quant(x), number=iterations)
activation_quant_scripted_time = timeit.timeit(lambda: activation_quant_scripted(x), number=iterations)

# Time weight quant function
weight_quant_time = timeit.timeit(lambda: weight_quant(w), number=iterations)
weight_quant_scripted_time = timeit.timeit(lambda: weight_quant_scripted(w), number=iterations)


print(f"{activation_quant_time:.4f} > {activation_quant_scripted_time:.4f}")
print()
print(f"{weight_quant_time:.4f} > {weight_quant_scripted_time:.4f}")
