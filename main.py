import torch;

x=torch.arange(12,dtype=torch.int32).reshape(3,4)

print(torch.exp(x))