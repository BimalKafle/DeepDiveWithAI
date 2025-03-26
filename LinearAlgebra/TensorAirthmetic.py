import torch

A=torch.arange(6,dtype=torch.float32).reshape(2,3)
B=A.clone()
print(A,B,A+B)


"""Element wise product of two matrices is called their Hadamard product"""

print(A*B)

a=2
X=torch.arange(24).reshape(2,3,4)

print(a+X,(a*X).shape)