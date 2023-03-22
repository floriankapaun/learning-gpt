import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

B, T, C = 4, 8, 2  # batch, time, channels
x = torch.randn(B, T, C)
# print(x.shape)

# 1st version
# we want x[b, t] = . mean_{i<=t} x[b, i]
xbow = torch.zeros((B, T, C))  # bow = bag of words
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]  # (t, C)
        xbow[b, t] = torch.mean(xprev, 0)

# print(torch.tril(torch.ones(3, 3)))

a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b

# print('a=', a)
# print('b=', b)
# print('c=', c)

# 2nd version
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x  # (B, T, T) @ (B, T, C) -> (B, T, C)

# print(wei)
# print(xbow2)
# print(torch.allclose(xbow, xbow2))

# 3rd version
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
# print(wei)
# print(xbow3)
# print(torch.allclose(xbow, xbow3))

# 4th version (self attention)
torch.manual_seed(1337)
B, T, C = 4, 8, 32  # batch, time, channels
x = torch.randn(B, T, C)

# let's see a single head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)  # (B, T, head_size)
q = query(x)  # (B, T, head_size)
# (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
wei = q @ k.transpose(-2, -1)  # * head_size**-0.5

tril = torch.tril(torch.ones(T, T))
print(wei[0])
wei = wei.masked_fill(tril == 0, float('-inf'))
print(wei[0])
wei = F.softmax(wei, dim=-1)
# out = wei @ x
v = value(x)
out = wei @ v

print(wei[0])
# print(out.shape)
