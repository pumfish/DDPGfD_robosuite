import time
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512))

    def forward(self, x):
        s = time.time()
        out = self.net(x)
        e = time.time()
        print(f"cost time = {e-s}s")
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
net = Net(19, 10)
net.to(device)

s = time.time()
for _ in range(10):
    x = torch.randn(1, 19)
    x = x.to(device)
    out = net(x)
e = time.time()
print(f"total cost = {e-s}s")
