import torch
import math

def gelu(x):
    return x*(1+torch.erf(x/math.sqrt(2)))/2

if __name__ == '__main__':
    print(gelu(torch.randn(100, 100)))