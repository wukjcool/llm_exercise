import torch

from math import log, exp


a = [0.2, 1.4, -0.3]


p2 = exp(1.4)/(exp(0.2) + exp(1.4) + exp(-0.3))


print(-log(p2))