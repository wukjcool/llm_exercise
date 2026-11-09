"""Lesson 8：Attention 入门。练习对两个位置的向量做加权求和。"""

import torch

qu = torch.tensor([1.0, 2.0, 3.0, 4.0])   # 去
bei = torch.tensor([5.0, 6.0, 7.0, 8.0])  # 北

X = torch.stack([qu, bei], dim=0)
print(X, X.shape)

W_q = torch.tensor([
    [1.0, 0.0],
    [0.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
])

Q = X @ W_q
print("Q:", Q)
print("Q 的形状:", Q.shape)

W_k = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 0.0],
    [0.0, 1.0],
])

K = X @ W_k
print("K:", K)
print("K 的形状:", K.shape)

W_v = torch.tensor([
    [1.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
])

V = X @ W_v
print("V:", V)
print("V 的形状:", V.shape)

attention_scores = Q @ K.T
print("点积分数:", attention_scores)

d_k = K.shape[-1]
scaled_scores = attention_scores / (d_k ** 0.5)
print("缩放后的分数:", scaled_scores)
attention_weights = torch.softmax(scaled_scores, dim=-1)
print("注意力权重:", attention_weights)
print("每行权重之和:", attention_weights.sum(dim=-1))


attention_output = attention_weights @ V
print("Attention 输出:", attention_output)
print("输出形状:", attention_output.shape)
