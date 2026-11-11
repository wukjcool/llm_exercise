"""Lesson 9 因果掩码前后对照。"""

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

all_true = torch.ones_like(scaled_scores, dtype=torch.bool)
mask = torch.triu(all_true, diagonal=1)
masked_scores = scaled_scores.masked_fill(mask, float("-inf"))

masked_weights = torch.softmax(masked_scores, dim=-1)
masked_output = masked_weights @ V
print("加掩码前「去」的输出:", attention_output[0])
print("加掩码后「去」的输出:", masked_output[0])
