"""Lesson 9 因果掩码（causal mask）

沿用「去北」的例子，让每个位置只使用自己和前面的字。
第一步在 Softmax 前把「去 → 北」的分数设为负无穷，
再检查「去」这一行的注意力权重是否为 [1, 0]。
"""
import torch

# scores = torch.tensor([[14.0, 30.0],
#                        [38.0, 86.0]]) / (2 ** 0.5)

scores = torch.zeros((3, 3))

all_true = torch.ones_like(scores, dtype=torch.bool)
mask = torch.triu(all_true, diagonal=1)
print(mask)
masked_scores = scores.masked_fill(mask, float("-inf"))
weights = torch.softmax(masked_scores, dim=-1)

print("「去」的注意力权重:", weights)