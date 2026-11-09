"""Lesson 8：单查询练习。给两个位置打分，再对字向量加权求和。"""

import torch

qu = torch.tensor([1.0, 2.0, 3.0, 4.0])   # 去
bei = torch.tensor([5.0, 6.0, 7.0, 8.0])  # 北

# scores = torch.tensor([2.0, 1.0]) # 「去」的分数，「北」大分数
query = torch.tensor([1.0, 0.0, 0.0, 0.0])

score_qu = torch.dot(query, qu)
score_bei = torch.dot(query, bei)
scores = torch.stack([score_qu, score_bei])

print("点积分数:", scores)

weights = torch.softmax(scores, dim=0)

context = weights[0] * qu + weights[1] * bei
print("Softmax权重:", weights)
print("用Softmax权重算出的向量:", context)
