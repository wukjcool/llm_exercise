"""Lesson 5：分类与交叉熵。练习字符打分、类别编号、预测类别和损失计算。"""

import torch

vocab = ["北", "京", "上"]

scores = torch.tensor([[0.2, 1.4, -0.3]])
target = torch.tensor([1]) # 「京」的编号

# loss = torch.nn.CrossEntropyLoss()(scores, target)

log_probs = torch.log_softmax(scores, dim=1)
loss = -log_probs[0, 1]

predicted_index = int(scores.argmax(dim=1).item())
print("预测:", vocab[predicted_index])

print("损失:", loss.item())
