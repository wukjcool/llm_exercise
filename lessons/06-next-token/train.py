"""Lesson 6：下一字符预测。练习字符编码、构造输入与目标，以及用交叉熵训练模型。"""

import torch
from pathlib import Path

text = "北京上海北京"
vocab = ["北", "京", "上", "海"]
char_to_id = {char: i for i, char in enumerate(vocab)}

ids = torch.tensor([char_to_id[char] for char in text])

inputs = ids[:-1]
targets = ids[1:]
print("输入:", inputs)
print("目标:", targets)

torch.manual_seed(42)
vocab_size = len(vocab)

model = torch.nn.Embedding(vocab_size, vocab_size)
scores = model(inputs)

loss_fn = torch.nn.CrossEntropyLoss()

loss: torch.Tensor = loss_fn(scores, targets)
print("分数形状:", scores.shape)
print("初始损失:", loss.item())


optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for step in range(100):
    scores = model(inputs)
    loss = loss_fn(scores, targets)

    if step % 20 == 0:
        print(f"已更新{step}次, loss={loss.item():.6f}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


with torch.no_grad():
    final_loss = loss_fn(model(inputs), targets)
print("训练后损失:", final_loss.item())

save_path = Path(__file__).with_name("next_token.pt")
torch.save({"weights": model.state_dict(), "vocab":vocab}, save_path)
print("已保存", save_path)