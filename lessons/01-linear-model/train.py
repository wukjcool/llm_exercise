"""Lesson 1：线性模型、自动微分、梯度下降和优化器。"""

import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(x)
print(x.shape)

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

y = 3 * x + 2
# x为 1xn,  input 变成 nx1
inputs = x.reshape(-1, 1)
targets = y.reshape(-1, 1)

model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(100):
    outputs: torch.Tensor = model(inputs)
    model_loss = ((outputs-targets) ** 2).mean()
    if step % 10 == 0:
        print(f"已更新{step}次, loss={model_loss.item():.8f}")
    model_loss.backward()
    optimizer.step()
    optimizer.zero_grad()


print("最终权重:", model.weight.item())
print("最终偏置:", model.bias.item())
