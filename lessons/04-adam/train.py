"""Lesson 4：Adam 优化器与收敛速度比较。"""


import torch

x = torch.linspace(-2, 2, 101)

inputs = x.reshape(-1, 1)

targets = (x ** 2).reshape(-1, 1)

dataset = torch.utils.data.TensorDataset(inputs, targets)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

torch.manual_seed(42)

model = torch.nn.Sequential(
    torch.nn.Linear(1, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 1),
)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

with torch.no_grad():
    initial_loss = ((model(inputs) - targets) ** 2).mean()
print("训练前 loss:", initial_loss.item())

for epoch in range(100):
    for step, (batch_inputs, batch_targets) in enumerate(loader, start=1):
        outputs: torch.Tensor = model(batch_inputs)
        loss: torch.Tensor = ((outputs - batch_targets) ** 2).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(
        #     f"epoch={epoch + 1}, step={step}, "
        #     f"本批更新前 loss={loss.item():.6f}"
        # )

    with torch.no_grad():
        epoch_loss = ((model(inputs) - targets) ** 2).mean()

    print(
        f"完成 epoch {epoch + 1}，"
        f"整个训练集 loss={epoch_loss.item():.6f}"
    )