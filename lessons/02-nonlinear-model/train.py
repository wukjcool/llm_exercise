"""Lesson 2：非线性网络、训练效果可视化和测试集评测。"""

import torch
import matplotlib.pyplot as plt
torch.manual_seed(42)

x = torch.linspace(-2, 2, 101)
y = x ** 2

inputs = x.reshape(-1, 1)
targets = y.reshape(-1, 1)

# model = torch.nn.Linear(1, 1)
model = torch.nn.Sequential(
    torch.nn.Linear(1, 16),
    torch.nn.Tanh(),
    torch.nn.Linear(16, 1)
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


for step in range(1000):
    outputs = model(inputs)

    model_loss: torch.Tensor = ((targets - outputs) ** 2).mean()



    model_loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if (step + 1) % 10 == 0:
        print(f"\n完成第 {step + 1} 次更新后的参数：")

        for name, parameter in model.named_parameters():
            print(name, parameter.detach())


with torch.no_grad():
    final_loss = ((model(inputs)-targets) ** 2).mean()
print("最终损失:", final_loss.item())

with torch.no_grad():
    predictions = model(inputs).squeeze(1)

plt.plot(x.numpy(), y.numpy(), label="Target: x²")
plt.plot(x.numpy(), predictions.numpy(), "--", label="Model prediction")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()



test_inputs = torch.tensor([[0.37], [3.0]])
test_targets = test_inputs ** 2
with torch.no_grad():
    test_predictions = model(test_inputs)
for i in range(len(test_inputs)):
    x_value = test_inputs[i].item()
    predicted = test_predictions[i].item()
    target = test_targets[i].item()

    print(
        f"x={x_value:.2f} | "
        f"预测={predicted:.6f} | "
        f"正确值={target:.6f} | "
        f"绝对误差={abs(predicted - target):.6f}"
    )


test_x = (x[:-1]+x[1:]) / 2
test_inputs = test_x.reshape(-1, 1)
test_targets = test_inputs ** 2

with torch.no_grad():
    test_outputs = model(test_inputs)
    test_mse = ((test_outputs-test_targets) ** 2).mean()
    test_max_error = (test_outputs - test_targets).abs().max()
print("测试点数量:", len(test_x))
print("测试集均方误差:", test_mse.item())
print("测试集最大绝对误差:", test_max_error.item())
plt.show()
