# backward()：使用行为笔记

日期：2026-11-06。数学背景已掌握，本文重点记录 PyTorch 接口的使用行为。接口默认值已对照本地 PyTorch 2.15.0 的内置文档确认。

## 1. 三个调用各自做什么

```python
loss.backward()        # 计算梯度，并累加到参数的 .grad
optimizer.step()       # 读取 .grad，更新优化器管理的参数
optimizer.zero_grad()  # 清理这些参数的 .grad，准备下一批
```

| 调用 | 参数数值 | 参数的 .grad |
| --- | --- | --- |
| `loss.backward()` | 保持原值 | 写入或累加本次梯度 |
| `optimizer.step()` | 根据已有梯度更新 | 保留已有梯度 |
| `optimizer.zero_grad()` | 保持原值 | 默认设为 `None` |

`backward()` 不会自动调用优化器。`step()` 不会自动重新求导，也不会自动清理梯度。

## 2. 梯度存在哪里

`nn.Linear` 创建的权重和偏置是需要梯度的叶参数。通常在首次反向传播前，它们的 `.grad` 是 `None`；反向传播后，`.grad` 是与参数形状相同的 Tensor。

```python
loss.backward()

for name, parameter in model.named_parameters():
    print(name, parameter.grad)
```

`parameter` 是参数本身，`parameter.grad` 是损失对该参数的梯度。打印训练后的参数应使用前者，而不是 `.grad`。

默认情况下，中间结果（例如 `outputs`）通常不保留自己的 `.grad`。检查模型参数的梯度即可；中间量梯度的保留留待需要时学习。

## 3. 为什么每批之后要清理梯度

反向传播默认执行“旧梯度 + 本次梯度”，不是覆盖旧值。

在普通小批量训练中，每批都要依据当前参数和当前批次重新计算梯度。如果没有清理，下一批求导时会混入上一批留下的梯度。

`optimizer.zero_grad()` 默认把梯度设为 `None`，下一次反向传播再创建梯度 Tensor。也可以写：

```python
optimizer.zero_grad(set_to_none=False)
```

这会把已有的梯度 Tensor 清零。`None` 与零张量并非完全等价：优化器会跳过梯度为 `None` 的参数；对零梯度，动量或权重衰减等机制仍可能影响更新。当前实验使用的普通 SGD 没有启用这些额外机制。

## 4. 对应 Lesson 3 的训练顺序

以下代码放在模型、优化器和 loader 已定义之后：

```python
for batch_inputs, batch_targets in loader:
    outputs = model(batch_inputs)
    loss = ((outputs - batch_targets) ** 2).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

每次循环先重新前向计算，再反向求导。因为这里的损失只使用本批样本，所以梯度也是本批损失的梯度。

当前写法在更新后清理梯度，下一批开始时梯度为 `None`。另一种常见写法是在每批开始时清理；关键是本批求导前没有遗留梯度，并且不要在 `backward()` 与 `step()` 之间清理。

## 5. 计算图与梯度 Tensor 是两回事

- **计算图及保存的中间值**：用于计算本次梯度。每批重新执行前向计算，会建立新的求导关系。默认反向传播后会释放本轮反向所需的已保存中间值，不应直接对同一份 `loss` 重复执行 `backward()`。
- **参数的 `.grad`**：保存算出来的梯度，会继续存在，直到被清理或替换。因此下一批即使建立了新图，梯度仍会累加。

普通训练不需要设置 `retain_graph=True`。若要再次求导，先按当前参数重新执行前向计算即可。

`optimizer.zero_grad()` 清理梯度，不负责让模型重新预测。

## 6. loss、item() 与 no_grad()

当前 `.mean()` 将一批误差归约为标量 Tensor，可以直接调用 `loss.backward()`。如果输出包含多个元素，通常先明确如何求和或取平均；直接调用无参数的 `backward()` 会报错。

```python
loss.backward()  # 对 Tensor 求导
print(loss.item())  # 转为普通 Python 数值，仅用于查看或记录
```

`loss.item()` 得到的 Python 数值没有求导关系，不能用它代替 `loss` 调用 `backward()`。

```python
with torch.no_grad():
    predictions = model(inputs)
```

`no_grad()` 让代码块内的前向运算不建立新的求导记录，适合评测。它不会清理已有 `.grad`，也不会把模型参数的 `requires_grad` 改为 `False`。完全在该代码块中算出的普通评测损失不能用于随后直接反向求导。

## 7. 一个最小的行为观察实验

下面是独立示例，不需要改动当前训练文件：

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)

print("求导前:", w.item(), w.grad)

loss = (w - 3) ** 2
loss.backward()
print("求导后:", w.item(), w.grad)

optimizer.step()
print("更新后:", w.item(), w.grad)

optimizer.zero_grad()
print("清理后:", w.item(), w.grad)
```

预期行为（忽略浮点末位）：

| 时刻 | w | w.grad |
| --- | ---: | --- |
| 求导前 | 2.0 | `None` |
| 求导后 | 2.0 | `tensor(-2.)` |
| 更新后 | 2.2 | `tensor(-2.)` |
| 清理后 | 2.2 | `None` |

更新后仍保留的梯度来自更新前的参数点。下一批需要重新前向和反向，得到新参数点对应的新梯度。

## 当前掌握目标

能够区分参数与梯度，解释三次调用的状态变化，并独立写对每批训练的调用顺序。C++ 自动微分引擎、调度和 GPU 内核实现暂不展开。
