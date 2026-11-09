# Softmax、对数概率与交叉熵

Lesson 5 用三个候选字符「北」「京」「上」练习分类。模型为每个候选字符给出一个原始分数，称为 logit。正确答案设为「京」，类别编号是 `1`。

| 量 | 含义 | 北 | 京 | 上 |
| --- | --- | ---: | ---: | ---: |
| 原始分数 | 候选类别的 logits | 0.2 | 1.4 | −0.3 |
| `softmax` 输出 | 各类别的概率 | 0.20298 | 0.67391 | 0.12311 |
| `log_softmax` 输出 | 各类别的对数概率 | −1.59466 | −0.39466 | −2.09466 |

## 概率与对数概率

`softmax` 可以作为分类模型的输出激活函数。它把同一条样本的整组分数转换成一组总和为 1 的概率。给定分数向量 $z$，第 $i$ 个类别的概率是

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

如果只想求「京」的概率，直接算这个比值即可。

$$
P(\text{京}) = \frac{e^{1.4}}{e^{0.2}+e^{1.4}+e^{-0.3}} \approx 0.67391.
$$

**对数概率**是概率的自然对数 $\ell_i=\ln p_i$。**`log_softmax`** 是从原始分数计算对数概率的函数。

$$
\ell_i = \operatorname{log\_softmax}(z)_i
       = z_i-\ln\!\left(\sum_j e^{z_j}\right),
\qquad p_i=e^{\ell_i}.
$$

至少有两个类别且所有分数有限时，数学上每个 $p_i$ 都严格介于 0 和 1 之间，所以每个 $\ell_i$ 都是负数。概率为 1 时，对数概率为 0。

## 交叉熵与 PyTorch 用法

在这个单样本、单一正确类别的例子中，交叉熵取正确类别的对数概率并加负号。正确答案是「京」，所以损失为 $-\ln P(\text{京})\approx 0.39466$。模型给正确类别的概率越高，损失越小。

`scores` 的形状是 `[1, 3]`，表示一个样本和三个候选类别。`dim=1` 指定类别所在的维度。`CrossEntropyLoss` 直接接收原始分数与正确类别编号，在内部计算对数概率和损失。

```python
import torch

scores = torch.tensor([[0.2, 1.4, -0.3]])
target = torch.tensor([1])  # 「京」的编号

probs = torch.softmax(scores, dim=1)
log_probs = torch.log_softmax(scores, dim=1)
loss = torch.nn.CrossEntropyLoss()(scores, target)

print(probs)
print(log_probs)
print(loss.item())  # 约 0.394659
```
