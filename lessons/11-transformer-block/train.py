"""Lesson 11 前馈网络与 Transformer Block

学习目标
1. 用前馈网络（FFN）分别处理每个位置的特征。
2. 观察四维输入经过十六维隐藏层和 GELU 后回到四维。
3. 将前馈网络与注意力、残差连接和 LayerNorm 组合。

第一个练习
准备形状为 [2, 4] 的输入，编写 Linear(4, 16)、GELU、
Linear(16, 4) 组成的前馈网络，检查输出形状是否为 [2, 4]。
"""


import torch

Q = torch.tensor([[1., 3.],     # 去
                  [5., 7.]])    # 北

q_head_1 = Q[:, :1]
q_head_2 = Q[:, 1:]

print("第一个头:", q_head_1)
print("第二个头:", q_head_2)

K = torch.tensor([[2., 4.],
                  [6., 8.]])
V = torch.tensor([[3., 7.],
                  [11., 15.]])
k_head_1 = K[:, :1]
k_head_2 = K[:, 1:]
v_head_1 = V[:, :1]
v_head_2 = V[:, 1:]

print("头 1 的 K 和 V:", k_head_1, v_head_1)
print("头 2 的 K 和 V:", k_head_2, v_head_2)

scores_head_1 = q_head_1 @ k_head_1.T
scores_head_2 = q_head_2 @ k_head_2.T

print("头 1 的分数:", scores_head_1)
print("头 2 的分数:", scores_head_2)

weights_head_1 = torch.softmax(scores_head_1, dim=-1)
weights_head_2 = torch.softmax(scores_head_2, dim=-1)

print("头 1 中「去」的权重:", weights_head_1[0])
print("头 2 中「去」的权重:", weights_head_2[0])

output_head_1 = weights_head_1 @ v_head_1
output_head_2 = weights_head_2 @ v_head_2

print("头 1 中「去」的输出:", output_head_1[0])
print("头 2 中「去」的输出:", output_head_2[0])

output = torch.cat([output_head_1, output_head_2], dim=-1)
print("两个头拼接后的输出:", output)
print("输出形状:", output.shape)

W_o = torch.tensor([[1., 0., 1., 1.],
                    [0., 1., 1., -1.]])
projected_output = output @ W_o
print("投影后的输出:", projected_output)
print("输出形状:", projected_output.shape)

X = torch.tensor([[1., 2., 3., 4.],   # 去
                  [5., 6., 7., 8.]])  # 北

residual_output = X + projected_output
print("残差相加后的输出:", residual_output)
print("输出形状:", residual_output.shape)

layer_norm = torch.nn.LayerNorm(4, elementwise_affine=False)
normalized_output = layer_norm(residual_output)

print("归一化后的输出:", normalized_output)
print("每行均值:", normalized_output.mean(dim=-1))
print("每行方差:", normalized_output.var(dim=-1, unbiased=False))

torch.manual_seed(42)

ffn = torch.nn.Sequential(
    torch.nn.Linear(4, 16),
    torch.nn.GELU(),
    torch.nn.Linear(16, 4)
)

ffn_output = ffn(normalized_output)

print("FFN 输出:", ffn_output)
print("FFN 输出形状:", ffn_output.shape)
