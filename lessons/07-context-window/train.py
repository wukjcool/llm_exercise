"""Lesson 7：用两个字预测下一个字。练习构造两个字的上下文窗口和对应的预测目标。"""
import torch

sentences = ["去北京", "在北方"]
vocab = ["去", "北", "京", "在", "方"]

char_to_id = {char: index for index, char in enumerate(vocab)}

ids = torch.tensor([
    [char_to_id[char] for char in sentence]
    for sentence in sentences
])

inputs = ids[:, :2] #每行取前 2 列
targets = ids[:, 2] # 每行去第 3 列

print("输入:", inputs)
print("目标:", targets)

torch.manual_seed(42)

vocab_size = len(vocab)
embedding_dim = 4

model = torch.nn.Sequential(
    torch.nn.Embedding(vocab_size, embedding_dim),
    torch.nn.Flatten(start_dim=1),
    torch.nn.Linear(2 * embedding_dim, vocab_size)
)
scores = model(inputs)
print("分数形状:", scores.shape)

loss_fn = torch.nn.CrossEntropyLoss()

loss = loss_fn(scores, targets)
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


with torch.no_grad():
    embeddings = model[0](inputs)
    predicted_ids = model(inputs).argmax(dim=1).tolist()

print("两条样本中第二个字的向量:", embeddings[:, 1])

for sentence, predicted_id in zip(sentences, predicted_ids):
    print(f"{sentence[:2]}->{vocab[predicted_id]}")