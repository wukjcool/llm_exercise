from pathlib import Path

import torch

save_path = Path(__file__).with_name("next_token.pt")
saved = torch.load(save_path, map_location="cpu", weights_only=True)
vocab: list[str] = saved["vocab"]


model = torch.nn.Embedding(len(vocab), len(vocab))
model.load_state_dict(saved["weights"])
model.eval()

# current_char = "北"
# input_id = torch.tensor([vocab.index(current_char)])

# with torch.no_grad():
#     next_id = int(model(input_id).argmax(dim=1).item())

# print(f"{current_char} -> {vocab[next_id]}")

temperature = 2

generted = "北"
with torch.no_grad():
    for _ in range(1):
        current_char = generted[-1]
        input_id = torch.tensor([vocab.index(current_char)])
        # next_id = int(model(input_id).argmax(dim=1).item())
        # print(f"{current_char} -> {vocab[next_id]}")

        scores = model(input_id)
        # probs = torch.softmax(scores, dim=1)
        probs = torch.softmax(scores / temperature, dim=1)
        print(current_char, probs.tolist())
        next_id = int(torch.multinomial(probs, num_samples=1).item())

        generted += vocab[next_id]

print(generted)