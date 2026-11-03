"""Lesson 1: fit y = 3x + 2 with scalar parameters and autograd."""
import argparse
import csv
from pathlib import Path

import torch


def train(learning_rate=0.1, initial_w=0.0, initial_b=0.0, steps=101):
    torch.manual_seed(42)
    x = torch.linspace(-2, 2, 100)
    y = 3 * x + 2
    w = torch.tensor(float(initial_w), requires_grad=True)
    b = torch.tensor(float(initial_b), requires_grad=True)
    history = []

    # Step 0 is the initial state; each later row follows one parameter update.
    for step in range(steps + 1):
        prediction = w * x + b
        loss = ((prediction - y) ** 2).mean()
        history.append({"step": step, "w": w.item(), "b": b.item(), "loss": loss.item()})
        if step == steps:
            break

        loss.backward()
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

    return history


def save_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "w", "b", "loss"])
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--initial-w", type=float, default=0.0)
    parser.add_argument("--initial-b", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=101)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "lessons/01-linear-model/results/baseline.csv")
    args = parser.parse_args()
    history = train(args.lr, args.initial_w, args.initial_b, args.steps)
    save_history(history, args.output)
    for row in history:
        if row["step"] % 10 == 0 or row["step"] == args.steps:
            print(f"Step {row['step']:3d} | w={row['w']:.6f} | b={row['b']:.6f} | loss={row['loss']:.10g}")
