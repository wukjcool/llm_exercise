"""Compare learning rates/initial values and verify the lesson's mathematics."""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train import save_history, train

output = Path(__file__).parent / "results"
output.mkdir(exist_ok=True)
steps = 101
cases = [
    ("baseline", 0.1, 0.0, 0.0),
    ("slow_lr_0.01", 0.01, 0.0, 0.0),
    ("fast_lr_0.5", 0.5, 0.0, 0.0),
    ("oscillating_lr_0.7", 0.7, 0.0, 0.0),
    ("diverging_lr_0.8", 0.8, 0.0, 0.0),
    ("init_10_minus8", 0.1, 10.0, -8.0),
]

# Independent analytical gradient and accumulation checks, using float64.
x = torch.linspace(-2, 2, 100, dtype=torch.float64)
y = 3 * x + 2
w = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
loss = ((w * x + b - y) ** 2).mean()
loss.backward()
expected = torch.stack((2 * ((w.detach() * x + b.detach() - y) * x).mean(),
                        2 * (w.detach() * x + b.detach() - y).mean()))
actual = torch.stack((w.grad.clone(), b.grad.clone()))
torch.testing.assert_close(actual, expected)
((w * x + b - y) ** 2).mean().backward()
torch.testing.assert_close(torch.stack((w.grad, b.grad)), 2 * expected)

# With symmetric x, L = a(w-3)^2 + (b-2)^2; use this exact recurrence
# to check every saved state. In the divergent run, huge residuals cause
# float32 cancellation in the bias gradient, so only check w's recurrence.
a = x.square().mean().item()
summary = []
for name, lr, initial_w, initial_b in cases:
    history = train(lr, initial_w, initial_b, steps)
    for row in history:
        k = row["step"]
        exact_w = 3 + (initial_w - 3) * (1 - 2 * a * lr) ** k
        exact_b = 2 + (initial_b - 2) * (1 - 2 * lr) ** k
        actual_values = [row["w"]] if lr == 0.8 else [row["w"], row["b"]]
        expected_values = [exact_w] if lr == 0.8 else [exact_w, exact_b]
        torch.testing.assert_close(torch.tensor(actual_values, dtype=torch.float64),
                                   torch.tensor(expected_values, dtype=torch.float64),
                                   rtol=5e-5, atol=5e-6)
    if name != "baseline":
        save_history(history, output / f"compare_{name}.csv")
    if name in ("baseline", "fast_lr_0.5", "init_10_minus8"):
        assert abs(history[-1]["w"] - 3) < 1e-5
        assert abs(history[-1]["b"] - 2) < 1e-5
    if name == "diverging_lr_0.8":
        assert history[-1]["loss"] > history[0]["loss"] * 1e6
    summary.append({"name": name, "learning_rate": lr, "initial_w": initial_w,
                    "initial_b": initial_b, "steps": steps, "seed": 42,
                    "device": "cpu", "dtype": "float32", "samples": 100,
                    "x_range": [-2, 2], "initial_loss": history[0]["loss"],
                    "final": history[-1]})

result = {"gradient_at_origin": actual.tolist(), "gradient_accumulation_verified": True,
          "closed_form_verified_for_convergent_runs": True,
          "divergent_w_recurrence_verified": True,
          "divergent_bias_note": "Large residuals cause float32 cancellation; b can deviate from the exact recurrence.", "a": a,
          "learning_rate_convergence_upper_bound": 1 / a, "experiments": summary}
(output / "experiments.json").write_text(json.dumps(result, indent=2) + "\n")
for run in summary:
    final = run["final"]
    print(f"{run['name']:24s} w={final['w']:.8g} b={final['b']:.8g} loss={final['loss']:.8g}")
print("PASS: analytical gradients, accumulation, convergent recurrences, divergent w recurrence and loss growth")
