"""Render saved CSV histories without rerunning training."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

output = Path(__file__).parent / "results"
with (output / "baseline.csv").open() as stream:
    rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
steps = [row["step"] for row in rows]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
for axis, key, target in zip(axes[:2], ["w", "b"], [3, 2]):
    axis.plot(steps, [row[key] for row in rows], linewidth=2)
    axis.axhline(target, color="gray", linestyle="--", label=f"target = {target}")
    axis.set(title=f"Parameter {key}", xlabel="Update step", ylabel=key)
    axis.legend()
axes[2].semilogy(steps, [max(row["loss"], 1e-16) for row in rows], linewidth=2)
axes[2].set(title="Mean squared error", xlabel="Update step", ylabel="Loss (log scale)")
for axis in axes:
    axis.grid(alpha=0.25)
fig.suptitle("Learning y = 3x + 2 | CPU float32 | learning rate 0.1")
fig.savefig(output / "training.png", dpi=160)
plt.close(fig)
print(output / "training.png")

paths = sorted(output.glob("compare_*.csv"))
if paths:
    fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    for path in [output / "baseline.csv", *paths]:
        with path.open() as stream:
            history = list(csv.DictReader(stream))
        axis.semilogy([int(row["step"]) for row in history],
                      [max(float(row["loss"]), 1e-16) for row in history],
                      label=path.stem.removeprefix("compare_"))
    axis.set(title="Learning rate and initialization experiments", xlabel="Update step", ylabel="Loss (log scale; zeros shown at 1e-16)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output / "comparisons.png", dpi=160)
    plt.close(fig)
    print(output / "comparisons.png")
