"""Run a real forward/backward calculation on CPU and, when available, MPS."""
import json
import platform
from pathlib import Path

import torch


def check(device):
    x = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
    loss = x.square().sum()
    loss.backward()
    assert loss.item() == 5.0
    assert torch.allclose(x.grad, torch.tensor([2.0, 4.0], device=device))
    return {"device": str(x.device), "loss": loss.item(), "gradient": x.grad.cpu().tolist()}


result = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "torch": torch.__version__,
    "mps_built": torch.backends.mps.is_built(),
    "mps_available": torch.backends.mps.is_available(),
    "cpu": check("cpu"),
}
if result["mps_available"]:
    result["mps"] = check("mps")
output = Path(__file__).parent / "results"
output.mkdir(exist_ok=True)
(output / "environment.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
