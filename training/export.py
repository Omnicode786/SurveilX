import argparse
import json
from pathlib import Path

import numpy as np
import torch

from surveilx.accelerators import ONNXExecutor
from training.models import SVANet


def export(run):
    run = Path(run)
    metadata = json.loads((run / "manifest.json").read_text())
    if metadata.get("representation") == "scene_clip":
        raise ValueError("Scene-clip ONNX export is not yet validated; use the PyTorch scene runtime")
    model = SVANet(classes=len(metadata["classes"]), **metadata["architecture"])
    model.load_state_dict(torch.load(run / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()

    class EventExport(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, clips, boxes, context):
            return self.inner(clips, boxes, context)["event"]

    wrapper = EventExport(model)
    contract = metadata.get("input_contract", {"frames": 8, "entities": 2, "image_size": 64})
    frames, entities, size = contract["frames"], contract["entities"], contract["image_size"]
    args = (torch.rand(1, frames, 3, size, size), torch.rand(1, frames, entities, 4), torch.rand(1, 4))
    target = run / "event.onnx"
    torch.onnx.export(
        wrapper,
        args,
        str(target),
        input_names=["clips", "boxes", "context"],
        output_names=["event"],
        opset_version=18,
        dynamo=False,
    )
    executor = ONNXExecutor(target, ["CPUExecutionProvider"])
    result, latency = executor.infer(dict(zip(["clips", "boxes", "context"], [a.numpy() for a in args])))
    with torch.no_grad():
        expected = wrapper(*args).numpy()
    np.testing.assert_allclose(result[0], expected, rtol=1e-4, atol=1e-5)
    report = {
        "onnx_parity": True,
        "single_inference_ms": latency,
        "providers": executor.providers,
        "scope": "CPU numerical parity on a random input; not an accuracy benchmark",
    }
    (run / "export.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    print(export(parser.parse_args().run))
