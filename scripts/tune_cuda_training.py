"""Measure bounded real-sample GPU batches before selecting a training recipe."""

import gc
import json
from pathlib import Path
import time

import psutil
import torch

from scripts.benchmark_cuda_training import measure


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new tuning report path")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training required")
    torch.set_num_threads(3)
    battery = psutil.sensors_battery()
    report = {
        "created": time.time(),
        "scope": "Real train samples; float32 loss/backward/AdamW; batch selected on measured throughput with 512 MiB PyTorch memory reserve",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "power_plugged": battery.power_plugged if battery else None,
        "results": [],
        "selected": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    limit = torch.cuda.get_device_properties(0).total_memory / 1024**2 - 512
    for kind in ("yolo", "scratch", "scene"):
        trials = []
        for batch in (1, 2, 4, 8):
            try:
                result = measure(kind, "cuda", 3, batch)
                result["headroom_passed"] = result["peak_reserved_mib"] <= limit
                trials.append(result)
                print(
                    json.dumps(
                        {
                            "architecture": kind,
                            "batch": batch,
                            "samples_per_second": result["samples_per_second"],
                            "reserved_mib": result["peak_reserved_mib"],
                            "headroom_passed": result["headroom_passed"],
                        }
                    ),
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError:
                trials.append({"batch_size": batch, "error": "CUDA out of memory", "headroom_passed": False})
                break
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        successful = [trial for trial in trials if trial["headroom_passed"]]
        if not successful:
            raise RuntimeError(f"No verified batch with sufficient headroom: {kind}")
        best = max(successful, key=lambda row: row["samples_per_second"])
        report["selected"][kind] = {
            "batch_size": best["batch_size"],
            "samples_per_second": best["samples_per_second"],
            "peak_reserved_mib": best["peak_reserved_mib"],
        }
        report["results"].append({"architecture": kind, "trials": trials})
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["state"] = "completed"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    run(parser.parse_args().output)
