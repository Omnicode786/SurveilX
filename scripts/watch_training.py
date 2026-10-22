"""Read-only terminal progress viewer; closing it never interrupts training."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from surveilx.generations import read_generation, read_queue


def tail(path):
    if not path.is_file():
        return "Preparing inputs; the first training log has not been written yet."
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 8192))
        value = stream.read().decode("utf-8", errors="replace")
    value = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", value)
    rows = [line.strip() for line in value.replace("\r", "\n").splitlines() if line.strip()]
    return "\n".join(rows[-3:])


def snapshot(data, names):
    lines = [
        "SURVEILX | LIVE TRAINING PROGRESS",
        time.strftime("Updated %H:%M:%S"),
        "Refresh: 2 seconds. Ctrl+C or closing this window stops only the viewer.",
        "Epoch limits are maximums; validation patience can stop training earlier.",
        "",
    ]
    for name in names:
        directory = data / "generations" / name
        try:
            if (directory / "status.json").exists():
                run = read_generation(directory / "status.json", data)
            else:
                run = read_queue(directory / "queue.json")
                content = (directory / "plan.json").read_bytes()
                queue = json.loads((directory / "queue.json").read_text())
                if hashlib.sha256(content).hexdigest() != queue["plan_sha256"]:
                    raise ValueError("Frozen queue plan changed")
                run["jobs"] = [
                    {
                        "version": job["version"],
                        "state": "queued",
                        "completed_epochs": 0,
                        "epochs": job["options"]["epochs"],
                    }
                    for job in json.loads(content)["jobs"]
                ]
            lines.append(f"{name}: {run['state'].upper().replace('_', ' ')}")
            if run.get("error"):
                lines.append(f"  ERROR: {run['error']}")
            for job in run["jobs"]:
                lines.append(
                    f"  {job['state'].upper():<14} {job['version']}  "
                    f"epochs {job['completed_epochs']}/{job.get('epochs') or '?'}"
                )
                comparison = job.get("comparison")
                lines.extend(f"    {row}" for row in format_epoch(job.get("latest_epoch")))
                if comparison:
                    label = "Test AP50" if comparison["metric"] == "map50" else "Test accuracy"
                    lines.append(
                        f"    {label}: {comparison['before']:.4f} -> {comparison['after']:.4f} "
                        f"({comparison['delta']:+.4f}); candidate requires separate acceptance"
                    )
                if job["state"] == "running":
                    lines.extend(
                        f"    {row}" for row in tail(directory / f"{job['version']}.log").splitlines()
                    )
                if job.get("error"):
                    lines.append(f"    ERROR: {job['error']}")
            lines.append("")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            lines.extend([f"{name}: unable to read progress ({exc})", ""])
    return "\n".join(lines)


def format_epoch(row):
    """Missing legacy components stay unavailable; never infer historical losses."""
    if not row:
        return []
    def number(value):
        return f"{value:.5g}" if isinstance(value, (int, float)) else "not recorded"
    metrics = row.get("validation_metrics", {})
    lines = [f"validation AP50 {number(row.get('validation_map50'))}; "
             f"P/R/F1 {number(metrics.get('precision'))}/{number(metrics.get('recall'))}/{number(metrics.get('f1'))}; "
             f"LR {number(row.get('learning_rate'))}; elapsed {number(row.get('elapsed_seconds'))}s"]
    training = row.get("training", {})
    losses = {"loss": row.get("train_loss"), **training.get("losses", {})}
    keys = ("box_regression", "classification", "distribution_focal") if row.get("metric_source") else ("loss", "objectness", "classification", "box_giou", "box_l1")
    lines.append("train losses " + ", ".join(f"{key}={number(losses.get(key))}" for key in keys))
    if training.get("positive_assignments") is not None:
        lines.append(f"positive assignments {training.get('positive_assignments')}; "
                     f"FPN {training.get('positives_per_level')}; unassigned objects {training.get('unassigned_objects')}")
    if row.get("learning_rate_groups"):
        lines.append("LR ranges: " + ", ".join(
            f"group {key} {number(value.get('minimum'))}..{number(value.get('maximum'))}"
            for key, value in row["learning_rate_groups"].items()))
    if row.get("optimizer_step_attempts") is not None:
        lines.append(f"optimizer step attempts {row['optimizer_step_attempts']}; "
                     f"batch {row.get('physical_batch_size')}; nominal effective batch {row.get('nominal_effective_batch_size')}")
    quality = training.get("score_quality", {})
    if quality:
        lines.append(f"score quality: {quality}")
    if metrics.get("per_class"):
        lines.append("class AP50: " + ", ".join(f"{item.get('class_name', item['class_id'])}={number(item.get('ap50'))}"
                     for item in metrics["per_class"]))
    selection = row.get("selection", {})
    if selection:
        lines.append(f"best epoch {selection['best_epoch']}; stale {selection['stale_epochs']}/{selection.get('patience')}; "
                     f"stop: {selection.get('stopping_reason') or 'not stopped'}")
    return lines


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", nargs="*", default=["cuda-accuracy-g4a", "cuda-accuracy-g4b"])
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", name) for name in args.campaigns):
        parser.error("Campaign names must be simple directory names")
    if os.name == "nt" and not args.once:
        ctypes.windll.kernel32.SetConsoleTitleW("SurveilX - Live Training Progress")
        from colorama import just_fix_windows_console

        just_fix_windows_console()
    gpu, last_gpu = "GPU telemetry pending", 0
    try:
        while True:
            if time.monotonic() - last_gpu >= 10:
                try:
                    gpu = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader",
                        ],
                        text=True,
                        timeout=2,
                        creationflags=0x08000000 if os.name == "nt" else 0,
                    ).strip()
                except (OSError, subprocess.SubprocessError):
                    gpu = "GPU telemetry unavailable"
                last_gpu = time.monotonic()
            if not args.once:
                print("\033[2J\033[H", end="")
            print(snapshot(args.data_dir, args.campaigns), "\nGPU:", gpu, flush=True)
            if args.once:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nProgress viewer stopped. Training continues independently.")


if __name__ == "__main__":
    main()
