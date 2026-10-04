"""Record production hardware-policy simulations without claiming device execution."""

import argparse
import json
from pathlib import Path

from surveilx.hardware_simulation import simulate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/hardware-simulation.json")
    args = parser.parse_args()
    result = simulate()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Passed {len(result['targets'])} hardware policy scenarios; simulation only")
