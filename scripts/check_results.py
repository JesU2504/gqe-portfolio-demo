"""Audit saved results without CUDA-Q or plotting dependencies."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from results import validation_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    failures, total = 0, 0
    for path in sorted(args.folder.glob("*.json")):
        if path.stem.endswith("_trajectory"):
            continue
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "best_energy" not in data:
            continue
        total += 1
        errors = validation_errors(data)
        failures += bool(errors)
        print(f"{path.name}: " + ("INVALID: " + "; ".join(errors) if errors else "checks passed"))
    print(f"{total} result files; {failures} invalid.")
    return 1 if failures or not total else 0


if __name__ == "__main__":
    sys.exit(main())
