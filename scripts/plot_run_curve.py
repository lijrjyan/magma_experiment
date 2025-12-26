"""Plot a single run's server test accuracy curve.

Reads `results/<run_id>/artifacts/metrics.jsonl` and writes a PNG figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a single run curve")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory containing run subfolders (default: results)",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run id (subfolder under results_dir)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: docs/figures/<run_id>_curve.png)",
    )
    return parser.parse_args()


def load_series(metrics_path: Path) -> Tuple[List[int], List[float]]:
    rounds: List[int] = []
    acc: List[float] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            round_idx = int(record["round"])
            test_acc = record.get("server", {}).get("test_acc")
            if test_acc is None:
                continue
            rounds.append(round_idx)
            acc.append(float(test_acc))
    pairs = sorted(zip(rounds, acc), key=lambda x: x[0])
    if not pairs:
        raise ValueError(f"No server.test_acc found in {metrics_path}")
    sorted_rounds, sorted_acc = zip(*pairs)
    return list(sorted_rounds), list(sorted_acc)


def load_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    run_dir = results_dir / args.run_id

    metrics_path = run_dir / "artifacts" / "metrics.jsonl"
    summary_path = run_dir / "artifacts" / "summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    out_path = Path(args.out) if args.out else Path(f"docs/figures/{args.run_id}_curve.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    rounds, acc = load_series(metrics_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=160, constrained_layout=True)
    ax.plot(rounds, acc, marker="o", linewidth=1.5)

    dataset = summary.get("dataset", "dataset")
    aggregator = summary.get("aggregator", "aggregator")
    attack = summary.get("attack_strategy", "none")
    title = f"{args.run_id} ({dataset}, {aggregator}, attack={attack})"
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Server test acc (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    attack_start = summary.get("attack_start_round")
    if attack != "none" and isinstance(attack_start, int):
        ax.axvline(attack_start, color="black", linestyle="--", linewidth=1, alpha=0.6)

    fig.savefig(out_path)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()

