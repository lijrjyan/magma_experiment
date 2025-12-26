"""Plot Stage4 baseline curves from committed results.

This script reads `results/<run_id>/artifacts/metrics.jsonl` and produces a
single PNG figure used in `README.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Stage4 curves")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory containing run subfolders (default: results)",
    )
    parser.add_argument(
        "--out",
        default="docs/figures/stage4_mnist_curves.png",
        help="Output PNG path (default: docs/figures/stage4_mnist_curves.png)",
    )
    return parser.parse_args()


def load_series(run_dir: Path) -> Tuple[List[int], List[float]]:
    metrics_path = run_dir / "artifacts" / "metrics.jsonl"
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
        raise ValueError(f"No test_acc found in {metrics_path}")
    sorted_rounds, sorted_acc = zip(*pairs)
    return list(sorted_rounds), list(sorted_acc)


def load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "artifacts" / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def validate_runs_exist(results_dir: Path, run_ids: Iterable[str]) -> None:
    missing = [run_id for run_id in run_ids if not (results_dir / run_id).exists()]
    if missing:
        raise FileNotFoundError(f"Missing run dirs under {results_dir}: {missing}")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_runs = [
        "stage4_mnist_noattack_fedavg",
        "stage4_mnist_noattack_krum",
        "stage4_mnist_noattack_median",
        "stage4_mnist_noattack_trimmedmean",
        "stage4_mnist_noattack_clipmedian",
        "stage4_mnist_noattack_cosdefense",
    ]
    comparison_runs = [
        "stage4_mnist_noattack_clipmedian",
        "stage4_mnist_scaling_clipmedian",
    ]
    validate_runs_exist(results_dir, baseline_runs + comparison_runs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150, constrained_layout=True)

    # --- Panel 1: no-attack baselines
    ax = axes[0]
    for run_id in baseline_runs:
        run_dir = results_dir / run_id
        summary = load_summary(run_dir)
        rounds, acc = load_series(run_dir)
        label = summary.get("aggregator", run_id)
        ax.plot(rounds, acc, marker="o", linewidth=1.5, label=label)
    ax.set_title("MNIST IID (no attack) baselines")
    ax.set_xlabel("Round")
    ax.set_ylabel("Server test acc (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- Panel 2: ClipMedian under scaling attack
    ax = axes[1]
    for run_id in comparison_runs:
        run_dir = results_dir / run_id
        summary = load_summary(run_dir)
        rounds, acc = load_series(run_dir)
        attack = summary.get("attack_strategy", "none")
        label = f"{summary.get('aggregator', run_id)} ({attack})"
        ax.plot(rounds, acc, marker="o", linewidth=1.5, label=label)

    scaling_summary = load_summary(results_dir / "stage4_mnist_scaling_clipmedian")
    attack_start = scaling_summary.get("attack_start_round")
    if isinstance(attack_start, int):
        ax.axvline(attack_start, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(
            attack_start + 0.05,
            5,
            f"attack_start={attack_start}",
            fontsize=8,
            rotation=90,
            va="bottom",
            ha="left",
        )

    ax.set_title("ClipMedian: no attack vs scaling attack")
    ax.set_xlabel("Round")
    ax.set_ylabel("Server test acc (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Stage4 curves (committed evidence runs)", fontsize=12)
    fig.savefig(out_path)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()

