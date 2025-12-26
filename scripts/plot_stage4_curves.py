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
        "--dataset",
        default="mnist",
        help="Dataset key used in run_ids (default: mnist)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: docs/figures/stage4_<dataset>_curves.png)",
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

def run_is_complete(run_dir: Path) -> bool:
    return (run_dir / "artifacts" / "summary.json").is_file() and (run_dir / "artifacts" / "metrics.jsonl").is_file()


def validate_runs_exist(results_dir: Path, run_ids: Iterable[str]) -> None:
    missing = [run_id for run_id in run_ids if not run_is_complete(results_dir / run_id)]
    if missing:
        raise FileNotFoundError(f"Missing run dirs under {results_dir}: {missing}")

def resolve_first_complete_run(results_dir: Path, candidates: Iterable[str]) -> str:
    for run_id in candidates:
        if run_is_complete(results_dir / run_id):
            return run_id
    raise FileNotFoundError(f"No complete run found under {results_dir} for candidates: {list(candidates)}")


def dataset_display_name(dataset: str) -> str:
    display = {
        "mnist": "MNIST",
        "fmnist": "FashionMNIST",
        "cifar10": "CIFAR-10",
        "tinyimagenet": "TinyImageNet",
    }
    return display.get(dataset.lower(), dataset)


def build_run_ids(results_dir: Path, dataset: str) -> tuple[list[str], list[str]]:
    prefix = f"stage4_{dataset}_noattack_"
    fedavg = resolve_first_complete_run(results_dir, [f"{prefix}fedavg_v2", f"{prefix}fedavg"])
    baseline_runs = [
        fedavg,
        f"{prefix}krum",
        f"{prefix}median",
        f"{prefix}trimmedmean",
        f"{prefix}clipmedian",
        f"{prefix}cosdefense",
    ]
    comparison_runs = [
        f"{prefix}clipmedian",
        f"stage4_{dataset}_scaling_clipmedian",
    ]
    return baseline_runs, comparison_runs


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_path = Path(args.out) if args.out else Path(f"docs/figures/stage4_{args.dataset}_curves.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_runs, comparison_runs = build_run_ids(results_dir, args.dataset)
    validate_runs_exist(results_dir, baseline_runs + comparison_runs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150, constrained_layout=True)
    dataset_name = dataset_display_name(args.dataset)

    # --- Panel 1: no-attack baselines
    ax = axes[0]
    for run_id in baseline_runs:
        run_dir = results_dir / run_id
        summary = load_summary(run_dir)
        rounds, acc = load_series(run_dir)
        label = summary.get("aggregator", run_id)
        ax.plot(rounds, acc, marker="o", linewidth=1.5, label=label)
    ax.set_title(f"{dataset_name} IID (no attack) baselines")
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

    scaling_summary = load_summary(results_dir / comparison_runs[1])
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

    fig.suptitle(f"Stage4 curves: {dataset_name} (committed evidence runs)", fontsize=12)
    fig.savefig(out_path)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()
