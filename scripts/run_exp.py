"""Unified experiment entrypoint for MAGMA / Dual Defense studies.

Stage 1 focuses on orchestration and logging so every subsequent stage
in the execution plan has a reproducible backbone to build upon.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MAGMA/DDFed experiment")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name to override the auto-generated identifier",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional key=value overrides using dot notation (repeatable)",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_override(config: Dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}', expected key=value")
    key_path, raw_value = override.split("=", 1)
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError:
        value = raw_value
    keys = key_path.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def ensure_dirs(run_dir: Path) -> Dict[str, Path]:
    log_dir = run_dir / "logs"
    tb_dir = run_dir / "tensorboard"
    output_dir = run_dir / "artifacts"
    for folder in (run_dir, log_dir, tb_dir, output_dir):
        folder.mkdir(parents=True, exist_ok=True)
    return {
        "run": run_dir,
        "logs": log_dir,
        "tensorboard": tb_dir,
        "artifacts": output_dir,
    }


def build_run_id(config: Dict[str, Any], run_name: Optional[str]) -> str:
    if run_name:
        return run_name
    dataset = config.get("data", {}).get("dataset", "dataset")
    agg = config.get("aggregation", {}).get("method", "agg")
    attack = config.get("attacks", {}).get("strategy", "none")
    seed = config.get("experiment", {}).get("seed", 0)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{dataset}-{agg}-{attack}-seed{seed}-{timestamp}"


def compose_simulation_config(
    config: Dict[str, Any],
    *,
    log_dir: Path,
    tensorboard_dir: Path,
    run_id: str,
):
    data_cfg = config.get("data", {})
    partition_cfg = data_cfg.get("partition", {})
    train_cfg = config.get("training", {})
    attack_cfg = config.get("attacks", {})
    agg_cfg = config.get("aggregation", {})
    experiment_cfg = config.get("experiment", {})
    runtime_cfg = config.get("runtime", {})

    sim_config = {
        "num_clients": data_cfg.get("num_clients", 100),
        "clients_per_round": data_cfg.get("clients_per_round", 10),
        "dataset": data_cfg.get("dataset", "mnist"),
        "fusion": agg_cfg.get("method", "dual_defense"),
        "training_round": train_cfg.get("rounds", 100),
        "local_epochs": train_cfg.get("local_epochs", 3),
        "optimizer": train_cfg.get("optimizer", "sgd"),
        "learning_rate": train_cfg.get("learning_rate", 0.01),
        "batch_size": train_cfg.get("batch_size", 64),
        "data_dir": data_cfg.get("dir", "./data"),
        "use_fake_data": bool(data_cfg.get("use_fake_data", False)),
        "fake_train_size": int(data_cfg.get("fake_train_size", 2000)),
        "fake_test_size": int(data_cfg.get("fake_test_size", 500)),
        "strict_data": bool(data_cfg.get("strict_data", False)),
        "partition_type": partition_cfg.get("type", "iid"),
        "partition_dirichlet_beta": partition_cfg.get("beta", 0.5),
        "regularization": train_cfg.get("regularization", 1e-5),
        "attacker_ratio": attack_cfg.get("attacker_ratio", 0.0),
        "attacker_strategy": attack_cfg.get("strategy", "none"),
        "attack_start_round": attack_cfg.get("attack_start_round", 50),
        "label_flipping_ratio": attack_cfg.get("label_flipping_ratio", 0.0),
        "epsilon": attack_cfg.get("epsilon", 0.01),
        "ipm_multiplier": attack_cfg.get("ipm_multiplier", 5),
        "alie_epsilon": attack_cfg.get("alie_epsilon", 0.1),
        "device": runtime_cfg.get("device", "cuda"),
        "seed": experiment_cfg.get("seed", 0),
        "data_balance_strategy": data_cfg.get("data_balance_strategy", "balanced"),
        "imbalance_percentage": data_cfg.get("imbalance_percentage", 0.1),
        "use_fhe": bool(agg_cfg.get("use_fhe", False)),
        "log_dir": str(log_dir),
        "plot_partition_stats": bool(data_cfg.get("plot_partition_stats", False)),
        "min_samples_per_client": partition_cfg.get("min_samples_per_client", 100),
        "max_samples_per_client": partition_cfg.get("max_samples_per_client", 2000),
        "fake_data_size_multiplier": agg_cfg.get("fake_data_size_multiplier", 10.0),
    }
    sim_config["filename_core"] = run_id
    sim_config["results_dir"] = str(log_dir.parent)
    sim_config["tensorboard_dir"] = str(tensorboard_dir)
    return sim_config


def write_metrics(metrics: Dict[int, Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for round_idx in sorted(metrics.keys()):
            record = {"round": round_idx}
            record.update(metrics[round_idx])
            json.dump(record, handle)
            handle.write("\n")


def summarize_run(
    sim_config: Dict[str, Any],
    metrics: Dict[int, Dict[str, Any]],
    *,
    summary_path: Path,
    config_path: Path,
    resolved_config: Dict[str, Any],
    run_id: str,
) -> None:
    rounds = sorted(metrics.keys())
    server_acc = [metrics[r]["server"].get("test_acc") for r in rounds]
    final_acc = server_acc[-1] if server_acc else None
    best_acc = max(server_acc) if server_acc else None
    attack_start = sim_config.get("attack_start_round", 0)
    after_attack = [acc for idx, acc in zip(rounds, server_acc) if idx >= attack_start and acc is not None]
    min_after_attack = min(after_attack) if after_attack else None

    summary = {
        "run_id": run_id,
        "config_file": str(config_path),
        "final_test_acc": final_acc,
        "best_test_acc": best_acc,
        "min_test_acc_after_attack": min_after_attack,
        "rounds": len(rounds),
        "dataset": sim_config.get("dataset"),
        "aggregator": sim_config.get("fusion"),
        "attack_strategy": sim_config.get("attacker_strategy"),
        "attacker_ratio": sim_config.get("attacker_ratio"),
        "seed": sim_config.get("seed"),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    resolved_path = summary_path.parent / "resolved_config.yaml"
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(str(config_path))
    for override in args.override:
        apply_override(cfg, override)

    run_id = build_run_id(cfg, args.run_name)
    results_root = Path(cfg.get("experiment", {}).get("output_dir", "results"))
    run_dir = results_root / run_id
    dirs = ensure_dirs(run_dir)
    log_file = dirs["logs"] / "train.log"
    os.environ["LOG_FILE_NAME"] = str(log_file)

    # Late imports to ensure LOG_FILE_NAME is visible before logger instantiation.
    from utils.logging import setup_tensorboard  # noqa: E402
    from fl import SimulationFL  # noqa: E402

    sim_config = compose_simulation_config(
        cfg,
        log_dir=dirs["logs"],
        tensorboard_dir=dirs["tensorboard"],
        run_id=run_id,
    )

    tensorboard_writer = None
    if cfg.get("logging", {}).get("tensorboard", True):
        tensorboard_writer = setup_tensorboard(str(dirs["tensorboard"]), run_id)
    sim_config["tensorboard"] = tensorboard_writer
    sim_config["filename_core"] = run_id
    sim_config["log_dir"] = str(dirs["logs"])

    simulation = SimulationFL(sim_config)
    simulation.start()

    metrics_path = dirs["artifacts"] / "metrics.jsonl"
    write_metrics(simulation.metrics, metrics_path)
    summary_path = dirs["artifacts"] / "summary.json"
    summarize_run(
        sim_config,
        simulation.metrics,
        summary_path=summary_path,
        config_path=config_path,
        resolved_config=cfg,
        run_id=run_id,
    )
    print(f"Run complete. Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
