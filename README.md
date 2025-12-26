# MAGMA Execution Workspace

This repository hosts the engineering effort for evaluating MAGMA (Manifold-Aware
Geometric Aggregation) against Dual Defense (DDFed). All implementers **must**
follow the staged workflow captured in `EXECUTION_PLAN.md`, commit after each
stage, and push to `git@github.com:lijrjyan/magma_experiment.git`.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run a sanity experiment (results stored under `results/<run_id>/`):

```bash
python scripts/run_exp.py --config configs/mnist.yaml --run-name mnist_sanity
```

For TinyImageNet (after downloading/extracting the dataset under `./data`):

```bash
python scripts/run_exp.py --config configs/tinyimagenet.yaml --run-name tinyimagenet_sanity
```

Key directories:

- `configs/`: dataset/attack/aggregator/FHE parameters (MNIST/FMNIST/CIFAR10/TinyImageNet)
- `scripts/`: experiment runner plus future sweep/summarize/plot utilities
- `magma/`: geometric aggregator modules (distance, clustering, FHE backend)
- `baselines/`: FedAvg, Krum, Median, TrimmedMean, ClipMedian, CosDefense, DDFed
- `attacks/`: IPM, ALIE, Scaling, Label Flip
- `results/`: auto-generated logs, metrics, summaries, and future figures/tables
- `docs/`: MAGMA vs DDFed comparison notes and threat-model alignment docs

## Stage Discipline

- Every stage listed in `EXECUTION_PLAN.md` must end with:
  ```bash
  git status
  git add -A
  git commit -m "stageX: <summary>"
  git push origin main
  ```
- No work is considered complete until code, docs, and logs are pushed.
- Keep `results/` deterministic: each run emits `logs/train.log`,
  `tensorboard/`, and `artifacts/{metrics.jsonl,summary.json,resolved_config.yaml}`.

## Reference Plan

The full step-by-step execution roadmap (including Stage 1-11 deliverables,
metrics, plotting requirements, and ablation expectations) lives in
`EXECUTION_PLAN.md`. Read it before touching the codebase and update both the
plan and this README if requirements change.
