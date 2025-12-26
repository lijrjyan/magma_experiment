# Quick Look

This file is a fast way to browse the *committed* evidence runs and their key metrics.

## Stage4 results (MNIST, IID, seed=7890, rounds=5, local_epochs=1)

Notes:
- `test_acc` is reported in percent (0-100).
- Values are read from `results/<run_id>/artifacts/summary.json`.

| run_id | aggregator | attack | final_test_acc | best_test_acc | min_test_acc_after_attack |
| --- | --- | --- | ---: | ---: | ---: |
| `stage4_mnist_noattack_fedavg` | `fedavg` | `none` | 67.42 | 67.42 |  |
| `stage4_mnist_noattack_krum` | `krum` | `none` | 69.06 | 69.06 |  |
| `stage4_mnist_noattack_median` | `median` | `none` | 68.43 | 68.43 |  |
| `stage4_mnist_noattack_trimmedmean` | `trimmed_mean` | `none` | 67.57 | 67.57 |  |
| `stage4_mnist_noattack_clipmedian` | `clipping_median` | `none` | 68.39 | 68.39 |  |
| `stage4_mnist_noattack_cosdefense` | `cos_defense` | `none` | 69.68 | 69.68 |  |
| `stage4_mnist_scaling_clipmedian` | `clipping_median` | `model_poisoning_scaling` | 68.16 | 68.16 | 54.92 |

![Stage4 MNIST curves](docs/figures/stage4_mnist_curves.png)

## Plot regeneration

From `magma_experiment/`:

```bash
python scripts/plot_stage4_curves.py
```

