# argumentative-reward-shaping-LBF

Argumentative reward shaping (ArgRS) for explainable reinforcement learning agents in Level-Based Foraging (LBF).

Current state: single-player 5×5 LBF with random, heuristic and tabular Q-learning agents, value iteration as the exact optimum, and a config-driven experiment runner. The argumentation and shaping modules come next.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

```bash
uv run pytest                                                                        # tests
PYTHONPATH=src uv run python -m argrl.experiments.run --config experiments/configs/baseline_5x5.yaml
uv run python scripts/plot_learning_curve.py                                         # learning-curve figure
PYTHONPATH=src uv run python scripts/compute_regret.py --exp baseline_5x5            # regret vs value iteration
uv run python scripts/play_random.py                                                 # watch random agents play
```


