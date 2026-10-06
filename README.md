# Learning to Allocate Points on a Game Skill Tree: Exploring Amortized Combinatorial Optimization with Graph RL

Graduate research project, preliminary results.

## Problem Statement

Action role-playing games such as *Path of Exile 2* let players build a
character by spending a limited budget of points on a large passive skill
tree (a graph with hundreds to thousands of nodes). A build must start at a
fixed root node, and each new allocation must be adjacent to the already
allocated set (the *frontier*). Choosing the best set of B nodes is a
constrained combinatorial optimization problem: the objective is non-additive
in practice, high-value "notable" nodes are gated behind chains of low-value
"travel" nodes, and the feasible region is connected-subgraph selection.

Classical solvers answer each query from scratch. This project studies
**amortized combinatorial optimization**: train a graph-based reinforcement
learning policy once, so that producing a build at inference time costs a
single fast forward pass per query rather than a fresh search.

## What This Preliminary Baseline Does

This repository contains only the baseline experiment. It deliberately does
**not** include the Gymnasium environment, the RL algorithms (PPO, DQN), or
the JAX evaluator, which are future work.

`preliminary_results.py`:

1. Generates a synthetic, connected 500-node skill tree (seeded). 350 nodes
   (70%) are low-value travel nodes forming a branching backbone with some
   cycles; 150 nodes (30%) are high-value notables arranged in 30 small
   clusters, each reachable only through a single gate node on the backbone.
2. Enforces the allocation constraint: builds start at root node 0 and every
   subsequent node must be on the frontier.
3. Runs two baseline agents with budget B = 30:
   - **Random Valid Walk**: picks a frontier node uniformly at random.
   - **Greedy**: picks the frontier node with the highest immediate isolated
     value (ties broken by lowest node id).
4. Simulates 100 episodes per agent, recording total objective value (sum of
   allocated node values) and mean wall-clock time per query (one query = one
   complete 30-point build), in milliseconds.
5. Normalizes objectives so the Random Valid Walk mean is exactly 1.00 and
   prints a table with columns `Agent | Norm. objective | Time / query`.

Greedy is deterministic, so its 100 episodes yield the same build; episodes
matter for the random agent and for timing. Greedy's myopia (it cannot "see
through" travel nodes to the notables behind them) is the gap that a learned
policy is expected to close.

## Installation and Usage

Requires Python 3.9+.

1. Clone or download this repository and open a terminal in its folder.
2. (Optional) create a virtual environment:

   ```bash
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the experiment:

   ```bash
   python preliminary_results.py
   ```

Optional flags: `--seed N` (graph and agent seed, default 0), `--episodes N`
(default 100), `--budget N` (default 30).

## Example Output

Timings vary by machine; objectives are reproducible for a given seed.

```
+-------------------+-----------------+--------------+
| Agent             | Norm. objective | Time / query |
+-------------------+-----------------+--------------+
| Random Valid Walk |            1.00 |     0.032 ms |
| Greedy            |            1.86 |     0.041 ms |
+-------------------+-----------------+--------------+
```
