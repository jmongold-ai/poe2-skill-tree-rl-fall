"""Render the synthetic skill tree with each baseline's B-point build highlighted.

Usage:
    python plot_builds.py [--seed 0] [--budget 30] [--out builds.pdf]
"""

import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from preliminary_results import (AGENTS, N_CLUSTERS, N_NODES, CLUSTER_SIZE,
                                 ROOT, FrontierState, build_skill_tree)


def build(g, value, policy, budget, seed):
    rng = random.Random(seed)
    state = FrontierState(g)
    for _ in range(budget):
        state.allocate(policy(state, value, rng))
    return state.allocated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--out", default="builds.pdf")
    args = ap.parse_args()

    g, value = build_skill_tree(args.seed)
    pos = nx.spring_layout(g, seed=args.seed, iterations=200)
    n_travel = N_NODES - N_CLUSTERS * CLUSTER_SIZE
    travel = [v for v in g if v < n_travel]
    notable = [v for v in g if v >= n_travel]

    builds = {name: build(g, value, fn, args.budget, args.seed)
              for name, fn in AGENTS.items()}
    # Zoom to the region the builds occupy, padded so nearby notables show.
    used = [pos[v] for a in builds.values() for v in a]
    xs, ys = zip(*used)
    pad = 0.12 * max(max(xs) - min(xs), max(ys) - min(ys))
    xlim = (min(xs) - pad, max(xs) + pad)
    ylim = (min(ys) - pad, max(ys) + pad)

    fig, axes = plt.subplots(1, len(AGENTS), figsize=(11, 5.2))
    for ax, (name, alloc) in zip(axes, builds.items()):
        score = sum(value[v] for v in alloc)
        picked = alloc - {ROOT}

        nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#cccccc", width=0.5)
        nx.draw_networkx_nodes(g, pos, nodelist=travel, ax=ax, node_size=22,
                               node_color="#b0b0b0")
        nx.draw_networkx_nodes(g, pos, nodelist=notable, ax=ax, node_size=60,
                               node_color="#f4b942", edgecolors="#c07f00",
                               linewidths=0.5)
        sub = g.subgraph(alloc)
        nx.draw_networkx_edges(sub, pos, ax=ax, edge_color="#1f5fbf", width=2.0)
        nx.draw_networkx_nodes(g, pos, nodelist=sorted(picked), ax=ax,
                               node_size=70, node_color="#1f5fbf")
        nx.draw_networkx_nodes(g, pos, nodelist=[ROOT], ax=ax, node_size=260,
                               node_color="#d62728", node_shape="*")
        n_not = len(picked & set(notable))
        ax.set_title(f"{name}\nobjective = {score:.1f}, "
                     f"{n_not} notables reached", fontsize=10)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_axis_off()

    handles = [
        plt.Line2D([], [], marker="*", ls="", color="#d62728", ms=11,
                   label="root"),
        plt.Line2D([], [], marker="o", ls="", color="#1f5fbf", ms=6,
                   label="allocated"),
        plt.Line2D([], [], marker="o", ls="", color="#b0b0b0", ms=5,
                   label="travel node"),
        plt.Line2D([], [], marker="o", ls="", color="#f4b942", ms=7,
                   label="notable node"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
