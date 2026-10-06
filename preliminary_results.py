"""Preliminary baselines for skill-tree point allocation.

Generates a synthetic 500-node skill tree, then compares two non-learned
agents (Random Valid Walk, Greedy) over 100 episodes with a budget of B = 30.
Prints a table of [Agent | Norm. objective | Time / query].

Usage:
    python preliminary_results.py [--seed 0] [--episodes 100] [--budget 30]
"""

import argparse
import random
import time

import networkx as nx

N_NODES = 500
N_CLUSTERS = 30 # gated notable clusters
CLUSTER_SIZE = 5 # nodes per notable cluster
ROOT = 0 # every build starts here


def build_skill_tree(seed: int = 0):
    """Return (graph, value) for a connected 500-node synthetic skill tree.

    Roughly 70% of nodes are low-value "travel" nodes forming a branching,
    mildly cyclic backbone. The other 30% are high-value "notable" nodes
    arranged in small clusters, each reachable only through a single gate
    (one travel node), so notables are gated behind travel nodes.
    """
    rng = random.Random(seed)
    n_notable = N_CLUSTERS * CLUSTER_SIZE
    n_travel = N_NODES - n_notable

    g = nx.Graph()
    g.add_node(ROOT)

    # Travel backbone: random tree biased toward recent nodes -> long paths.
    for v in range(1, n_travel):
        lo = max(0, v - 8)
        g.add_edge(v, rng.randint(lo, v - 1))
    # A few extra edges so the backbone has cycles (like a real tree).
    for _ in range(n_travel // 12):
        a = rng.randrange(n_travel)
        b = min(n_travel - 1, max(0, a + rng.randint(-6, 6)))
        if a != b:
            g.add_edge(a, b)

    # Notable clusters: small connected rings hanging off one gate node.
    depth = nx.single_source_shortest_path_length(g, ROOT)
    gates = [v for v in range(n_travel) if depth[v] >= 4]
    nxt = n_travel
    for _ in range(N_CLUSTERS):
        members = list(range(nxt, nxt + CLUSTER_SIZE))
        nxt += CLUSTER_SIZE
        for i, u in enumerate(members): # ring / path
            g.add_edge(u, members[(i + 1) % CLUSTER_SIZE])
        g.add_edge(rng.choice(gates), members[0]) # the gate

    value = {}
    for v in range(n_travel):
        value[v] = rng.uniform(0.5, 1.5) # low value
    for v in range(n_travel, N_NODES):
        value[v] = rng.uniform(8.0, 15.0) # high value
    value[ROOT] = 0.0 # start node is free

    assert g.number_of_nodes() == N_NODES and nx.is_connected(g)
    return g, value


class FrontierState:
    """Allocated set starting at ROOT; new nodes must be adjacent to it."""

    def __init__(self, g):
        self.g = g
        self.allocated = {ROOT}
        self.frontier = set(g[ROOT])

    def allocate(self, v):
        if v not in self.frontier:
            raise ValueError(f"node {v} is not on the frontier")
        self.allocated.add(v)
        self.frontier.discard(v)
        self.frontier.update(u for u in self.g[v] if u not in self.allocated)


def random_valid_walk(state, value, rng):
    return rng.choice(tuple(state.frontier))


def greedy(state, value, rng):
    # Highest immediate isolated value; ties broken by lowest node id.
    return max(state.frontier, key=lambda v: (value[v], -v))


AGENTS = {"Random Valid Walk": random_valid_walk, "Greedy": greedy}


def run_episode(g, value, policy, budget, rng):
    state = FrontierState(g)
    for _ in range(budget):
        state.allocate(policy(state, value, rng))
    return sum(value[v] for v in state.allocated)


def evaluate(g, value, policy, episodes, budget, seed):
    rng = random.Random(seed)
    scores, times = [], []
    for _ in range(episodes):
        t0 = time.perf_counter()
        scores.append(run_episode(g, value, policy, budget, rng))
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(scores) / len(scores), sum(times) / len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--budget", type=int, default=30)
    args = ap.parse_args()

    g, value = build_skill_tree(args.seed)
    n_notable = N_CLUSTERS * CLUSTER_SIZE
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
          f"({N_NODES - n_notable} travel, {n_notable} notable in "
          f"{N_CLUSTERS} clusters) | root={ROOT} | B={args.budget} | "
          f"episodes={args.episodes} | seed={args.seed}\n")

    results = {name: evaluate(g, value, fn, args.episodes, args.budget, args.seed)
               for name, fn in AGENTS.items()}

    base = results["Random Valid Walk"][0]
    rows = [(name, obj / base, ms) for name, (obj, ms) in results.items()]

    header = ("Agent", "Norm. objective", "Time / query")
    body = [(n, f"{o:.2f}", f"{t:.3f} ms") for n, o, t in rows]
    widths = [max(len(r[i]) for r in [header] + body) for i in range(3)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = lambda r: "| " + " | ".join(
        c.ljust(w) if i == 0 else c.rjust(w)
        for i, (c, w) in enumerate(zip(r, widths))) + " |"
    print(sep, fmt(header), sep, *map(fmt, body), sep, sep="\n")
    print(f"\nRaw mean objective: " + ", ".join(
        f"{n} = {results[n][0]:.2f}" for n in results))


if __name__ == "__main__":
    main()
