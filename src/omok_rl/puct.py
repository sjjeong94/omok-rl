"""Batched PUCT search with a policy-value network (Stage 5): the search inside AlphaZero.

Each simulation walks down the tree from the root, choosing at every position the move that maximizes

    Q(s, a) + c_puct * P(s, a) * sqrt(N(s) + 1) / (1 + N(s, a))

(`Q` = mean value of the move so far, 0 if unvisited; `P` = the network's prior; `N(s, a)` = visits of the move;
`N(s)` = visits of the position), until it reaches a position that is not in the tree yet. Unlike Stage 4's `MCTSAgent`,
the walk stops *before* the network is called: the new position's planes are collected, and the network evaluates the
leaves of many simulations in **one batch** on the GPU. The leaves come from two sources:

- **many games at once**: every game (self-play, or a match) has its own tree, and each contributes a leaf per round;
- **virtual loss**: one tree can contribute several leaves per round. Each walk marks its path as a loss
  (`N += 1`, `W -= 1`) until its leaf is evaluated, so the next walk is pushed onto a different path.

The network's priors and value then expand the leaves (**expansion**) and the value is added to every move on the path
(**backup**), with the sign flipped at every level. A finished game is scored exactly (+1 for the player who just won,
0 for a draw); the network is only asked about positions that are still open.

At the root of a self-play search, Dirichlet noise is mixed into the prior so that every move gets tried sometimes:
`P = (1 - eps) * P + eps * Dir(alpha / legal moves)`.
"""

import numpy as np
from omok.env import PLAYER_NONE, BoardGame


class Node:
    """A position in the tree, with the statistics of its legal moves."""

    __slots__ = ('moves', 'prior', 'n', 'w', 'children', 'visits', 'value')

    def __init__(self, moves: np.ndarray, prior: np.ndarray, value: float):
        self.moves = moves  # legal moves
        self.prior = prior  # P(a) for each move
        self.n = np.zeros(len(moves))  # visits of each move
        self.w = np.zeros(len(moves))  # total value of each move, for the player who makes it
        self.children: list[Node | None] = [None] * len(moves)
        self.visits = 0
        self.value = value  # the network's value of this position, for the player to move

    def policy(self, size: int) -> np.ndarray:
        """Visit counts as a distribution over all `size * size` cells (the training target pi)."""
        pi = np.zeros(size * size, np.float32)
        pi[self.moves] = self.n / self.n.sum()
        return pi


def backup(path: list[tuple[Node, int]], value: float):
    """Add `value` (for the player who made the last move of `path`) to every move on the path."""
    for node, i in reversed(path):
        node.n[i] += 1
        node.w[i] += value
        node.visits += 1
        value = -value


class PUCT:
    """PUCT search over many games at once, with one tree per game.

    The search is written as a generator (`search_steps`) that yields every batch of positions it needs evaluated,
    `(obs, masks)`, and receives `(probs, values)` back. `search` answers with `evaluator`, e.g. `nets.NetEvaluator` on
    this process's device; a worker process can instead interleave several searches while their batches are on the
    GPU (see `inference`). This module doesn't import torch, so worker processes don't either."""

    def __init__(self, evaluator=None, c_puct: float = 1.5, dirichlet_alpha: float = 10.0,
                 dirichlet_eps: float = 0.25, seed: int | None = None):
        self.evaluator, self.c_puct = evaluator, c_puct
        self.dirichlet_alpha, self.dirichlet_eps = dirichlet_alpha, dirichlet_eps
        self.rng = np.random.default_rng(seed)

    def select(self, node: Node) -> int:
        q = np.divide(node.w, node.n, out=np.zeros_like(node.w), where=node.n > 0)
        u = self.c_puct * node.prior * np.sqrt(node.visits + 1) / (1 + node.n)  # +1: the prior decides the first visit
        return int(np.argmax(q + u))

    def search(self, envs: list[BoardGame], simulations: int, noise: bool = False, leaves: int = 1) -> list[Node]:
        """Search every position in `envs` (left unchanged) until its root has `simulations` visits.

        `leaves` is the number of leaves one tree may contribute to each network batch (virtual loss when > 1)."""
        return drive(self.search_steps(envs, simulations, noise, leaves), self.evaluator)

    def search_steps(self, envs: list[BoardGame], simulations: int, noise: bool = False, leaves: int = 1):
        """`search` as a generator: yields (obs, masks), receives (probs, values), returns the roots."""
        masks = np.stack([env.get_legal_mask() for env in envs])
        probs, values = yield np.stack([env.get_observation() for env in envs]), masks
        roots = []
        for mask, p, v in zip(masks, probs, values):
            moves = np.flatnonzero(mask)
            prior = p[moves]
            if noise:  # Dirichlet noise at the root
                eta = self.rng.dirichlet(np.full(len(moves), self.dirichlet_alpha / len(moves)))
                prior = (1 - self.dirichlet_eps) * prior + self.dirichlet_eps * eta
            roots.append(Node(moves, prior, float(v)))

        envs = [env.clone() for env in envs]
        active = [g for g in range(len(envs)) if simulations > 0]
        while active:
            pending, obs, masks = [], [], []
            for g in active:
                env, root = envs[g], roots[g]
                taken = set()  # leaves of this tree already waiting in this batch
                for _ in range(min(leaves, simulations - root.visits)):
                    node, path = root, []
                    while True:
                        i = self.select(node)
                        path.append((node, i))
                        env.move(int(node.moves[i]))
                        if env.is_done() or node.children[i] is None:
                            break
                        node = node.children[i]
                    if env.is_done():  # exact result: a win for the player who just moved, or a draw
                        backup(path, 0.0 if env.get_winner() == PLAYER_NONE else 1.0)
                    elif (id(node), i) in taken:  # virtual loss didn't divert this walk: stop collecting from this tree
                        for _ in path:
                            env.move_back()
                        break
                    else:
                        taken.add((id(node), i))
                        obs.append(env.get_observation())
                        masks.append(env.get_legal_mask())
                        pending.append(path)
                        for n, j in path:  # virtual loss
                            n.n[j] += 1
                            n.w[j] -= 1
                    for _ in path:
                        env.move_back()
            if pending:
                masks = np.stack(masks)
                probs, values = yield np.stack(obs), masks
                for path, mask, p, v in zip(pending, masks, probs, values):
                    for n, j in path:
                        n.n[j] -= 1
                        n.w[j] += 1
                    node, i = path[-1]
                    moves = np.flatnonzero(mask)
                    node.children[i] = Node(moves, p[moves], float(v))
                    backup(path, -float(v))  # v is for the player to move at the leaf, the opponent of the mover
            active = [g for g in active if roots[g].visits < simulations]
        return roots


def drive(steps, evaluator):
    """Run a generator like `PUCT.search_steps` to the end, answering each request with `evaluator`; return its result."""
    try:
        request = next(steps)
        while True:
            request = steps.send(evaluator(*request))
    except StopIteration as stop:
        return stop.value


def choose_move(root: Node, rng: np.random.Generator, sample: bool = False) -> int:
    """The most visited move (ties broken at random), or a move drawn in proportion to the visits."""
    if sample:
        return int(rng.choice(root.moves, p=root.n / root.n.sum()))
    return int(root.moves[rng.choice(np.flatnonzero(root.n == root.n.max()))])
