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

import math

import numpy as np
from omok.env import PLAYER_NONE, BoardGame


class Node:
    """A position in the tree, with the statistics of its legal moves."""

    __slots__ = ('moves', 'prior', 'n', 'w', 'children', 'visits', 'value', 'gumbel', 'schedule')

    def __init__(self, moves: np.ndarray, prior: np.ndarray, value: float):
        self.moves = moves  # legal moves
        self.prior = prior  # P(a) for each move
        self.n = np.zeros(len(moves))  # visits of each move
        self.w = np.zeros(len(moves))  # total value of each move, for the player who makes it
        self.children: list[Node | None] = [None] * len(moves)
        self.visits = 0
        self.value = value  # the network's value of this position, for the player to move
        self.gumbel = self.schedule = None  # root of a Gumbel search: the Gumbel noise and the visit schedule

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

    def search(self, envs: list[BoardGame], simulations, noise=False, leaves: int = 1,
               roots: list | None = None) -> list[Node]:
        """Search every position in `envs` (left unchanged) until its root has `simulations` visits.

        `simulations` and `noise` are one value for all positions or a list with one per position. `leaves` is the
        number of leaves one tree may contribute to each network batch (virtual loss when > 1). `roots` may hold the
        subtree of each position kept from the previous move (tree reuse), or None to start from scratch."""
        return drive(self.search_steps(envs, simulations, noise, leaves, roots), self.evaluator)

    def make_root(self, moves: np.ndarray, prior: np.ndarray, value: float, noise: bool, simulations: int) -> Node:
        if noise:  # Dirichlet noise at the root
            eta = self.rng.dirichlet(np.full(len(moves), self.dirichlet_alpha / len(moves)))
            prior = (1 - self.dirichlet_eps) * prior + self.dirichlet_eps * eta
        return Node(moves, prior, value)

    def reuse_root(self, node: Node, noise: bool, simulations: int) -> Node:
        """A subtree kept from the previous move becomes the root: it gets fresh noise, its statistics are kept."""
        if noise:
            node.prior = self.make_root(node.moves, node.prior, node.value, True, simulations).prior
        return node

    def search_steps(self, envs: list[BoardGame], simulations, noise=False, leaves: int = 1, roots: list | None = None):
        """`search` as a generator: yields (obs, masks), receives (probs, values), returns the roots."""
        n_envs = len(envs)
        sims = list(simulations) if np.ndim(simulations) else [simulations] * n_envs
        noise = list(noise) if np.ndim(noise) else [noise] * n_envs
        roots = list(roots) if roots is not None else [None] * n_envs
        fresh = [g for g in range(n_envs) if roots[g] is None]
        for g in range(n_envs):
            if roots[g] is not None:
                roots[g] = self.reuse_root(roots[g], noise[g], sims[g])
        if fresh:
            masks = np.stack([envs[g].get_legal_mask() for g in fresh])
            probs, values = yield np.stack([envs[g].get_observation() for g in fresh]), masks
            for g, mask, p, v in zip(fresh, masks, probs, values):
                moves = np.flatnonzero(mask)
                roots[g] = self.make_root(moves, p[moves], float(v), noise[g], sims[g])

        envs = [env.clone() for env in envs]
        active = [g for g in range(n_envs) if roots[g].visits < sims[g]]
        while active:
            pending, obs, masks = [], [], []
            for g in active:
                env, root = envs[g], roots[g]
                taken = set()  # leaves of this tree already waiting in this batch
                for _ in range(min(leaves, sims[g] - root.visits)):
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
            active = [g for g in active if roots[g].visits < sims[g]]
        return roots

    def target(self, root: Node, size: int) -> np.ndarray:
        """The policy target pi of a search: the visit distribution."""
        return root.policy(size)

    def choose(self, root: Node, rng: np.random.Generator, sample: bool = False) -> int:
        return choose_move(root, rng, sample)


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


# ---------------------------------------------------------------------------- Gumbel AlphaZero


def halving_schedule(considered: int, simulations: int) -> np.ndarray:
    """Sequential Halving as a schedule: entry k is the visit count a root move must have to get simulation k.

    The `considered` moves get visits in rounds; after each phase, only the better half stays considered.
    Each phase gets about `simulations / log2(considered)` simulations (as in `mctx`)."""
    if considered <= 1:
        return np.arange(simulations)
    log2 = math.ceil(math.log2(considered))
    sequence, visits, k = [], [0] * considered, considered
    while len(sequence) < simulations:
        for _ in range(max(1, simulations // (log2 * k))):
            sequence.extend(visits[:k])
            for i in range(k):
                visits[i] += 1
        k = max(2, k // 2)
    return np.array(sequence[:simulations])


class Gumbel(PUCT):
    """Gumbel AlphaZero search (Danihelka et al., 2022): a policy improvement that works with few simulations.

    PUCT's visit counts are only a good policy target when there are many more simulations than moves. Gumbel search
    instead plans which root moves to try:

    - **Root**: draw Gumbel noise g(a) (in self-play; 0 when playing for real) and take the `considered` moves with the
      largest g(a) + logits(a): sampling without replacement from the prior. Sequential Halving then splits the
      simulations among them, dropping the worse half by g + logits + sigma(q) after each phase. The move played is
      the last one left; with the noise, it is a sample from (an approximation of) the improved policy below.
    - **Policy target**: pi' = softmax(logits + sigma(completed q)), where unvisited moves get the value estimate
      `v_mix` (the network's value mixed with the visited moves' q) and `sigma(q) = (c_visit + max N) * c_scale * q`
      on q rescaled to [0, 1]. Every move gets a target, also with 2 simulations.
    - **Below the root**: the move whose share of visits is furthest below pi' (deterministic, no exploration term).
    """

    def __init__(self, evaluator=None, considered: int = 16, c_visit: float = 50.0, c_scale: float = 0.1,
                 seed: int | None = None):
        super().__init__(evaluator, seed=seed)
        self.considered, self.c_visit, self.c_scale = considered, c_visit, c_scale

    @staticmethod
    def logits(node: Node) -> np.ndarray:
        logits = np.log(np.maximum(node.prior, 1e-30))
        return logits - logits.max()

    def sigma_q(self, node: Node) -> np.ndarray:
        """sigma(completed q) of every move, for the player to move at `node`."""
        n = node.n
        visited = n > 0
        q = np.divide(node.w, n, out=np.zeros_like(node.w), where=visited)
        total = n.sum()
        if total > 0:  # v_mix: the network's value and the prior-weighted q of the visited moves
            p = np.maximum(node.prior[visited], 1e-30)
            v_mix = (node.value + total * (p * q[visited]).sum() / p.sum()) / (1 + total)
        else:
            v_mix = node.value
        q = np.where(visited, q, v_mix)
        q = (q - q.min()) / max(q.max() - q.min(), 1e-8)
        return (self.c_visit + n.max()) * self.c_scale * q

    def improved_policy(self, node: Node) -> np.ndarray:
        z = self.logits(node) + self.sigma_q(node)
        z = np.exp(z - z.max())
        return z / z.sum()

    def root_scores(self, root: Node, visit: float) -> np.ndarray:
        """g + logits + sigma(q) of the root moves with `visit` visits (the considered ones); -inf for the others."""
        score = np.maximum(root.gumbel + self.logits(root), -1e9) + self.sigma_q(root)
        return np.where(root.n == visit, score, -np.inf)

    def select(self, node: Node) -> int:
        if node.schedule is not None:  # root: Sequential Halving
            k = int(node.n.sum())
            visit = node.schedule[k] if k < len(node.schedule) else node.n.max()
            return int(np.argmax(self.root_scores(node, visit)))
        return int(np.argmax(self.improved_policy(node) - node.n / (1 + node.n.sum())))

    def make_root(self, moves, prior, value, noise, simulations):
        root = Node(moves, prior, value)
        root.gumbel = self.rng.gumbel(size=len(moves)) if noise else np.zeros(len(moves))
        root.schedule = halving_schedule(min(self.considered, len(moves)), simulations)
        return root

    def reuse_root(self, node, noise, simulations):
        raise NotImplementedError('Sequential Halving starts from a fresh root; Gumbel search does not reuse trees')

    def target(self, root: Node, size: int) -> np.ndarray:
        pi = np.zeros(size * size, np.float32)
        pi[root.moves] = self.improved_policy(root)
        return pi

    def choose(self, root: Node, rng: np.random.Generator, sample: bool = False) -> int:
        """The move left by Sequential Halving (the noise already made it a sample); `sample` is ignored."""
        if root.n.sum() == 0:  # no search: the most likely move after the noise
            return int(root.moves[np.argmax(root.gumbel + self.logits(root))])
        return int(root.moves[np.argmax(self.root_scores(root, root.n.max()))])


def make_search(kind: str = 'puct', evaluator=None, c_puct: float = 1.5, dirichlet_alpha: float = 10.0,
                dirichlet_eps: float = 0.25, considered: int = 16, c_visit: float = 50.0, c_scale: float = 0.1,
                seed: int | None = None) -> PUCT:
    if kind == 'puct':
        return PUCT(evaluator, c_puct, dirichlet_alpha, dirichlet_eps, seed)
    if kind == 'gumbel':
        return Gumbel(evaluator, considered, c_visit, c_scale, seed)
    raise ValueError(f'unknown search {kind!r}')
