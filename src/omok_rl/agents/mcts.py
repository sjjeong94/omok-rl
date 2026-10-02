"""Monte Carlo Tree Search (MCTS).

Each **simulation** walks down the tree from the current position, choosing moves with a bandit rule, until it
reaches a position not in the tree yet. That position is added to the tree (**expansion**) and scored
(**evaluation**), and the score is added to the statistics of every move on the path (**backup**).
After all simulations, the agent plays the move that was visited most.

Selection rules (`Q` = mean value of a move so far, `N` = visits of the move, `N_parent` = visits of the position):

- **UCT** (no prior): `Q + c * sqrt(ln N_parent / N)`, every move tried once first (in random order).
- **PUCT** (with a prior `P`, as in AlphaZero): `Q + c * P * sqrt(N_parent) / (1 + N)`, with `Q = 0` for unvisited moves.
  The prior comes from a policy network, or from the heuristic's move scores: `P ∝ (score + 1) ** (1 / temperature)`.

Evaluation of a new position, from the point of view of the player to move there (+1 win, 0 draw, -1 loss):

- `random`: play uniformly random moves to the end of the game (a **rollout**);
- `heuristic`: a rollout where each move is the heuristic's choice, or a random move with probability `epsilon`;
- `value`: the value head of a policy-value network, no rollout.

Values are stored per move from the point of view of the player who makes it, so the sign flips at every level.
"""

import time

import numpy as np
from omok.env import PLAYER_NONE, BoardGame

from omok_rl.agents.base import Agent
from omok_rl.agents.heuristic import HeuristicAgent

EVALUATIONS = ('random', 'heuristic', 'value')
PRIORS = (None, 'heuristic', 'net')


class Node:
    """A position in the tree, with the statistics of its legal moves."""

    __slots__ = ('moves', 'prior', 'n', 'w', 'children', 'visits')

    def __init__(self, moves: np.ndarray, prior: np.ndarray | None):
        self.moves = moves  # legal moves
        self.prior = prior  # P(a) for each move (PUCT), or None (UCT)
        self.n = np.zeros(len(moves))  # visits of each move
        self.w = np.zeros(len(moves))  # total value of each move, for the player who makes it
        self.children: list[Node | None] = [None] * len(moves)
        self.visits = 0

    def q(self) -> np.ndarray:
        return np.divide(self.w, self.n, out=np.zeros_like(self.w), where=self.n > 0)


class MCTSAgent(Agent):
    """MCTS player: UCT without a prior, PUCT with one (`prior='heuristic'` or `'net'`).

    `net` is a `PolicyAgent`, needed for `prior='net'` and `evaluation='value'`."""

    def __init__(self, simulations: int = 400, evaluation: str = 'random', prior: str | None = None, net=None,
                 c: float | None = None, epsilon: float = 0.1, temperature: float = 2.0, seed: int | None = None,
                 name: str | None = None):
        if evaluation not in EVALUATIONS:
            raise ValueError(f'unknown evaluation {evaluation!r}, choose from {EVALUATIONS}')
        if prior not in PRIORS:
            raise ValueError(f'unknown prior {prior!r}, choose from {PRIORS}')
        if net is None and (evaluation == 'value' or prior == 'net'):
            raise ValueError("evaluation='value' and prior='net' need a network")
        self.simulations, self.evaluation, self.prior, self.net = simulations, evaluation, prior, net
        self.epsilon, self.temperature = epsilon, temperature
        self.c = c if c is not None else (1.0 if prior is None else 1.5)
        self.rng = np.random.default_rng(seed)
        self.heuristic = HeuristicAgent(seed)
        self.name = name or f'mcts-{evaluation}-{prior or "uct"}-{simulations}'
        self.root: Node | None = None  # tree of the last search, for analysis
        self.seconds = 0.0  # total search time
        self.searches = 0

    # ------------------------------------------------------------------ search

    def act(self, env: BoardGame) -> int:
        start = time.perf_counter()
        env = env.clone()
        root, _ = self.expand(env)
        for _ in range(self.simulations):
            self.simulate(root, env)
        self.root = root
        self.seconds += time.perf_counter() - start
        self.searches += 1
        best = np.flatnonzero(root.n == root.n.max())
        return int(root.moves[self.rng.choice(best)])

    def simulate(self, root: Node, env: BoardGame):
        """One simulation from `root`, whose position is `env` (restored afterwards)."""
        node, path = root, []
        while True:
            i = self.select(node)
            path.append((node, i))
            env.move(int(node.moves[i]))
            if env.is_done():
                value = 0.0 if env.get_winner() == PLAYER_NONE else 1.0  # the player who just moved won, or a draw
                break
            if node.children[i] is None:
                node.children[i], leaf_value = self.expand(env)
                value = -leaf_value  # the leaf is scored for the player to move there, i.e. the opponent
                break
            node = node.children[i]
        for node, i in reversed(path):  # backup: `value` is for the player who chose move i at `node`
            node.n[i] += 1
            node.w[i] += value
            node.visits += 1
            value = -value
        for _ in path:
            env.move_back()

    def select(self, node: Node) -> int:
        if node.prior is None:  # UCT
            unvisited = np.flatnonzero(node.n == 0)
            if len(unvisited):
                return int(unvisited[0])  # moves were shuffled at expansion, so this is a random untried move
            return int(np.argmax(node.w / node.n + self.c * np.sqrt(np.log(node.visits) / node.n)))
        u = self.c * node.prior * np.sqrt(node.visits + 1) / (1 + node.n)  # +1: the prior decides the first visit
        return int(np.argmax(node.q() + u))

    def expand(self, env: BoardGame) -> tuple[Node, float]:
        """A new node for the position of `env`, and the position's value for the player to move."""
        moves = np.flatnonzero(env.get_legal_mask())
        prior = value = None
        if self.net is not None:
            probs, value = self.net.evaluate(env)
            if self.prior == 'net':
                prior = probs[moves]
        if self.prior == 'heuristic':
            prior = (self.heuristic.scores(env)[moves] + 1) ** (1 / self.temperature)
        if prior is None:
            self.rng.shuffle(moves)  # UCT tries untried moves in this order
        else:
            prior = prior / prior.sum() if prior.sum() > 0 else np.full(len(moves), 1 / len(moves))
        if self.evaluation != 'value':
            value = self.rollout(env)
        return Node(moves, prior), value

    # ------------------------------------------------------------------ evaluation

    def rollout(self, env: BoardGame) -> float:
        """Result of a rollout from `env` for the player to move (+1 win, 0 draw, -1 loss). Leaves `env` unchanged."""
        player = env.get_player()
        sim = env.clone()
        if self.evaluation == 'random':
            empty = np.flatnonzero(sim.get_legal_mask())
            self.rng.shuffle(empty)  # playing the empty cells in a random order = uniformly random moves
            for a in empty:
                sim.move(int(a))
                if sim.is_done():
                    break
        else:
            while not sim.is_done():
                if self.rng.random() < self.epsilon:
                    sim.move(int(self.rng.choice(np.flatnonzero(sim.get_legal_mask()))))
                else:
                    sim.move(self.heuristic.act(sim))
        winner = sim.get_winner()
        return 0.0 if winner == PLAYER_NONE else 1.0 if winner == player else -1.0

    # ------------------------------------------------------------------ analysis

    def root_policy(self) -> tuple[np.ndarray, np.ndarray]:
        """Moves at the root of the last search and their share of the visits."""
        return self.root.moves, self.root.n / self.root.n.sum()
