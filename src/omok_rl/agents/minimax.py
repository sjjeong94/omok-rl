"""Exact game solving for tic-tac-toe-sized games, used as ground truth in Stage 1.

Values are always from the point of view of the player to move (negamax): +1 win, 0 draw, -1 loss
under perfect play by both sides. The tree is only small enough for tic-tac-toe (5,478 positions);
larger boards will run out of time and memory.
"""

from collections.abc import Callable

import numpy as np
from omok.env import PLAYER_NONE, BoardGame

from omok_rl.agents.base import Agent


def position_key(env: BoardGame) -> bytes:
    return env.get_state().tobytes() + bytes([env.get_player()])


class MinimaxSolver:
    """Negamax with a transposition table (memoized value of every position visited)."""

    def __init__(self):
        self.values: dict[bytes, float] = {}

    def value(self, env: BoardGame) -> float:
        """Value of the position for the player to move. Leaves `env` unchanged."""
        key = position_key(env)
        if key not in self.values:
            self.values[key] = max(self.action_value(env, a) for a in np.flatnonzero(env.get_legal_mask()))
        return self.values[key]

    def action_value(self, env: BoardGame, action: int) -> float:
        """Value of playing `action` for the player to move. Leaves `env` unchanged."""
        env.move(action)
        if env.is_done():
            value = 0.0 if env.get_winner() == PLAYER_NONE else 1.0  # the mover can only win or draw
        else:
            value = -self.value(env)
        env.move_back()
        return value

    def optimal_actions(self, env: BoardGame) -> list[int]:
        actions = np.flatnonzero(env.get_legal_mask())
        values = [self.action_value(env, a) for a in actions]
        best = max(values)
        return [int(a) for a, v in zip(actions, values) if v == best]


class MinimaxAgent(Agent):
    """Perfect player: a uniformly random move among the optimal ones."""

    name = 'minimax'

    def __init__(self, seed: int | None = None, solver: MinimaxSolver | None = None):
        self.rng = np.random.default_rng(seed)
        self.solver = solver or MinimaxSolver()

    def greedy_actions(self, env: BoardGame) -> list[int]:
        return self.solver.optimal_actions(env.clone())

    def act(self, env: BoardGame) -> int:
        return int(self.rng.choice(self.greedy_actions(env)))


def all_positions(env: BoardGame) -> list[BoardGame]:
    """Every distinct non-terminal position reachable from `env` (4,520 for tic-tac-toe)."""
    seen, positions, stack = set(), [], [env.clone()]
    while stack:
        node = stack.pop()
        key = position_key(node)
        if key in seen:
            continue
        seen.add(key)
        positions.append(node)
        for a in np.flatnonzero(node.get_legal_mask()):
            child = node.clone()
            child.move(a)
            if not child.is_done():
                stack.append(child)
    return positions


# A policy's candidate moves: every action it might play in a position (its tied greedy moves).
Candidates = Callable[[BoardGame], list[int]]


def optimal_move_rate(candidates: Candidates, positions: list[BoardGame], optimal: list[set[int]]) -> float:
    """Share of positions where every move the policy might play is optimal.

    `optimal[i]` is the set of optimal moves in `positions[i]` (from `MinimaxSolver.optimal_actions`)."""
    ok = sum(set(candidates(p)) <= best for p, best in zip(positions, optimal))
    return ok / len(positions)


def worst_case(candidates: Candidates, env: BoardGame, color: int) -> float:
    """Result for a policy playing `color` against a perfect adversary: +1 win, 0 draw, -1 loss.

    The adversary tries every move, and the policy is assumed to pick the worst of its tied
    candidates, so 0 means "can never lose" no matter how ties are broken.
    """
    env = env.clone()
    memo = {}

    def search():
        key = position_key(env)
        if key in memo:
            return memo[key]
        mine = env.get_player() == color
        results = []
        for a in candidates(env) if mine else np.flatnonzero(env.get_legal_mask()):
            env.move(a)
            if env.is_done():
                winner = env.get_winner()
                results.append(0.0 if winner == PLAYER_NONE else 1.0 if winner == color else -1.0)
            else:
                results.append(search())
            env.move_back()
        memo[key] = min(results)  # both the adversary and our tie-breaking are worst case
        return memo[key]

    return search()
