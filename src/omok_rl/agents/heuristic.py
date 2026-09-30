"""A hand-written baseline based on counting stones in every possible winning line.

A "window" is any `win` consecutive cells in a row, column, or diagonal. A window that contains
only our stones (and empty cells) is still winnable for us; one that contains both colors is dead.
Each empty cell is scored by the windows passing through it:

- attack:  for each window with no opponent stones, ATTACK[k] where k is our stone count
           after playing the cell.
- defense: for each window with no stones of ours, DEFENSE[k] where k is the opponent's stone
           count if they played the cell.

The weights grow by 10x per stone, and completing a window dwarfs everything else, which gives
the priority: win now > block the opponent's win > extend our longest line > block theirs.
The agent plays the highest-scoring legal move and breaks ties at random.
"""

from functools import cache

import numpy as np
from omok.env import BoardGame

from omok_rl.agents.base import Agent

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


@cache
def line_windows(size: int, win: int) -> np.ndarray:
    """Flat indices of every `win`-long line on a size x size board, shape (n_windows, win)."""
    windows = []
    for y in range(size):
        for x in range(size):
            for dy, dx in DIRECTIONS:
                end_y, end_x = y + dy * (win - 1), x + dx * (win - 1)
                if 0 <= end_y < size and 0 <= end_x < size:
                    windows.append([(y + dy * i) * size + (x + dx * i) for i in range(win)])
    return np.array(windows)


def line_weights(win: int) -> tuple[np.ndarray, np.ndarray]:
    """ATTACK and DEFENSE weights indexed by stone count (0..win)."""
    attack = np.array([0.0] + [10.0**k for k in range(1, win)] + [1e12])
    defense = np.array([0.0] + [0.5 * 10.0**k for k in range(1, win)] + [1e10])
    return attack, defense


class HeuristicAgent(Agent):
    """Greedy one-ply agent that scores moves by the lines they build and block."""

    name = 'heuristic'

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def scores(self, env: BoardGame) -> np.ndarray:
        """Score of every cell, shape (size*size,); illegal cells are -inf."""
        board = env.get_state().ravel()
        me = env.get_player()
        opponent = 3 - me

        windows = line_windows(env.size, env.win)
        cells = board[windows]
        n_me = (cells == me).sum(axis=1)
        n_opponent = (cells == opponent).sum(axis=1)
        attack, defense = line_weights(env.win)

        # Every cell in a window gets the window's value; occupied cells are masked out below.
        value = np.where(n_opponent == 0, attack[np.minimum(n_me + 1, env.win)], 0.0)
        value += np.where(n_me == 0, defense[np.minimum(n_opponent + 1, env.win)], 0.0)
        scores = np.zeros(board.size)
        np.add.at(scores, windows, value[:, None])

        scores[~env.get_legal_mask()] = -np.inf
        return scores

    def act(self, env: BoardGame) -> int:
        scores = self.scores(env)
        best = np.flatnonzero(scores == scores.max())
        return int(self.rng.choice(best))
