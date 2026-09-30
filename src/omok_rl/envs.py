"""Game environments used throughout the project, from tic-tac-toe up to 15x15 Renju.

The small boards subclass `omok.env.BoardGame`, which only needs `size` and `win`.
They are freestyle (no forbidden moves) and have 4 observation planes, while
`omok.Omok` has a 5th plane for Renju forbidden points.
"""

from omok import Omok
from omok.env import BoardGame


class TicTacToe(BoardGame):
    """3x3 board, three in a row wins."""

    size, win = 3, 3


class Omok6(BoardGame):
    """6x6 board, four in a row wins."""

    size, win = 6, 4


class Omok9(BoardGame):
    """9x9 board, five in a row wins."""

    size, win = 9, 5


ENVS = {
    'tictactoe': TicTacToe,
    'omok6': Omok6,
    'omok9': Omok9,
    'freestyle15': lambda: Omok(rule='freestyle'),
    'renju15': lambda: Omok(rule='renju'),
}


def make_env(name: str) -> BoardGame:
    """Create a new environment by name (see `ENVS`)."""
    if name not in ENVS:
        raise ValueError(f'unknown env {name!r}, choose from {list(ENVS)}')
    return ENVS[name]()
