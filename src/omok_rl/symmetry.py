"""Canonical board keys: the 8 rotations/reflections of a board share one key.

A tic-tac-toe position and its mirror image have the same value, so a lookup table can store them
once. `canonical` picks the symmetric variant with the smallest byte string as the representative.
"""

from functools import cache

import numpy as np
from omok.transforms import NUM_SYMMETRIES, transform_board


@cache
def permutations(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays (8, size*size) for every symmetry code.

    `board[perms[c]]` is the flat board transformed by code c, and an action `a` on the original board
    is the action `inverse[c][a]` on the transformed board.
    """
    cells = np.arange(size * size).reshape(size, size)
    perms = np.stack([transform_board(cells, code).ravel() for code in range(NUM_SYMMETRIES)])
    inverse = np.argsort(perms, axis=1)
    return perms, inverse


def canonical(board: np.ndarray, size: int) -> tuple[bytes, np.ndarray]:
    """Key of the canonical form of a flat uint8 board, and the action map into that form.

    Returns (key, action_map) where `action_map[a]` is where action `a` lands in the canonical board.
    """
    perms, inverse = permutations(size)
    variants = board[perms]
    keys = [v.tobytes() for v in variants]
    code = min(range(NUM_SYMMETRIES), key=keys.__getitem__)
    return keys[code], inverse[code]
