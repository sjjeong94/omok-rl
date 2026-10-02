"""Alpha-beta search: an exact solver for small boards, and a depth-limited player for larger ones.

Both are negamax searches: a position's value is from the point of view of the player to move, and the value
of a move is minus the value of the position it leads to. Alpha-beta skips moves that can't change the result:
once a move is found that is at least as good as anything the opponent would allow (`value >= beta`),
the remaining moves are not searched (a **cutoff**).

Both use the same exact **threat rules**, which only look at lines one stone short of a win:

- if the player to move can complete a line, they win now;
- otherwise, if the opponent can complete a line in two or more places, the player to move loses
  (they can block only one);
- otherwise, if the opponent can complete a line in exactly one place, that is the only move worth searching.
"""

from dataclasses import dataclass

import numpy as np
from omok.env import PLAYER_NONE, BoardGame

from omok_rl.agents.base import Agent
from omok_rl.agents.heuristic import HeuristicAgent, line_windows
from omok_rl.symmetry import canonical

EXACT, LOWER, UPPER = 0, 1, 2  # transposition-table entry: the stored value is exact, a lower bound or an upper bound


def winning_cells(env: BoardGame, player: int) -> np.ndarray:
    """Empty cells where `player` would complete `win` in a row (sorted, unique)."""
    board = env.get_state().ravel()
    windows = line_windows(env.size, env.win)
    cells = board[windows]
    open_ = ((cells == player).sum(1) == env.win - 1) & ((cells == PLAYER_NONE).sum(1) == 1)
    w = windows[open_]
    return np.unique(w[board[w] == PLAYER_NONE])


def threat_moves(env: BoardGame) -> tuple[float | None, np.ndarray | None]:
    """Apply the threat rules. Returns (value, None) if they decide the position, (None, [block]) if one move
    is forced, and (None, None) otherwise."""
    me = env.get_player()
    mine = winning_cells(env, me)
    if len(mine):
        return 1.0, None
    theirs = winning_cells(env, 3 - me)
    if len(theirs) >= 2:
        return -1.0, None
    if len(theirs) == 1:
        return None, theirs
    return None, None


@dataclass
class SolverOptions:
    """Which speed-ups the exact solver uses (all of them by default; switched off one at a time in Stage 4)."""

    alphabeta: bool = True  # False: plain negamax, every move is searched
    table: bool = True  # transposition table: remember the value of positions already searched
    symmetry: bool = True  # the table shares entries between the 8 rotations/reflections of a board
    ordering: bool = True  # search the moves the heuristic likes first (more cutoffs)
    threats: bool = True  # the threat rules above


class SearchLimit(Exception):
    """The solver visited more than `max_nodes` positions."""


class Solver:
    """Exact negamax value (+1 win, 0 draw, -1 loss for the player to move) with alpha-beta and a transposition table."""

    def __init__(self, options: SolverOptions | None = None, max_nodes: int | None = None):
        self.options = options or SolverOptions()
        self.max_nodes = max_nodes
        self.table: dict[bytes, tuple[float, int]] = {}
        self.nodes = 0  # positions visited (calls to `negamax`)
        self.heuristic = HeuristicAgent()

    def key(self, env: BoardGame) -> bytes:
        board = env.get_state().ravel()
        if self.options.symmetry:
            return canonical(board, env.size)[0]  # the player to move follows from the stone count
        return board.tobytes()

    def ordered_moves(self, env: BoardGame) -> np.ndarray:
        if not self.options.ordering:
            return np.flatnonzero(env.get_legal_mask())
        scores = self.heuristic.scores(env)
        legal = np.flatnonzero(np.isfinite(scores))
        return legal[np.argsort(-scores[legal], kind='stable')]

    def value(self, env: BoardGame) -> float:
        """Value of the position for the player to move. Leaves `env` unchanged, unless `SearchLimit` is raised."""
        return self.negamax(env, -1.0, 1.0)

    def negamax(self, env: BoardGame, alpha: float, beta: float) -> float:
        self.nodes += 1
        if self.max_nodes is not None and self.nodes > self.max_nodes:
            raise SearchLimit
        opts = self.options
        if not opts.alphabeta:
            alpha, beta = -1.0, 1.0  # a full window: no move can cause a cutoff
        alpha_in = alpha
        key = self.key(env) if opts.table else None
        if key is not None and key in self.table:
            value, flag = self.table[key]
            if flag == EXACT or (flag == LOWER and value >= beta) or (flag == UPPER and value <= alpha):
                return value
        moves = None
        if opts.threats:
            decided, moves = threat_moves(env)
            if decided is not None:
                return decided
        if moves is None:
            moves = self.ordered_moves(env)

        best = -1.0
        for a in moves:
            env.move(int(a))
            if env.is_done():
                value = 0.0 if env.get_winner() == PLAYER_NONE else 1.0
            else:
                value = -self.negamax(env, -beta, -alpha)
            env.move_back()
            best = max(best, value)
            alpha = max(alpha, value)
            if opts.alphabeta and (alpha >= beta or best == 1.0):
                break  # cutoff (a win can't be improved on)
        if key is not None:
            flag = UPPER if best <= alpha_in else LOWER if best >= beta else EXACT
            self.table[key] = (best, flag)
        return best

    def move_values(self, env: BoardGame) -> dict[int, float]:
        """Exact value of every legal move for the player to move."""
        env = env.clone()
        values = {}
        for a in np.flatnonzero(env.get_legal_mask()):
            env.move(int(a))
            values[int(a)] = (0.0 if env.get_winner() == PLAYER_NONE else 1.0) if env.is_done() else -self.value(env)
            env.move_back()
        return values


WIN_SCORE = 1e9


class AlphaBetaAgent(Agent):
    """Depth-limited alpha-beta with a heuristic evaluation, searching only the `width` most promising moves.

    The leaves are scored by `evaluate`: the line counts of `HeuristicAgent`, ours minus the opponent's.
    A win scores `WIN_SCORE` minus the number of plies to reach it, so that faster wins are preferred.
    """

    def __init__(self, depth: int = 2, width: int = 8, seed: int | None = None):
        self.depth, self.width = depth, width
        self.name = f'alphabeta{depth}'
        self.rng = np.random.default_rng(seed)
        self.heuristic = HeuristicAgent()
        self.nodes = 0

    def evaluate(self, env: BoardGame) -> float:
        """Heuristic score of the position for the player to move."""
        board = env.get_state().ravel()
        cells = board[line_windows(env.size, env.win)]
        me, opponent = env.get_player(), 3 - env.get_player()
        n_me, n_opp = (cells == me).sum(1), (cells == opponent).sum(1)
        weight = 10.0 ** np.arange(env.win + 1)
        mine = weight[n_me][(n_opp == 0) & (n_me > 0)].sum()
        theirs = weight[n_opp][(n_me == 0) & (n_opp > 0)].sum()
        return float(mine - theirs)

    def candidates(self, env: BoardGame) -> np.ndarray:
        scores = self.heuristic.scores(env)
        legal = np.flatnonzero(np.isfinite(scores))
        order = legal[np.argsort(-scores[legal] + 1e-9 * self.rng.random(len(legal)), kind='stable')]  # random tie-break
        return order[:self.width]

    def negamax(self, env: BoardGame, depth: int, ply: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        decided, moves = threat_moves(env)
        if decided is not None:
            return decided * (WIN_SCORE - ply - (decided < 0))  # a forced loss happens one ply after the opponent's win
        if depth == 0:
            return self.evaluate(env)
        if moves is None:
            moves = self.candidates(env)
        best = -np.inf
        for a in moves:
            env.move(int(a))
            if env.is_done():
                value = 0.0 if env.get_winner() == PLAYER_NONE else WIN_SCORE - ply
            else:
                value = -self.negamax(env, depth - 1, ply + 1, -beta, -alpha)
            env.move_back()
            best = max(best, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best

    def act(self, env: BoardGame) -> int:
        env = env.clone()
        decided, forced = threat_moves(env)
        if decided == 1.0:
            return int(winning_cells(env, env.get_player())[0])
        moves = forced if forced is not None else self.candidates(env)
        best, best_value, alpha = int(moves[0]), -np.inf, -np.inf
        for a in moves:
            env.move(int(a))
            if env.is_done():
                value = 0.0 if env.get_winner() == PLAYER_NONE else WIN_SCORE
            else:
                value = -self.negamax(env, self.depth - 1, 1, -np.inf, -alpha)
            env.move_back()
            if value > best_value:
                best, best_value = int(a), value
            alpha = max(alpha, value)
        return best
