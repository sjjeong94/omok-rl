"""Tabular reinforcement learning by self-play (Stage 1).

Both agents keep a lookup table and learn by playing against themselves with an epsilon-greedy policy.
Values are zero-sum and always from the point of view of the player who acts (negamax), so one table
serves both colors: a position that is worth +v to the player to move is worth -v to the other.

AfterstateAgent learns V(b), the value of the board b right after our move ("afterstate"):
    'mc'  Monte Carlo: after the game, move V(b) toward the final result G (+1 win, 0 draw, -1 loss).
    'td'  TD(0): after the opponent replies with afterstate b', move V(b) toward -V(b').

QAgent learns Q(s, a), the value of playing a in state s:
    'sarsa'       on-policy:  move Q(s, a) toward -Q(s', a'), where a' is the move the opponent actually plays.
    'q-learning'  off-policy: move Q(s, a) toward -max_a' Q(s', a'), the opponent's best reply.

A move that ends the game moves its value toward the reward r instead (+1 win, 0 draw).
"""

import pickle
from pathlib import Path

import numpy as np
from omok.env import PLAYER_NONE, BoardGame

from omok_rl.agents.base import Agent
from omok_rl.envs import make_env
from omok_rl.symmetry import canonical


class TabularAgent(Agent):
    methods: tuple[str, ...] = ()

    def __init__(self, method: str, alpha: float = 0.1, symmetry: bool = True, seed: int | None = None):
        if method not in self.methods:
            raise ValueError(f'method must be one of {self.methods}, got {method!r}')
        self.name = method
        self.method = method
        self.alpha = alpha
        self.symmetry = symmetry
        self.rng = np.random.default_rng(seed)
        self.table: dict = {}

    def key(self, board: np.ndarray, size: int) -> tuple[bytes, np.ndarray | None]:
        """Table key of a flat board, plus the action map into the key's frame (None without symmetry)."""
        if self.symmetry:
            return canonical(board, size)
        return board.tobytes(), None

    def action_values(self, env: BoardGame) -> tuple[np.ndarray, np.ndarray]:
        """(legal actions, their current value estimates) for the player to move."""
        raise NotImplementedError

    def greedy_actions(self, env: BoardGame) -> list[int]:
        actions, values = self.action_values(env)
        return [int(a) for a in actions[values == values.max()]]

    def act(self, env: BoardGame) -> int:
        return int(self.rng.choice(self.greedy_actions(env)))

    def select(self, values: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy index into `values`: random with probability epsilon, else a greedy one (random tie-break)."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(len(values)))
        return int(self.rng.choice(np.flatnonzero(values == values.max())))

    def train_game(self, env_name: str, epsilon: float) -> int:
        """Play one self-play game, learning along the way. Returns the number of moves."""
        raise NotImplementedError

    def save(self, path: str | Path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> 'TabularAgent':
        with open(path, 'rb') as f:
            return pickle.load(f)


def terminal_reward(env: BoardGame) -> float:
    """Reward for the player who just ended the game: 1 for a win, 0 for a draw."""
    return 0.0 if env.get_winner() == PLAYER_NONE else 1.0


class AfterstateAgent(TabularAgent):
    """V(b): value of afterstate b for the player who just moved into it."""

    methods = ('mc', 'td')

    def afterstate_key(self, board: np.ndarray, action: int, player: int, size: int) -> bytes:
        after = board.copy()
        after[action] = player
        return self.key(after, size)[0]

    def afterstates(self, env: BoardGame) -> tuple[np.ndarray, list[bytes], np.ndarray]:
        """Legal actions, the keys of their afterstates, and the afterstate values."""
        board = env.get_state().ravel()
        player = env.get_player()
        actions = np.flatnonzero(env.get_legal_mask())
        keys = [self.afterstate_key(board, a, player, env.size) for a in actions]
        return actions, keys, np.array([self.table.get(k, 0.0) for k in keys])

    def action_values(self, env):
        actions, _, values = self.afterstates(env)
        return actions, values

    def knows(self, env: BoardGame) -> bool:
        """Whether any afterstate of the current state was visited during training."""
        return any(k in self.table for k in self.afterstates(env)[1])

    def update(self, key: bytes, target: float):
        v = self.table.get(key, 0.0)
        self.table[key] = v + self.alpha * (target - v)

    def train_game(self, env_name, epsilon):
        env = make_env(env_name)
        afterstates = []  # (player who moved, afterstate key)
        while not env.is_done():
            player = env.get_player()
            actions, keys, values = self.afterstates(env)
            i = self.select(values, epsilon)
            key = keys[i]
            env.move(actions[i])
            if self.method == 'td':
                if afterstates:  # the opponent's previous afterstate is worth minus ours
                    self.update(afterstates[-1][1], -self.table.get(key, 0.0))
                if env.is_done():
                    self.update(key, terminal_reward(env))
            afterstates.append((player, key))

        if self.method == 'mc':
            winner = env.get_winner()
            for player, key in afterstates:
                ret = 0.0 if winner == PLAYER_NONE else 1.0 if winner == player else -1.0
                self.update(key, ret)
        return len(afterstates)


class QAgent(TabularAgent):
    """Q(s, a): value of playing a in state s for the player to move."""

    methods = ('sarsa', 'q-learning')

    def q_row(self, env: BoardGame, create: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """The table row of the current state (zeros if unseen, stored only if `create`)
        and the map from actions to its indices."""
        board = env.get_state().ravel()  # the player to move follows from the stone count
        key, action_map = self.key(board, env.size)
        if key in self.table:
            row = self.table[key]
        else:
            row = np.zeros(board.size)
            if create:
                self.table[key] = row
        if action_map is None:
            action_map = np.arange(board.size)
        return row, action_map

    def knows(self, env: BoardGame) -> bool:
        """Whether the current state was visited during training."""
        return self.key(env.get_state().ravel(), env.size)[0] in self.table

    def action_values(self, env):
        row, action_map = self.q_row(env)
        actions = np.flatnonzero(env.get_legal_mask())
        return actions, row[action_map[actions]]

    def train_game(self, env_name, epsilon):
        env = make_env(env_name)
        previous = None  # (row, index) of the opponent's last move
        moves = 0
        while not env.is_done():
            row, action_map = self.q_row(env, create=True)
            actions = np.flatnonzero(env.get_legal_mask())
            values = row[action_map[actions]]
            i = self.select(values, epsilon)
            action = actions[i]
            if previous is not None:
                target = -values.max() if self.method == 'q-learning' else -values[i]
                prev_row, prev_index = previous
                prev_row[prev_index] += self.alpha * (target - prev_row[prev_index])
            env.move(action)
            moves += 1
            index = action_map[action]
            if env.is_done():
                row[index] += self.alpha * (terminal_reward(env) - row[index])
            previous = (row, index)
        return moves


def make_tabular(method: str, **kwargs) -> TabularAgent:
    cls = AfterstateAgent if method in AfterstateAgent.methods else QAgent
    return cls(method, **kwargs)
