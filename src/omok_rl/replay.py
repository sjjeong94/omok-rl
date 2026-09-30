"""Experience replay for two-player self-play, with n-step returns.

Every move is stored from the point of view of the player who made it. Players alternate, so over k plies
the value flips sign k times (negamax). The n-step target of a move a_t in state s_t is

    target = (-1)^k * r              if the game ended k plies later (k < n) with reward r for that ply's mover
    target = (-1)^n * max_a' Q(s_{t+n}, a')   otherwise

For n = 1 this is the Stage 1 Q-learning target: r at the end of the game, else minus the opponent's best value.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    obs: np.ndarray  # (B, C, size, size) float32
    action: np.ndarray  # (B,) int64
    ret: np.ndarray  # (B,) float32, the reward part of the target
    next_obs: np.ndarray  # (B, C, size, size) float32, state n plies later
    next_mask: np.ndarray  # (B, size*size) bool, legal moves there
    sign: np.ndarray  # (B,) float32, (-1)^n, or 0 if the game ended (no bootstrapping)


class ReplayBuffer:
    """Fixed-size ring buffer of transitions; old ones are overwritten first."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], seed: int | None = None):
        n_actions = obs_shape[-1] * obs_shape[-2]
        self.obs = np.zeros((capacity, *obs_shape), np.uint8)  # planes are 0/1, so store them as bytes
        self.next_obs = np.zeros((capacity, *obs_shape), np.uint8)
        self.action = np.zeros(capacity, np.int64)
        self.ret = np.zeros(capacity, np.float32)
        self.next_mask = np.zeros((capacity, n_actions), bool)
        self.sign = np.zeros(capacity, np.float32)
        self.capacity, self.size, self.index = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, ret, next_obs, next_mask, sign):
        i = self.index
        self.obs[i], self.action[i], self.ret[i] = obs, action, ret
        self.next_obs[i], self.next_mask[i], self.sign[i] = next_obs, next_mask, sign
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = self.rng.integers(self.size, size=batch_size)
        return Batch(self.obs[idx].astype(np.float32), self.action[idx], self.ret[idx],
                     self.next_obs[idx].astype(np.float32), self.next_mask[idx], self.sign[idx])

    def __len__(self):
        return self.size


class NStepWriter:
    """Turns the moves of one game into n-step transitions and writes them to the buffer."""

    def __init__(self, buffer: ReplayBuffer, n: int):
        self.buffer, self.n = buffer, n
        self.pending = []  # (obs, action) of the last moves, oldest first

    def step(self, obs, action, next_obs, next_mask, done: bool, reward: float):
        """Record that `action` was played in `obs`, leading to `next_obs` (the opponent's view).

        `reward` is for the player who just moved (1 for a win, 0 otherwise); only used when `done`.
        """
        self.pending.append((obs, action))
        if done:
            for i, (o, a) in enumerate(self.pending):
                k = len(self.pending) - 1 - i  # plies between this move and the final one
                self.buffer.add(o, a, (-1) ** k * reward, next_obs, next_mask, 0.0)
            self.pending = []
        elif len(self.pending) == self.n:
            o, a = self.pending.pop(0)
            self.buffer.add(o, a, 0.0, next_obs, next_mask, float((-1) ** self.n))
