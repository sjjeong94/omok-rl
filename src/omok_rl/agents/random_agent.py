import numpy as np
from omok.env import BoardGame

from omok_rl.agents.base import Agent


class RandomAgent(Agent):
    """Plays a uniformly random legal move."""

    name = 'random'

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, env: BoardGame) -> int:
        return int(self.rng.choice(np.flatnonzero(env.get_legal_mask())))
