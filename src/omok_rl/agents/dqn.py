from pathlib import Path

import numpy as np
import torch
from omok.env import BoardGame
from torch import nn

from omok_rl.agents.base import Agent
from omok_rl.nets import make_qnet


class DQNAgent(Agent):
    """Plays the legal move with the highest Q-value of a trained Q-network."""

    def __init__(self, net: nn.Module, name: str = 'dqn', device: str | torch.device = 'cpu'):
        self.net, self.name, self.device = net, name, torch.device(device)

    @torch.inference_mode()
    def q_values(self, env: BoardGame) -> np.ndarray:
        """Q-value of every cell for the player to move; illegal cells are -inf."""
        obs = torch.as_tensor(env.get_observation(), device=self.device)[None]
        q = self.net(obs)[0].float().cpu().numpy()
        q[~env.get_legal_mask()] = -np.inf
        return q

    def greedy_actions(self, env: BoardGame) -> list[int]:
        q = self.q_values(env)
        return [int(a) for a in np.flatnonzero(q == q.max())]

    def act(self, env: BoardGame) -> int:
        return int(np.argmax(self.q_values(env)))

    @classmethod
    def load(cls, path: str | Path, device: str = 'cpu') -> 'DQNAgent':
        """Load a checkpoint saved by `omok_rl.dqn.train`."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        c = ckpt['config']
        net = make_qnet(c['arch'], ckpt['in_channels'], ckpt['size'], c['dueling'], c['channels'], c['blocks'])
        net.load_state_dict(ckpt['state_dict'])
        net.to(device).eval()
        return cls(net, name=Path(path).stem, device=device)
