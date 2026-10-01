from pathlib import Path

import numpy as np
import torch
from omok.env import BoardGame

from omok_rl.agents.base import Agent
from omok_rl.nets import PolicyValueNet


class PolicyAgent(Agent):
    """Plays from a trained policy-value network: the most likely legal move, or a sample with `sample=True`."""

    def __init__(self, net: PolicyValueNet, name: str = 'policy', device: str | torch.device = 'cpu',
                 sample: bool = False, seed: int | None = None):
        self.net, self.name, self.device, self.sample = net, name, torch.device(device), sample
        self.rng = np.random.default_rng(seed)

    @torch.inference_mode()
    def evaluate(self, env: BoardGame) -> tuple[np.ndarray, float]:
        """pi(a | s) over all cells (0 for illegal ones) and V(s), both for the player to move."""
        obs = torch.as_tensor(env.get_observation(), device=self.device)[None]
        logits, value = self.net(obs)
        mask = torch.as_tensor(env.get_legal_mask(), device=self.device)
        probs = torch.softmax(logits[0].float().masked_fill(~mask, -torch.inf), 0)
        return probs.cpu().numpy(), float(value[0])

    def greedy_actions(self, env: BoardGame) -> list[int]:
        probs, _ = self.evaluate(env)
        return [int(a) for a in np.flatnonzero(probs == probs.max())]

    def act(self, env: BoardGame) -> int:
        probs, _ = self.evaluate(env)
        if self.sample:
            return int(self.rng.choice(len(probs), p=probs / probs.sum()))
        return int(np.argmax(probs))

    @classmethod
    def load(cls, path: str | Path, device: str = 'cpu', **kwargs) -> 'PolicyAgent':
        """Load a checkpoint saved by `omok_rl.pg.train`."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        c = ckpt['config']
        net = PolicyValueNet(ckpt['in_channels'], c['channels'], c['blocks'])
        net.load_state_dict(ckpt['state_dict'])
        net.to(device).eval()
        return cls(net, name=Path(path).stem, device=device, **kwargs)
