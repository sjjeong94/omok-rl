from pathlib import Path

import numpy as np
from omok.env import BoardGame

from omok_rl.agents.base import Agent
from omok_rl.puct import PUCT, choose_move


class AlphaZeroAgent(Agent):
    """Plays the most visited move of a PUCT search guided by an AlphaZero network (no noise, no sampling).

    `simulations=0` plays the network's most likely move without searching (the "raw policy").
    `act` searches one game; `act_batch` searches many games at once, which is how the training loop and
    `omok_rl.alphazero.play_match` use it. `leaves` > 1 batches several leaves of one tree (virtual loss).
    """

    def __init__(self, net=None, size: int | None = None, simulations: int = 200, c_puct: float = 1.5, leaves: int = 1,
                 device='cpu', symmetries: bool = True, seed: int | None = None, name: str = 'alphazero', evaluator=None):
        """Evaluates positions with `net` (an `AlphaZeroNet`) on `device`, or with any `evaluator`
        (e.g. `inference.RemoteEvaluator` in a worker process, which keeps this module free of torch)."""
        self.simulations, self.leaves, self.name = simulations, leaves, name
        self.rng = np.random.default_rng(seed)
        if evaluator is None:
            from omok_rl.nets import NetEvaluator

            evaluator = NetEvaluator(net, device, size, symmetries, seed)
        self.evaluator = evaluator
        self.search = PUCT(self.evaluator, c_puct, seed=seed)

    def act_batch(self, envs: list[BoardGame]) -> list[int]:
        roots = self.search.search(envs, self.simulations, leaves=self.leaves)
        if self.simulations == 0:  # the root's prior is the network's policy
            return [int(r.moves[self.rng.choice(np.flatnonzero(r.prior == r.prior.max()))]) for r in roots]
        return [choose_move(root, self.rng) for root in roots]

    def act(self, env: BoardGame) -> int:
        return self.act_batch([env])[0]

    @classmethod
    def load(cls, path: str | Path, device: str | None = None, **kwargs) -> 'AlphaZeroAgent':
        """Load a checkpoint saved by `omok_rl.alphazero`. Search settings default to the run's evaluation settings."""
        import torch

        from omok_rl.nets import AlphaZeroNet

        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        c = ckpt['config']
        net = AlphaZeroNet(ckpt['in_channels'], c['channels'], c['blocks'])
        net.load_state_dict(ckpt['state_dict'])
        net.to(device).eval()
        size = kwargs.pop('size', ckpt['size'])  # the network plays on any board size
        kwargs = dict(simulations=c['eval_simulations'], c_puct=c['c_puct'], name=Path(path).stem) | kwargs
        return cls(net, size, device=device, **kwargs)
