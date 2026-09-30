from omok import OmokAgent
from omok.env import BoardGame

from omok_rl.agents.base import Agent


class PretrainedAgent(Agent):
    """The pretrained policy network shipped with the `omok` package (15x15 only).

    The model is downloaded to `./omok_assets/` on first use. It plays the network's most likely
    legal move after a random board symmetry, so it is not fully deterministic.
    It draws from numpy's global random state; seed it with `np.random.seed`.
    """

    def __init__(self, model_index: int = 1):
        self.name = f'pretrained{model_index}'
        self.model = OmokAgent(model_index=model_index)

    def act(self, env: BoardGame) -> int:
        if env.size != 15:
            raise ValueError(f'{self.name} only plays on 15x15 boards, got {env.size}x{env.size}')
        self.model.rule = getattr(env, 'rule', 'freestyle')
        return int(self.model(env.get_state(), env.get_player()))
