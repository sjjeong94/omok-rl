from omok_rl.agents.base import Agent
from omok_rl.agents.heuristic import HeuristicAgent
from omok_rl.agents.random_agent import RandomAgent

AGENTS = ('random', 'heuristic', 'pretrained0', 'pretrained1')


def make_agent(name: str, seed: int | None = None) -> Agent:
    """Create an agent by name (see `AGENTS`)."""
    if name == 'random':
        return RandomAgent(seed)
    if name == 'heuristic':
        return HeuristicAgent(seed)
    if name in ('pretrained0', 'pretrained1'):
        from omok_rl.agents.pretrained import PretrainedAgent  # loads onnxruntime

        return PretrainedAgent(model_index=int(name[-1]))
    raise ValueError(f'unknown agent {name!r}, choose from {list(AGENTS)}')


__all__ = ['AGENTS', 'Agent', 'HeuristicAgent', 'RandomAgent', 'make_agent']
