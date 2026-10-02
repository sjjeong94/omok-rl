import numpy as np

from omok_rl.agents.alphabeta import AlphaBetaAgent
from omok_rl.agents.base import Agent
from omok_rl.agents.heuristic import HeuristicAgent
from omok_rl.agents.mcts import MCTSAgent
from omok_rl.agents.minimax import MinimaxAgent
from omok_rl.agents.random_agent import RandomAgent
from omok_rl.agents.tabular import AfterstateAgent, QAgent, TabularAgent

AGENTS = ('random', 'heuristic', 'minimax', 'alphabeta[:depth]', 'mcts[:simulations]', 'mcts-heuristic[:simulations]',
          'pretrained0', 'pretrained1', '<saved tabular agent .pkl>', '<saved DQN or policy .pt>')


def make_agent(name: str, seed: int | None = None) -> Agent:
    """Create an agent by name (see `AGENTS`)."""
    if name == 'random':
        return RandomAgent(seed)
    if name == 'heuristic':
        return HeuristicAgent(seed)
    if name == 'minimax':
        return MinimaxAgent(seed)
    base, _, arg = name.partition(':')
    if base == 'alphabeta':
        return AlphaBetaAgent(depth=int(arg or 3), seed=seed)
    if base == 'mcts':  # pure MCTS: UCT with random rollouts
        return MCTSAgent(int(arg or 400), seed=seed)
    if base == 'mcts-heuristic':  # heuristic prior and heuristic rollouts
        return MCTSAgent(int(arg or 400), evaluation='heuristic', prior='heuristic', seed=seed)
    if name.endswith('.pt'):
        import torch

        from omok_rl.agents.dqn import DQNAgent
        from omok_rl.agents.policy import PolicyAgent

        if torch.load(name, map_location='cpu', weights_only=False).get('kind') == 'policy':
            return PolicyAgent.load(name, seed=seed)
        return DQNAgent.load(name)
    if name.endswith('.pkl'):
        agent = TabularAgent.load(name)
        agent.rng = np.random.default_rng(seed)
        return agent
    if name in ('pretrained0', 'pretrained1'):
        from omok_rl.agents.pretrained import PretrainedAgent  # loads onnxruntime

        return PretrainedAgent(model_index=int(name[-1]))
    raise ValueError(f'unknown agent {name!r}, choose from {list(AGENTS)}')


__all__ = ['AGENTS', 'AfterstateAgent', 'Agent', 'AlphaBetaAgent', 'HeuristicAgent', 'MCTSAgent', 'MinimaxAgent', 'QAgent',
           'RandomAgent', 'TabularAgent', 'make_agent']
