"""Play matches between agents and report win rates.

Colors alternate every game (agent A is black in even games), because in Omok the first player
has a large advantage. Results are always reported per color.

    uv run omok-arena heuristic random --env omok9 --games 200
"""

import argparse
import json
from dataclasses import asdict, dataclass, field

import numpy as np
from omok.env import PLAYER_BLACK, PLAYER_NONE, PLAYER_WHITE, BoardGame

from omok_rl.agents import AGENTS, Agent, make_agent
from omok_rl.envs import ENVS, make_env


def play_game(env: BoardGame, black: Agent, white: Agent) -> BoardGame:
    """Play one game to the end on `env` and return it (read the result with `get_winner()`)."""
    while not env.is_done():
        agent = black if env.get_player() == PLAYER_BLACK else white
        pos = agent.act(env)
        if env.move(pos) == -1:
            raise RuntimeError(f'{agent.name} played an illegal move {pos} after {env.get_move_history()}')
    return env


@dataclass
class Game:
    a_color: int  # PLAYER_BLACK or PLAYER_WHITE
    winner: int  # PLAYER_NONE (draw), PLAYER_BLACK, or PLAYER_WHITE
    moves: list[int]


@dataclass
class MatchResult:
    """Games between agent A and agent B. Outcomes are counted from A's point of view."""

    agent_a: str
    agent_b: str
    env: str
    games: list[Game] = field(default_factory=list)

    def count(self, outcome: str, color: int | None = None) -> int:
        """Number of A's wins, draws or losses (`outcome`), optionally only as `color`."""
        n = 0
        for g in self.games:
            if color is not None and g.a_color != color:
                continue
            result = 'draw' if g.winner == PLAYER_NONE else 'win' if g.winner == g.a_color else 'loss'
            n += result == outcome
        return n

    def rate(self, outcome: str, color: int | None = None) -> float:
        total = sum(1 for g in self.games if color is None or g.a_color == color)
        return self.count(outcome, color) / total if total else float('nan')

    def score(self) -> float:
        """A's average score (win = 1, draw = 0.5, loss = 0)."""
        return self.rate('win') + 0.5 * self.rate('draw')

    def summary(self) -> str:
        lengths = [len(g.moves) for g in self.games]
        lines = [
            f'{self.agent_a} vs {self.agent_b} on {self.env} ({len(self.games)} games)',
            f'{"":10}{"win":>8}{"draw":>8}{"loss":>8}',
        ]
        for label, color in (('as black', PLAYER_BLACK), ('as white', PLAYER_WHITE), ('total', None)):
            rates = ''.join(f'{self.rate(o, color):8.1%}' for o in ('win', 'draw', 'loss'))
            lines.append(f'{label:10}{rates}')
        lines.append(f'score: {self.score():.3f}, game length: {np.mean(lengths):.1f} moves (mean)')
        return '\n'.join(lines)


def evaluate(agent_a: Agent, agent_b: Agent, env_name: str, n_games: int) -> MatchResult:
    """Play `n_games` between A and B on fresh `env_name` boards, alternating colors."""
    result = MatchResult(agent_a.name, agent_b.name, env_name)
    for i in range(n_games):
        a_is_black = i % 2 == 0
        black, white = (agent_a, agent_b) if a_is_black else (agent_b, agent_a)
        env = play_game(make_env(env_name), black, white)
        a_color = PLAYER_BLACK if a_is_black else PLAYER_WHITE
        result.games.append(Game(a_color, env.get_winner(), env.get_move_history()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('agent_a', help=f'one of {AGENTS}')
    parser.add_argument('agent_b', help=f'one of {AGENTS}')
    parser.add_argument('--env', default='omok9', choices=list(ENVS))
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', help='save every game as JSON to this path')
    args = parser.parse_args()

    np.random.seed(args.seed)  # the pretrained agent uses the global random state
    agent_a = make_agent(args.agent_a, seed=args.seed)
    agent_b = make_agent(args.agent_b, seed=args.seed + 1)
    result = evaluate(agent_a, agent_b, args.env, args.games)
    print(result.summary())
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(asdict(result), f)


if __name__ == '__main__':
    main()
