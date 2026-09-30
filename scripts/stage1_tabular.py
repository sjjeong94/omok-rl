"""Stage 1 experiments: tabular self-play learning on tic-tac-toe (and a scaling test on omok6).

Every run trains one agent and evaluates it at log-spaced checkpoints. Results go to
runs/stage1/<experiment>/<run>.json and the final agents to runs/stage1/agents/*.pkl.
Summarize them with scripts/plot_stage1.py.

    uv run python scripts/stage1_tabular.py
"""

import argparse
import json
import subprocess
import time
from functools import cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from omok.env import PLAYER_BLACK, PLAYER_WHITE

from omok_rl.agents import HeuristicAgent, RandomAgent
from omok_rl.agents.base import Agent
from omok_rl.agents.minimax import MinimaxSolver, all_positions, optimal_move_rate, worst_case
from omok_rl.agents.tabular import TabularAgent, make_tabular
from omok_rl.arena import evaluate
from omok_rl.envs import make_env

METHODS = ('mc', 'td', 'sarsa', 'q-learning')
SEEDS = range(5)
DEFAULT = dict(env='tictactoe', alpha=0.1, epsilon=0.1, symmetry=True, games=50_000)


def experiments():
    runs = []
    # 1. The four methods with default settings.
    for method in METHODS:
        for seed in SEEDS:
            runs.append(('methods', DEFAULT | dict(method=method, seed=seed)))
    # 2. Symmetry off (symmetry on comes from experiment 1).
    for method in ('td', 'q-learning'):
        for seed in SEEDS:
            runs.append(('symmetry', DEFAULT | dict(method=method, seed=seed, symmetry=False)))
    # 3. Exploration rate, on-policy vs off-policy (epsilon 0.1 comes from experiment 1).
    for method in ('sarsa', 'q-learning'):
        for epsilon in (0.0, 0.3, 1.0):
            for seed in SEEDS:
                runs.append(('epsilon', DEFAULT | dict(method=method, seed=seed, epsilon=epsilon)))
    # 4. Does it scale? Q-learning on 6x6 four-in-a-row.
    for seed in range(3):
        runs.append(('scaling', DEFAULT | dict(env='omok6', method='q-learning', seed=seed, games=100_000)))
    return runs


def checkpoints(games: int) -> list[int]:
    return sorted({int(round(x, -1)) for x in np.geomspace(100, games, 20)})


@cache
def ground_truth():
    """All tic-tac-toe positions and their optimal moves (computed once per worker process)."""
    solver = MinimaxSolver()
    positions = all_positions(make_env('tictactoe'))
    return positions, [set(solver.optimal_actions(p)) for p in positions]


class Recorder(Agent):
    """Plays like `agent` and counts how many of its decisions were in states seen during training."""

    def __init__(self, agent: TabularAgent):
        self.agent, self.name = agent, agent.name
        self.known = self.total = 0

    def act(self, env):
        self.known += self.agent.knows(env)
        self.total += 1
        return self.agent.act(env)


def evaluate_tictactoe(agent: TabularAgent, seed: int) -> dict:
    positions, optimal = ground_truth()
    vs_random = evaluate(agent, RandomAgent(seed + 1000), 'tictactoe', 200)
    return {
        'optimal_move_rate': optimal_move_rate(agent.greedy_actions, positions, optimal),
        'worst_case_black': worst_case(agent.greedy_actions, make_env('tictactoe'), PLAYER_BLACK),
        'worst_case_white': worst_case(agent.greedy_actions, make_env('tictactoe'), PLAYER_WHITE),
        'score_vs_random': vs_random.score(),
        'loss_vs_random': vs_random.rate('loss'),
    }


def evaluate_scaling(agent: TabularAgent, env: str, seed: int) -> dict:
    player = Recorder(agent)
    vs_random = evaluate(player, RandomAgent(seed + 1000), env, 100)
    vs_heuristic = evaluate(player, HeuristicAgent(seed + 1000), env, 100)
    return {
        'score_vs_random': vs_random.score(),
        'score_vs_heuristic': vs_heuristic.score(),
        'known_state_rate': player.known / player.total,
    }


def git_commit() -> str:
    git = lambda *args: subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()
    return git('rev-parse', '--short', 'HEAD') + ('-dirty' if git('status', '--porcelain') else '')


def run(experiment: str, config: dict, out_dir: Path) -> str:
    agent = make_tabular(config['method'], alpha=config['alpha'], symmetry=config['symmetry'], seed=config['seed'])
    np.random.seed(config['seed'])
    curve, played, train_seconds, moves = [], 0, 0.0, 0
    for checkpoint in checkpoints(config['games']):
        start = time.perf_counter()
        while played < checkpoint:
            moves += agent.train_game(config['env'], config['epsilon'])
            played += 1
        train_seconds += time.perf_counter() - start
        if config['env'] == 'tictactoe':
            metrics = evaluate_tictactoe(agent, config['seed'])
        else:
            metrics = evaluate_scaling(agent, config['env'], config['seed'])
        curve.append({'games': played, 'moves': moves, 'table_size': len(agent.table)} | metrics)

    name = (f'{config["env"]}-{config["method"]}-eps{config["epsilon"]}'
            f'-{"sym" if config["symmetry"] else "nosym"}-seed{config["seed"]}')
    record = {'experiment': experiment, 'config': config, 'curve': curve,
              'train_seconds': train_seconds, 'commit': git_commit()}
    (out_dir / experiment).mkdir(parents=True, exist_ok=True)
    (out_dir / experiment / f'{name}.json').write_text(json.dumps(record))
    if config['env'] == 'tictactoe':  # omok6 tables are too large to be worth keeping
        (out_dir / 'agents').mkdir(exist_ok=True)
        agent.save(out_dir / 'agents' / f'{name}.pkl')
    last = curve[-1]
    summary = ', '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in last.items())
    return f'{experiment}/{name} ({train_seconds:.0f}s): {summary}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=Path('runs/stage1'))
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--only', help='run only this experiment (methods, symmetry, epsilon, scaling)')
    args = parser.parse_args()

    runs = [(e, c) for e, c in experiments() if args.only in (None, e)]
    runs.sort(key=lambda r: -r[1]['games'])  # start the long runs first
    with Pool(args.workers) as pool:
        jobs = [pool.apply_async(run, (e, c, args.out_dir)) for e, c in runs]
        for job in jobs:
            print(job.get(), flush=True)


if __name__ == '__main__':
    main()
