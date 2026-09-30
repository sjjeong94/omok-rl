"""Stage 0 experiments: baseline matches between the random, heuristic and pretrained agents.

Each match is saved to runs/stage0/<agent_a>-vs-<agent_b>-<env>.json.
Summarize them with scripts/plot_stage0.py.

    uv run python scripts/stage0_baselines.py
"""

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from omok_rl.agents import make_agent
from omok_rl.arena import evaluate

SMALL = ['tictactoe', 'omok6', 'omok9']
LARGE = ['freestyle15', 'renju15']

# (agent_a, agent_b, env, games)
MATCHES = (
    [('random', 'random', env, 2000) for env in SMALL + LARGE]
    + [('heuristic', 'random', env, 200) for env in SMALL + LARGE]
    + [('heuristic', 'heuristic', env, 200) for env in SMALL + LARGE]
    + [('heuristic', model, env, 200) for model in ('pretrained0', 'pretrained1') for env in LARGE]
    + [('pretrained0', 'pretrained1', env, 200) for env in LARGE]
)


def git_commit() -> str:
    """Current commit hash, with '-dirty' if there are uncommitted (or untracked) changes."""
    git = lambda *args: subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()
    return git('rev-parse', '--short', 'HEAD') + ('-dirty' if git('status', '--porcelain') else '')


def run(match, seed, out_dir):
    agent_a, agent_b, env, games = match
    np.random.seed(seed)  # the pretrained agent uses the global random state
    a, b = make_agent(agent_a, seed=seed), make_agent(agent_b, seed=seed + 1)
    start = time.perf_counter()
    result = evaluate(a, b, env, games)
    seconds = time.perf_counter() - start

    record = asdict(result) | {'seed': seed, 'seconds': seconds, 'commit': git_commit()}
    path = out_dir / f'{agent_a}-vs-{agent_b}-{env}.json'
    path.write_text(json.dumps(record))
    return f'{path.name}: {games} games in {seconds:.1f}s, score {result.score():.3f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, default=Path('runs/stage0'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_agent('pretrained0'), make_agent('pretrained1')  # download the models once, before forking
    with Pool(args.workers) as pool:
        jobs = [pool.apply_async(run, (m, args.seed, args.out_dir)) for m in MATCHES]
        for job in jobs:
            print(job.get())


if __name__ == '__main__':
    main()
