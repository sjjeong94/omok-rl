"""Stage 3 experiments: REINFORCE, A2C and PPO self-play on tic-tac-toe and omok9.

Each run writes runs/stage3/<experiment>/<run>.json (learning curve), <run>.pt (final network), and
<run>/steps-*.pt (the network at every evaluation).
Runs share one GPU, several at a time. Summarize with scripts/plot_stage3.py.

    uv run python scripts/stage3_pg.py [--only algorithms]
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
from dataclasses import replace
from pathlib import Path

# batch sizes change every update; without this, cached blocks of old sizes pile up (~5 GB of GPU memory per run)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch  # noqa: E402

from omok_rl.pg import PGConfig, train

SEEDS = range(3)
ALGORITHMS = {
    'reinforce': dict(algo='reinforce', baseline=False),
    'reinforce-baseline': dict(algo='reinforce'),
    'a2c': dict(algo='a2c'),
    'ppo': dict(algo='ppo'),
}
# Controls for PPO's two changes to A2C: the same 16 gradient steps per batch, but each sample used once (a2c-16mb),
# or 4 epochs over the batch without the clipped ratio (noclip).
UPDATES = {
    'a2c-16mb': dict(algo='a2c', minibatches=16),
    'noclip': dict(algo='a2c', epochs=4, minibatches=4),
}
TICTACTOE = PGConfig(env='tictactoe', total_steps=300_000, eval_every=10_000, batch_moves=2048, eval_openings=0)
OMOK9 = PGConfig(env='omok9', total_steps=6_000_000, eval_every=250_000)
ENTROPY = (0.0, 0.03, 0.1)  # ent_coef values besides the default 0.01 (the 'ppo' runs of `algorithms`)


def experiments():
    runs = []
    for name in ('a2c', 'ppo'):
        for seed in SEEDS:
            runs.append(('tictactoe', f'{name}-seed{seed}', replace(TICTACTOE, seed=seed, **ALGORITHMS[name])))
    for ent in ENTROPY:
        for seed in SEEDS:
            runs.append(('tictactoe', f'ent{ent:g}-seed{seed}', replace(TICTACTOE, seed=seed, ent_coef=ent)))
    for name, overrides in ALGORITHMS.items():
        for seed in SEEDS:
            runs.append(('algorithms', f'{name}-seed{seed}', replace(OMOK9, seed=seed, **overrides)))
    for name, overrides in UPDATES.items():
        for seed in SEEDS:
            runs.append(('updates', f'{name}-seed{seed}', replace(OMOK9, seed=seed, **overrides)))
    for ent in ENTROPY:
        for seed in SEEDS:
            runs.append(('entropy', f'ent{ent:g}-seed{seed}', replace(OMOK9, seed=seed, ent_coef=ent)))
    for seed in SEEDS:
        runs.append(('league', f'pool-seed{seed}', replace(OMOK9, seed=seed, pool_prob=0.5)))
    return runs


def git_commit() -> str:
    git = lambda *args: subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()
    return git('rev-parse', '--short', 'HEAD') + ('-dirty' if git('status', '--porcelain') else '')


def run(experiment, name, config, out_dir, commit):
    torch.set_num_threads(2)
    out = out_dir / experiment / name
    record_path = Path(f'{out}.json')  # not with_suffix(): names like 'ent0.03-seed0' contain a dot
    if record_path.exists():
        return f'{experiment}/{name}: exists, skipped'
    result = train(config, out, log=lambda *_: None)
    # record the code version next to the curve
    record = json.loads(record_path.read_text()) | {'commit': commit}
    record_path.write_text(json.dumps(record))
    last = result['curve'][-1]
    return (f'{experiment}/{name} ({result["seconds"] / 60:.1f} min): score vs random {last["score_vs_random"]:.3f}, '
            f'vs heuristic {last["score_vs_heuristic"]:.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=Path('runs/stage3'))
    parser.add_argument('--workers', type=int, default=3)  # each run holds ~2.4 GB of RAM; 5 saturate a 16 GB GPU
    parser.add_argument('--only', help='run only this experiment (tictactoe, algorithms, updates, entropy, league)')
    args = parser.parse_args()

    runs = [r for r in experiments() if args.only in (None, r[0])]
    runs.sort(key=lambda r: -r[2].total_steps)  # start the long runs first
    commit = git_commit()
    with mp.get_context('spawn').Pool(args.workers) as pool:  # CUDA needs spawned workers
        jobs = [pool.apply_async(run, (*r, args.out_dir, commit)) for r in runs]
        for job in jobs:
            print(job.get(), flush=True)


if __name__ == '__main__':
    main()
