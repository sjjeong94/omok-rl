"""Stage 2 experiments: DQN self-play on tic-tac-toe, omok6 and omok9.

Each run writes runs/stage2/<experiment>/<run>.json (learning curve), <run>.pt (final network), and
<run>/steps-*.pt (the network at every evaluation).
Runs share one GPU, several at a time. Summarize with scripts/plot_stage2.py.

    uv run python scripts/stage2_dqn.py [--only ablation]
"""

import argparse
import json
import multiprocessing as mp
import subprocess
from dataclasses import replace
from pathlib import Path

import torch

from omok_rl.dqn import DQNConfig, train

SEEDS = range(3)
VARIANTS = {
    'dqn': {},
    'double': dict(double=True),
    'dueling': dict(dueling=True),
    'nstep3': dict(n_step=3),
    'all': dict(double=True, dueling=True, n_step=3),
}
TICTACTOE = DQNConfig(env='tictactoe', total_steps=300_000, eval_every=10_000, learning_starts=5_000, buffer_size=50_000,
                      eval_openings=0)  # exact metrics instead
OMOK6 = DQNConfig(env='omok6', total_steps=600_000, eval_every=30_000, eval_openings=2)
OMOK9 = DQNConfig(env='omok9', total_steps=2_000_000, eval_every=100_000, buffer_size=500_000, eval_openings=4)


def experiments():
    runs = []
    for variant in ('dqn', 'all'):
        for seed in SEEDS:
            runs.append(('tictactoe', f'{variant}-seed{seed}', replace(TICTACTOE, seed=seed, **VARIANTS[variant])))
    for variant, overrides in VARIANTS.items():
        for seed in SEEDS:
            runs.append(('ablation', f'{variant}-seed{seed}', replace(OMOK6, seed=seed, **overrides)))
    for name, overrides in (('cnn-noaug', dict(augment=False)), ('mlp-noaug', dict(arch='mlp', augment=False)),
                            ('mlp', dict(arch='mlp'))):
        for seed in SEEDS:
            runs.append(('architecture', f'{name}-seed{seed}', replace(OMOK6, seed=seed, **overrides)))
    for variant in ('dqn', 'all'):
        for seed in SEEDS:
            runs.append(('omok9', f'{variant}-seed{seed}', replace(OMOK9, seed=seed, **VARIANTS[variant])))
    return runs


def git_commit() -> str:
    git = lambda *args: subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()
    return git('rev-parse', '--short', 'HEAD') + ('-dirty' if git('status', '--porcelain') else '')


def run(experiment, name, config, out_dir, commit):
    torch.set_num_threads(2)
    out = out_dir / experiment / name
    if out.with_suffix('.json').exists():
        return f'{experiment}/{name}: exists, skipped'
    result = train(config, out, log=lambda *_: None)
    # record the code version next to the curve
    record = json.loads(out.with_suffix('.json').read_text()) | {'commit': commit}
    out.with_suffix('.json').write_text(json.dumps(record))
    last = result['curve'][-1]
    return (f'{experiment}/{name} ({result["seconds"] / 60:.1f} min): score vs random {last["score_vs_random"]:.3f}, '
            f'vs heuristic {last["score_vs_heuristic"]:.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=Path('runs/stage2'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--only', help='run only this experiment (tictactoe, ablation, architecture, omok9)')
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
