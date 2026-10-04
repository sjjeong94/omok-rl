"""Stage 6 experiments: Gumbel AlphaZero, playout cap randomization and tree reuse, on omok9 and then on 15x15.

Each run writes runs/stage6/<experiment>/<run>.json (learning curve), <run>.pt (final network) and <run>/gen-*.pt,
and is skipped if its .json exists; an interrupted run resumes from <run>/resume.pt. Runs go one at a time: each uses
all CPU cores (search workers) and the GPU (the main process). Summarize with scripts/stage6_eval.py and
scripts/plot_stage6.py.

    uv run python scripts/stage6_alphazero.py [--only omok9] [--run gumbel16-seed0]
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from stage5_alphazero import git_commit

OUT = Path('runs/stage6')
STAGE5_OMOK9 = 'runs/stage5/omok9/main-seed0.pt'  # Stage 5's final omok9 network (0.76 vs heuristic)
OMOK9_RUNS = {  # one change each from Stage 5's main run, 30 generations
    'gumbel16': dict(search='gumbel', simulations=16),
    'gumbel50': dict(search='gumbel', simulations=50),
    'gumbel200': dict(search='gumbel', simulations=200),
    'pcr200-50': dict(fast_simulations=50, full_prob=0.25),  # 200 on a quarter of the moves, 50 on the rest
    'reuse-tree': dict(reuse_tree=True),
}


def experiments():
    from omok_rl.alphazero import AZConfig

    omok9 = AZConfig(env='omok9', generations=30)
    runs = [('omok9', f'{name}-seed0', replace(omok9, **overrides)) for name, overrides in OMOK9_RUNS.items()]
    # cheap searches: as many generations as the 30-generation PUCT run's wall-clock time allows
    runs.append(('omok9', 'gumbel16-long-seed0', replace(omok9, search='gumbel', simulations=16, generations=200)))
    # 15x15 with Gumbel search: from scratch, and starting from Stage 5's omok9 network (as Stage 5's transfer run)
    freestyle = AZConfig(env='freestyle15', search='gumbel', simulations=50, generations=60)
    runs.append(('freestyle15', 'gumbel50-scratch-seed0', freestyle))
    runs.append(('freestyle15', 'gumbel50-transfer-seed0', replace(freestyle, init=STAGE5_OMOK9)))
    # Renju starts from the transferred freestyle network (its extra input plane, forbidden points, starts at zero)
    runs.append(('renju15', 'gumbel50-transfer-seed0',
                 replace(freestyle, env='renju15', init=str(OUT / 'freestyle15' / 'gumbel50-transfer-seed0.pt'))))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', help='run only this experiment (omok9, freestyle15, renju15)')
    parser.add_argument('--run', help='run only runs with this name')
    args = parser.parse_args()

    from omok_rl.alphazero import train

    commit = git_commit()
    for experiment, name, config in experiments():
        if args.only not in (None, experiment) or args.run not in (None, name):
            continue
        out = OUT / experiment / name
        record_path = Path(f'{out}.json')
        if record_path.exists():
            print(f'{experiment}/{name}: exists, skipped', flush=True)
            continue
        print(f'{experiment}/{name}: starting', flush=True)
        log_path = Path(f'{out}.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a') as log:
            result = train(config, out, log=lambda line: print(line, file=log, flush=True))
        record = json.loads(record_path.read_text()) | {'commit': commit}
        record_path.write_text(json.dumps(record))
        last = result['curve'][-1]
        print(f'{experiment}/{name} ({result["seconds"] / 60:.1f} min): score vs heuristic '
              f'{last.get("score_vs_heuristic_search", float("nan")):.3f} (search), '
              f'{last.get("score_vs_heuristic_raw", float("nan")):.3f} (raw policy)', flush=True)


if __name__ == '__main__':
    main()
