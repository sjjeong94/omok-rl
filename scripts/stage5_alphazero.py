"""Stage 5 experiments: AlphaZero training runs on tic-tac-toe, omok9 and the 15x15 boards.

Each run writes runs/stage5/<experiment>/<run>.json (learning curve), <run>.pt (final network) and <run>/gen-*.pt
(the network at every evaluation), and is skipped if its .json exists; an interrupted run resumes from
<run>/resume.pt. Runs go one at a time: each uses all CPU cores (search workers) and the GPU (the main process).
Summarize with scripts/stage5_eval.py (matches, Elo) and scripts/plot_stage5.py.

    uv run python scripts/stage5_alphazero.py [--only omok9] [--run main-seed0]
"""

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

OUT = Path('runs/stage5')
OMOK9_ABLATIONS = {
    'temp8': dict(temp_moves=8),  # sample twice as many opening moves
    'reuse4': dict(reuse=4.0),  # fewer gradient steps per self-play position
    'sims50': dict(simulations=50),  # a quarter of the search per self-play move
    'no-symmetry': dict(augment=False, symmetries=False),  # no 8x augmentation, in training or in the search
    'gating': dict(gating=True),  # AlphaGo Zero: self-play with the best network so far
}


def experiments():
    from omok_rl.alphazero import AZConfig

    omok9 = AZConfig(env='omok9')
    runs = [('tictactoe', 'main-seed0', AZConfig(env='tictactoe', channels=32, blocks=2, simulations=50, temp_moves=4,
                                                  games_per_gen=240, generations=20, eval_every=2, eval_openings=0,
                                                  eval_simulations=50, buffer_size=20_000, reuse=4))]
    runs.append(('omok9', 'main-seed0', omok9))
    # ablations: one change each, 30 generations (compared with the first 30 of the main run)
    ablation = replace(omok9, generations=30)
    for name, overrides in OMOK9_ABLATIONS.items():
        runs.append(('omok9', f'{name}-seed0', replace(ablation, **overrides)))
    return runs


def git_commit() -> str:
    git = lambda *args: subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()
    return git('rev-parse', '--short', 'HEAD') + ('-dirty' if git('status', '--porcelain') else '')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', help='run only this experiment (tictactoe, omok9, ...)')
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
