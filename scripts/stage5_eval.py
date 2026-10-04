"""Stage 5 evaluations of the trained AlphaZero networks (run after scripts/stage5_alphazero.py).

- tictactoe: exact metrics for every saved generation: share of optimal moves over all 4,520 positions and the result
  against a perfect opponent, for the raw policy and with search.
- elo: a round-robin between saved generations of a run (and fixed anchors), rated with the Bradley-Terry model.
- versus: final networks against the strongest players of Stages 2-4 and against each other.

Results go to runs/stage5/eval/<name>.json (skipped if it exists). Matches are played by the CPU search workers,
with the networks evaluated on the GPU by this process.

    uv run python scripts/stage5_eval.py [--only elo]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

OUT = Path('runs/stage5/eval')
RUNS = Path('runs/stage5')


def load_net(path, device='cuda'):
    import torch

    from omok_rl.nets import AlphaZeroNet

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    c = ckpt['config']
    net = AlphaZeroNet(ckpt['in_channels'], c['channels'], c['blocks'])
    net.load_state_dict(ckpt['state_dict'])
    return net.to(device).eval(), ckpt


# ---------------------------------------------------------------- tic-tac-toe: exact metrics


def tictactoe(run: str = 'main-seed0', simulations: int = 50) -> dict:
    from omok.env import PLAYER_BLACK, PLAYER_WHITE

    from omok_rl.agents.minimax import MinimaxSolver, all_positions, optimal_move_rate, position_key, worst_case
    from omok_rl.envs import make_env
    from omok_rl.nets import NetEvaluator
    from omok_rl.puct import PUCT

    positions = all_positions(make_env('tictactoe'))
    solver = MinimaxSolver()
    optimal = [set(solver.optimal_actions(p.clone())) for p in positions]
    keys = [position_key(p) for p in positions]
    points = []
    for path in sorted((RUNS / 'tictactoe' / run).glob('gen-*.pt')):
        net, ckpt = load_net(path)
        evaluator = NetEvaluator(net, 'cuda', 3, symmetries=False)  # deterministic: the network as it is
        roots = PUCT(evaluator, ckpt['config']['c_puct'], seed=0).search(positions, simulations)
        point = {'gen': ckpt['gen']}
        for kind, scores in (('raw', [r.prior for r in roots]), ('search', [r.n for r in roots])):
            table = {k: [int(m) for m in r.moves[s == s.max()]] for k, r, s in zip(keys, roots, scores)}
            candidates = lambda env: table[position_key(env)]
            point[f'optimal_move_rate_{kind}'] = optimal_move_rate(candidates, positions, optimal)
            point[f'worst_case_black_{kind}'] = worst_case(candidates, make_env('tictactoe'), PLAYER_BLACK)
            point[f'worst_case_white_{kind}'] = worst_case(candidates, make_env('tictactoe'), PLAYER_WHITE)
        points.append(point)
    return dict(run=run, simulations=simulations, positions=len(positions), curve=points)


# ---------------------------------------------------------------- jobs


def jobs():
    return [('tictactoe', tictactoe, ())]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', help='run only this job')
    args = parser.parse_args()
    from stage5_alphazero import git_commit

    commit = git_commit()
    for name, fn, fn_args in jobs():
        if args.only not in (None, name):
            continue
        path = OUT / f'{name}.json'
        if path.exists():
            print(f'{name}: exists, skipped', flush=True)
            continue
        start = time.perf_counter()
        result = fn(*fn_args) | {'commit': commit, 'wall_seconds': time.perf_counter() - start}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
        print(f'{name}: done in {result["wall_seconds"] / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
