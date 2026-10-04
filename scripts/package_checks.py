"""Checks before shipping a Stage 6 network in the `omok` package (see docs/stage6-advanced/experiments/2026-10-05-packaging.md).

- last-move: the network with its "last move" input plane set to zero, i.e. playing from the board alone, as `omok.OmokAgent`
  is called (`agent(state, player)`), against pretrained1 (`b.onnx`).
- cross-rule: the Renju network under freestyle rules (with an empty forbidden-points plane), against the freestyle network
  and pretrained1: can one network serve both rules?
- empty-board: the same networks from the empty board instead of random 4-move openings, as in the web UI.

Results go to runs/stage6/eval/package-<name>.json (skipped if it exists).

    uv run python scripts/package_checks.py [--only last-move]
"""

import argparse
import json
import time
from pathlib import Path

from stage5_alphazero import git_commit
from stage5_eval import GAMES, OPENINGS, load_net, summary

OUT = Path('runs/stage6/eval')
RENJU = 'runs/stage6/renju15/gumbel50-transfer-seed0.pt'
FREE = 'runs/stage6/freestyle15/gumbel50-transfer-seed0.pt'


def without_last_move(net):
    """The network with input plane 3 (the last move) always zero."""
    import torch

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x):
            x = x.clone()
            x[:, 3] = 0
            return self.net(x)

    return Wrapped().eval()


def matches(env_name: str, pairings: list[tuple], opening_plies: int = OPENINGS) -> dict:
    """pairings: ((path, simulations, search, last_move), opponent) with opponent an agent name or the same tuple."""
    from omok_rl.alphazero import Workers
    from omok_rl.envs import make_env
    from omok_rl.nets import NetEvaluator

    planes = make_env(env_name).get_observation().shape[0]
    workers = Workers(12)

    def spec(p):
        if isinstance(p, str):
            return p
        path, sims, search, last_move = p
        net_id = f'{path}:{last_move}'
        if net_id not in workers.evaluators:
            net, _ = load_net(path, in_channels=planes)
            workers.evaluators[net_id] = NetEvaluator(net if last_move else without_last_move(net), 'cuda', 15, seed=0)
        return dict(net=net_id, simulations=sims, c_puct=1.5, search=search, name=label(p))

    def label(p):
        if isinstance(p, str):
            return p
        path, sims, search, last_move = p
        return f'{Path(path).parent.name}-{search}{sims}' + ('' if last_move else '-no-last-move')

    start = time.perf_counter()
    results = workers.matches([(spec(a), spec(b), (label(a), label(b))) for a, b in pairings], env_name, GAMES,
                              opening_plies, seed=0)
    workers.close()
    return dict(env=env_name, opening_plies=opening_plies, seconds=time.perf_counter() - start,
                matches=[summary(r) for r in results])


def jobs():
    renju = lambda sims, search='gumbel', last=True: (RENJU, sims, search, last)
    free = lambda sims, search='gumbel', last=True: (FREE, sims, search, last)
    return [
        ('last-move', matches, ('renju15', [
            (renju(0), 'pretrained1'), (renju(0, last=False), 'pretrained1'),
            (renju(200, 'puct'), 'pretrained1'), (renju(200, 'puct', False), 'pretrained1'),
            (renju(200, 'puct', False), renju(200, 'puct'))])),
        ('cross-rule', matches, ('freestyle15', [
            (renju(0), free(0)), (renju(200, 'puct'), free(200, 'puct')),
            (renju(0), 'pretrained1'), (renju(200, 'puct'), 'pretrained1')])),
        ('empty-board', matches, ('renju15', [
            (renju(0), 'pretrained1'), (renju(0, last=False), 'pretrained1'), (renju(200, 'puct'), 'pretrained1')], 0)),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', help='run only this job')
    args = parser.parse_args()
    commit = git_commit()
    for name, fn, fn_args in jobs():
        if args.only not in (None, name):
            continue
        path = OUT / f'package-{name}.json'
        if path.exists():
            print(f'{name}: exists, skipped', flush=True)
            continue
        start = time.perf_counter()
        result = fn(*fn_args) | {'commit': commit, 'wall_seconds': time.perf_counter() - start}
        path.write_text(json.dumps(result))
        print(f'{name}: done in {result["wall_seconds"] / 60:.1f} min', flush=True)
        for m in result['matches']:
            print(f'  {m["a"]} vs {m["b"]}: {m["score"]:.2f} (B {m["win_black"]:.0%} / W {m["win_white"]:.0%}, '
                  f'draws {m["draws"]:.0%}, {m["game_length"]:.1f} moves)', flush=True)


if __name__ == '__main__':
    main()
