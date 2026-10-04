"""Stage 6 evaluations (run after scripts/stage6_alphazero.py).

- omok9-vs-main30: every 30-generation network of this stage against Stage 5's main run at generation 30, all with
  the same search (PUCT, 200 simulations), and the Gumbel networks also with their own search.
- omok9-search-at-play: one network (Stage 5's final omok9 network), searched with PUCT or Gumbel at 16 to 200
  simulations, against depth-5 alpha-beta: is Gumbel search also better for *playing* with few simulations?
- freestyle15-final / renju15-final: the 15x15 networks against the heuristic and the pretrained models.

Results go to runs/stage6/eval/<name>.json (skipped if it exists). Matches are played by the CPU search workers,
with the networks evaluated on the GPU by this process.

    uv run python scripts/stage6_eval.py [--only omok9-vs-main30]
"""

import argparse
import json
import time
from pathlib import Path

from stage5_alphazero import git_commit
from stage5_eval import GAMES, OPENINGS, load_net, summary

OUT = Path('runs/stage6/eval')
RUNS = Path('runs/stage6')
MAIN9 = 'runs/stage5/omok9/main-seed0.pt'
MAIN9_30 = 'runs/stage5/omok9/main-seed0/gen-0030.pt'


class Players:
    """AlphaZero checkpoints registered with the inference server; a player is (path, simulations, search)."""

    def __init__(self, workers, env_name: str):
        from omok_rl.envs import make_env

        env = make_env(env_name)
        self.size, self.in_channels = env.size, env.get_observation().shape[0]
        self.workers, self.loaded = workers, set()

    def spec(self, p):
        if isinstance(p, str):
            return p
        path, sims, search = p
        from omok_rl.nets import NetEvaluator

        if path not in self.loaded:
            net, _ = load_net(path, in_channels=self.in_channels)
            self.workers.evaluators[path] = NetEvaluator(net, 'cuda', self.size, seed=0)
            self.loaded.add(path)
        return dict(net=path, simulations=sims, c_puct=1.5, search=search, name=label(p))


def label(p) -> str:
    if isinstance(p, str):
        return Path(p).stem
    path, sims, search = p
    return f'{Path(path).parent.name}/{Path(path).stem}-{search}{sims}'


def versus(env_name: str, pairings: list[tuple]) -> dict:
    from omok_rl.alphazero import Workers

    workers = Workers(12)
    players = Players(workers, env_name)
    start = time.perf_counter()
    results = workers.matches([(players.spec(a), players.spec(b), (label(a), label(b))) for a, b in pairings],
                              env_name, GAMES, OPENINGS, seed=0)
    workers.close()
    return dict(env=env_name, seconds=time.perf_counter() - start, matches=[summary(r) for r in results])


def jobs():
    nine = lambda name: str(RUNS / 'omok9' / f'{name}-seed0.pt')
    main30 = (MAIN9_30, 200, 'puct')
    runs9 = ['gumbel16', 'gumbel50', 'gumbel200', 'pcr200-50', 'reuse-tree']
    pairs = [((nine(r), 200, 'puct'), main30) for r in runs9]
    pairs += [((nine(r), 200, 'gumbel'), main30) for r in runs9 if r.startswith('gumbel')]
    pairs += [((str(RUNS / 'omok9' / 'gumbel16-long-seed0' / f'gen-{g:04d}.pt'), 200, s), main30)
              for g in (100, 200) for s in ('puct', 'gumbel')]
    out = [('omok9-vs-main30', versus, ('omok9', pairs))]
    out.append(('omok9-search-at-play', versus, ('omok9', [((MAIN9, n, s), 'alphabeta:5') for s in ('puct', 'gumbel')
                                                          for n in (4, 16, 50, 200)])))
    free = {name: str(RUNS / 'freestyle15' / f'gumbel50-{name}-seed0.pt') for name in ('scratch', 'transfer')}
    out.append(('freestyle15-final', versus, ('freestyle15', [
        ((free['transfer'], 200, 'gumbel'), (free['scratch'], 200, 'gumbel')),
        *[((path, sims, 'gumbel'), opponent) for path in free.values() for sims in (0, 200)
          for opponent in ('heuristic', 'pretrained0', 'pretrained1')],
        *[((path, 200, 'puct'), opponent) for path in free.values() for opponent in ('heuristic', 'pretrained1')]])))
    renju = str(RUNS / 'renju15' / 'gumbel50-transfer-seed0.pt')
    out.append(('renju15-final', versus, ('renju15', [
        ((renju, 200, 'gumbel'), (free['transfer'], 200, 'gumbel')),  # does Renju training help under Renju?
        *[((renju, sims, 'gumbel'), opponent) for sims in (0, 200) for opponent in ('heuristic', 'pretrained0', 'pretrained1')],
        ((renju, 200, 'puct'), 'heuristic')])))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', help='run only this job')
    args = parser.parse_args()
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
