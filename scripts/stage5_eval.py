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
from stage5_alphazero import OMOK9_ABLATIONS, git_commit

OUT = Path('runs/stage5/eval')
RUNS = Path('runs/stage5')


def load_net(path, device='cuda', in_channels: int | None = None):
    """A saved network; with `in_channels`, for a board with more input planes (the extra planes get zero weights)."""
    import torch

    from omok_rl.nets import AlphaZeroNet

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    c = ckpt['config']
    net = AlphaZeroNet(in_channels or ckpt['in_channels'], c['channels'], c['blocks'])
    net.load_transfer(ckpt['state_dict'])
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


# ---------------------------------------------------------------- matches on the workers

PPO = 'runs/stage3/entropy/ent0.03-seed0.pt'  # Stage 3's best policy network (0.42 vs heuristic)
DQN = 'runs/stage2/omok9/all-seed0.pt'  # Stage 2's best omok9 Q-network (0.575 vs heuristic)
GAMES, OPENINGS = 100, 4  # as in Stages 2-4: 50 random 4-move openings, each played once per color


class Players:
    """AlphaZero checkpoints registered with the inference server, by name; other players are agent names."""

    def __init__(self, workers, env_name: str | None = None):
        from omok_rl.envs import make_env

        env = make_env(env_name) if env_name else None
        self.size = env.size if env else None  # networks play on this board, whatever size they were trained on
        self.in_channels = env.get_observation().shape[0] if env else None
        self.workers, self.c_puct = workers, {}

    def az(self, path: str, simulations: int, name: str | None = None) -> dict:
        """A player spec for `Workers.matches`."""
        from omok_rl.nets import NetEvaluator

        net_id = str(path)
        if net_id not in self.c_puct:
            net, ckpt = load_net(path, in_channels=self.in_channels)
            self.workers.evaluators[net_id] = NetEvaluator(net, 'cuda', self.size or ckpt['size'], seed=0)
            self.c_puct[net_id] = ckpt['config']['c_puct']
        return dict(net=net_id, simulations=simulations, c_puct=self.c_puct[net_id],
                    name=name or f'{Path(path).stem}-{simulations}')


def summary(r) -> dict:
    from omok.env import PLAYER_BLACK, PLAYER_WHITE

    return dict(a=r.agent_a, b=r.agent_b, score=r.score(), win_black=r.rate('win', PLAYER_BLACK),
                win_white=r.rate('win', PLAYER_WHITE), draws=r.rate('draw'), games=len(r.games),
                game_length=float(np.mean([len(g.moves) for g in r.games])),
                records=[dict(a_color=g.a_color, winner=g.winner, moves=g.moves) for g in r.games])


def bradley_terry(names: list[str], results: list[tuple[int, int, float, int]], anchor: str, iters: int = 2000):
    """Elo ratings from (i, j, score of i, games) results: P(i beats j) = 1 / (1 + 10 ** ((R_j - R_i) / 400)).

    A virtual draw is added to every pairing, so that a 100% result gives a large but finite gap."""
    k = len(names)
    wins = np.zeros((k, k))
    games = np.zeros((k, k))
    for i, j, score, n in results:
        wins[i, j] += score * n + 0.5
        wins[j, i] += (1 - score) * n + 0.5
        games[i, j] += n + 1
        games[j, i] += n + 1
    r = np.zeros(k)
    for _ in range(iters):  # minorization-maximization (Hunter 2004) on gamma = exp(r)
        gamma = np.exp(r)
        denom = (games / (gamma[:, None] + gamma[None, :])).sum(1)
        r = np.log(wins.sum(1) / denom)
        r -= r[names.index(anchor)]
    return r * 400 / np.log(10)


def elo(run: str = 'omok9/main-seed0', simulations: int = 200, games: int = 40, every: int = 5) -> dict:
    """Round-robin between every `every`-th saved generation of a run and the heuristic."""
    from omok_rl.alphazero import Workers

    paths = sorted((RUNS / run).glob('gen-*.pt'))
    paths = [p for p in paths if int(p.stem[4:]) % every == 0]
    env_name = run.split('/')[0]
    workers = Workers(12)
    players = Players(workers)
    entries = [players.az(p, simulations, name=p.stem) for p in paths] + ['heuristic']
    names = [p.stem for p in paths] + ['heuristic']
    pairs = [(i, j) for i in range(len(entries)) for j in range(i + 1, len(entries))]
    results = workers.matches([(entries[i], entries[j], (names[i], names[j])) for i, j in pairs], env_name, games,
                              OPENINGS, seed=0, games_per_task=games // 2)
    workers.close()
    table = [(i, j, r.score(), len(r.games)) for (i, j), r in zip(pairs, results)]
    ratings = bradley_terry(names, table, 'heuristic')
    rng = np.random.default_rng(0)  # bootstrap: resample the games of every pairing
    boot = []
    for _ in range(200):
        sample = []
        for (i, j), r in zip(pairs, results):
            s = np.array([1.0 if g.winner == g.a_color else 0.5 if g.winner == 0 else 0.0 for g in r.games])
            sample.append((i, j, float(rng.choice(s, len(s)).mean()), len(s)))
        boot.append(bradley_terry(names, sample, 'heuristic', iters=500))
    boot = np.array(boot)
    return dict(run=run, simulations=simulations, games_per_pair=games, names=names, elo=ratings.tolist(),
                elo_low=np.percentile(boot, 2.5, axis=0).tolist(), elo_high=np.percentile(boot, 97.5, axis=0).tolist(),
                matches=[dict(a=names[i], b=names[j], score=r.score(), games=len(r.games)) for (i, j), r in zip(pairs, results)])


def versus(env_name: str, pairings: list[tuple]) -> dict:
    """Matches (a, b) where a and b are (checkpoint path, simulations) or agent names; 100 games each.
    Networks play on `env_name`'s board, also if they were trained on another size."""
    from omok_rl.alphazero import Workers

    workers = Workers(12)
    players = Players(workers, env_name)
    spec = lambda p: players.az(*p) if isinstance(p, tuple) else p
    label = lambda p: f'{Path(p[0]).parent.name}/{Path(p[0]).stem}-{p[1]}' if isinstance(p, tuple) else Path(p).stem
    start = time.perf_counter()
    results = workers.matches([(spec(a), spec(b), (label(a), label(b))) for a, b in pairings], env_name, GAMES,
                              OPENINGS, seed=0)
    workers.close()
    return dict(env=env_name, seconds=time.perf_counter() - start, matches=[summary(r) for r in results])


# ---------------------------------------------------------------- jobs

MAIN9 = 'runs/stage5/omok9/main-seed0.pt'


def jobs():
    out = [('tictactoe', tictactoe, ())]
    out.append(('omok9-elo', elo, ()))
    out.append(('omok9-previous-stages', versus, ('omok9', [
        ((MAIN9, 200), 'mcts-heuristic:3200'), ((MAIN9, 200), 'alphabeta:5'), ((MAIN9, 200), DQN), ((MAIN9, 200), PPO),
        ((MAIN9, 0), 'heuristic'), ((MAIN9, 0), DQN), ((MAIN9, 0), PPO)])))
    out.append(('omok9-simulations', versus, ('omok9', [((MAIN9, n), 'alphabeta:5')
                                                        for n in (0, 25, 50, 100, 200, 400, 800, 1600)])))
    ablations = [f'{a}-seed0' for a in OMOK9_ABLATIONS]
    main30 = 'runs/stage5/omok9/main-seed0/gen-0030.pt'  # the ablations ran 30 generations
    out.append(('omok9-ablations', versus, ('omok9', [((f'runs/stage5/omok9/{a}.pt', 200), (main30, 200))
                                                      for a in ablations])))
    # 15x15: the omok9 network before any 15x15 training, then the trained networks against the pretrained models
    out.append(('freestyle15-zero-shot', versus, ('freestyle15', [((MAIN9, 0), 'heuristic'), ((MAIN9, 200), 'heuristic')])))
    free = {name: f'runs/stage5/freestyle15/{name}-seed0.pt' for name in ('scratch', 'transfer')}
    out.append(('freestyle15-final', versus, ('freestyle15', [
        ((free['transfer'], 200), (free['scratch'], 200)),
        *[((path, sims), opponent) for path in free.values() for sims in (0, 200)
          for opponent in ('heuristic', 'pretrained0', 'pretrained1')]])))
    renju = 'runs/stage5/renju15/transfer-seed0.pt'
    out.append(('renju15-final', versus, ('renju15', [((renju, sims), opponent) for sims in (0, 200, 800)
                                                      for opponent in ('heuristic', 'pretrained0', 'pretrained1')])))
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
