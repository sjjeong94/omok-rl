"""Does `omok.AlphaZeroAgent` (the `omok` package's copy of the Stage 6 networks and search) play as well as in omok-rl?

Plays it against `omok.OmokAgent(model_index=1)` (b.onnx) with the protocol of scripts/package_checks.py: 100 games from
50 random 4-move openings, each played once per color, on CPU worker processes. Run with the package checkout first on
the path:

    PYTHONPATH=../omok uv run python scripts/package_parity.py --rule renju --simulations 0 200
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

OUT = Path('runs/stage6/eval')


class PackageAgent:
    """omok-rl's Agent interface around `omok.AlphaZeroAgent`, with a single-threaded ONNX session per process."""

    def __init__(self, rule, simulations, model_path, seed):
        import onnxruntime
        import omok

        self.name = f'omok.AlphaZeroAgent-{rule}-{simulations}'
        self.agent = omok.AlphaZeroAgent(rule=rule, simulations=simulations, model_path=model_path, seed=seed)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.agent.session = onnxruntime.InferenceSession(model_path, options, providers=['CPUExecutionProvider'])

    def act(self, env):
        return self.agent(env.get_state(), env.get_player())


def task(rule, simulations, model_path, openings, first, seed):
    from omok_rl.agents.pretrained import PretrainedAgent
    from omok_rl.selfplay import play_match

    np.random.seed(seed)  # the pretrained agent draws from the global random state
    games = play_match(PackageAgent(rule, simulations, model_path, seed), PretrainedAgent(1), f'{rule}15', openings, first)
    return [dict(a_color=g.a_color, winner=g.winner, moves=g.moves) for g in games]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rule', default='renju')
    parser.add_argument('--simulations', type=int, nargs='+', default=[0, 200])
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()

    import omok

    from omok_rl.selfplay import match_openings

    print('omok package:', omok.__file__)
    model_path = str(Path(omok.__file__).parent.parent / 'models' / f'alphazero-{args.rule}.onnx')
    openings = match_openings(f'{args.rule}15', args.games, 4, seed=0)
    chunks = [c for c in np.array_split(np.arange(args.games), args.workers) if len(c)]
    results = {}
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for sims in args.simulations:
            start = time.perf_counter()
            parts = pool.starmap(task, [(args.rule, sims, model_path, [openings[g] for g in c], int(c[0]), k)
                                        for k, c in enumerate(chunks)])
            games = [g for p in parts for g in p]
            score = np.mean([1.0 if g['winner'] == g['a_color'] else 0.5 if g['winner'] == 0 else 0.0 for g in games])
            black = np.mean([g['winner'] == 1 for g in games if g['a_color'] == 1])
            white = np.mean([g['winner'] == 2 for g in games if g['a_color'] == 2])
            seconds = time.perf_counter() - start
            moves = sum(len(g['moves']) for g in games)
            print(f'{args.rule}, {sims} simulations vs b.onnx: score {score:.2f} (wins as black {black:.0%}, '
                  f'as white {white:.0%}), {seconds / 60:.1f} min, {seconds * len(chunks) / (moves / 2):.2f} s per move',
                  flush=True)
            results[sims] = dict(score=score, win_black=black, win_white=white, seconds=seconds,
                                 seconds_per_move=seconds * len(chunks) / (moves / 2), games=games)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'package-parity-{args.rule}.json').write_text(json.dumps(results))


if __name__ == '__main__':
    main()
