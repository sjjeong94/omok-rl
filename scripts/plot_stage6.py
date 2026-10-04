"""Summarize the Stage 6 runs: print Markdown tables and save figures to docs/stage6-advanced/images/.

    uv run python scripts/stage6_alphazero.py   # training runs: runs/stage6/<experiment>/<run>.json
    uv run python scripts/stage6_eval.py        # evaluations:   runs/stage6/eval/*.json
    uv run python scripts/plot_stage6.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_stage5 import INK, MUTED, SLOTS, load_json, parse_log, save as save5, series

RUNS = Path('runs/stage6')
STAGE5 = Path('runs/stage5')
IMAGES = Path('docs/stage6-advanced/images')


def save(fig, name):
    import plot_stage5

    plot_stage5.IMAGES = IMAGES
    save5(fig, name)


def load_run(path: Path) -> dict | None:
    """A finished run's record, or the curve of an unfinished one parsed from its log."""
    record = load_json(Path(f'{path}.json'))
    if record is None and Path(f'{path}.log').exists():
        record = {'curve': parse_log(Path(f'{path}.log')), 'unfinished': True}
    return record


def cost(curve: list[dict]) -> float:
    return curve[-1]['seconds'] / 3600 if curve else float('nan')


# ---------------------------------------------------------------- omok9

OMOK9 = {  # label -> (run path, color, line style)
    'PUCT, 200 sims (Stage 5 main)': (STAGE5 / 'omok9' / 'main-seed0', INK, '-'),
    'PUCT, 100 sims (Stage 5)': (STAGE5 / 'omok9' / 'sims100-seed0', MUTED, '--'),
    'PUCT, 50 sims (Stage 5)': (STAGE5 / 'omok9' / 'sims50-seed0', MUTED, ':'),
    'Gumbel, 16 sims': (RUNS / 'omok9' / 'gumbel16-seed0', SLOTS[3], '-'),
    'Gumbel, 50 sims': (RUNS / 'omok9' / 'gumbel50-seed0', SLOTS[1], '-'),
    'Gumbel, 200 sims': (RUNS / 'omok9' / 'gumbel200-seed0', SLOTS[0], '-'),
    'PUCT 200 + playout cap (50 fast)': (RUNS / 'omok9' / 'pcr200-50-seed0', SLOTS[2], '-'),
    'PUCT 200 + tree reuse': (RUNS / 'omok9' / 'reuse-tree-seed0', SLOTS[4], '-'),
    'Gumbel, 16 sims, 200 generations': (RUNS / 'omok9' / 'gumbel16-long-seed0', SLOTS[3], '--'),
}


def omok9(horizon: int = 30):
    runs = {k: (load_run(p), c, ls) for k, (p, c, ls) in OMOK9.items()}
    runs = {k: v for k, v in runs.items() if v[0] and v[0]['curve']}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for label, (run, color, ls) in runs.items():
        long = 'generations' in label
        curve = run['curve'] if long else [p for p in run['curve'] if p['gen'] <= horizon]
        kw = dict(color=color, ls=ls, marker='o', ms=3, lw=2.2 if 'main' in label else 1.6, label=label)
        if not long:
            axes[0].plot(*series(curve, 'score_vs_heuristic_search'), **kw)
        x, y = series(curve, 'score_vs_heuristic_search', 'seconds')
        axes[1].plot(x / 60, y, **kw)
    axes[0].set_xlabel('Generation (480 self-play games each)')
    axes[1].set_xlabel('Wall-clock minutes (self-play + training + evaluation)')
    axes[1].set_xscale('log')
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.set_ylabel('Score vs the heuristic (with search)')
    axes[0].set_title('omok9: score by generation (first 30 generations)')
    axes[1].set_title('The same runs against time (log scale)')
    axes[1].legend(loc='upper left', fontsize=7.5)
    save(fig, 'omok9-search-variants.png')

    print('\n### omok9: search variants (score vs heuristic, 100 games, evaluated with the run\'s own search, 200 sims)\n')
    print('| Run | Gen 10 | Gen 20 | Gen 30 | Raw policy, gen 30 | Sims per move (avg) | Minutes for 30 gens | '
          'Self-play length, gen 30 | Black wins, gen 30 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for label, (run, _, _) in runs.items():
        c = run['curve']
        scores = dict(zip(*series(c, 'score_vs_heuristic_search')))
        raw = dict(zip(*series(c, 'score_vs_heuristic_raw')))
        at = next((p for p in c if p['gen'] == horizon), None)
        cfg = run.get('config', {})
        sims = cfg.get('simulations', float('nan'))
        if cfg.get('fast_simulations'):
            sims = cfg['full_prob'] * sims + (1 - cfg['full_prob']) * cfg['fast_simulations']
        cells = ' | '.join(f'{scores[g]:.2f}' if g in scores else '–' for g in (10, 20, 30))
        minutes = at['seconds'] / 60 if at else float('nan')
        print(f'| {label} | {cells} | {raw.get(horizon, float("nan")):.2f} | {sims:.0f} | {minutes:.1f} | '
              f'{at["game_length"] if at else float("nan"):.1f} | {at["black_wins"] if at else float("nan"):.0%} |')
    long = runs.get('Gumbel, 16 sims, 200 generations')
    if long:
        print('\nGumbel 16, long run:', ', '.join(f'gen {g:.0f}: {s:.2f}' for g, s in
                                                   zip(*series(long[0]['curve'], 'score_vs_heuristic_search')) if g % 25 == 0))


def selfplay_speed():
    """Self-play seconds per generation and positions per second, PUCT vs tree reuse vs playout cap."""
    print('\n### omok9: self-play cost per generation (mean over generations 11-30)\n')
    print('| Run | Self-play seconds per generation | Moves played per second | Training samples per generation |')
    print('|---|---:|---:|---:|')
    for label in ('PUCT, 200 sims (Stage 5 main)', 'PUCT 200 + tree reuse', 'PUCT 200 + playout cap (50 fast)',
                  'Gumbel, 200 sims', 'Gumbel, 50 sims', 'Gumbel, 16 sims'):
        run = load_run(OMOK9[label][0])
        pts = [p for p in run['curve'] if 10 < p['gen'] <= 30] if run else []
        if not pts:
            continue
        sp = np.mean([p['self_play_seconds'] for p in pts])
        moves = np.mean([p['game_length'] * 480 for p in pts])
        samples = np.mean([p.get('full_moves', p['positions'] / p['gen']) for p in pts])
        print(f'| {label} | {sp:.1f} | {moves / sp:,.0f} | {samples:,.0f} |')


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    omok9()
    selfplay_speed()


if __name__ == '__main__':
    main()
