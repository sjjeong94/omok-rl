"""Summarize the Stage 2 runs: print Markdown tables and save figures to docs/stage2-dqn/images/.

    uv run python scripts/stage2_dqn.py   # produces runs/stage2/**/*.json and *.pt
    uv run python scripts/plot_stage2.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from omok.env import PLAYER_BLACK

from omok_rl.agents import HeuristicAgent
from omok_rl.agents.dqn import DQNAgent
from omok_rl.arena import play_game
from omok_rl.envs import make_env
from omok_rl.viz import draw_board

RUNS = Path('runs/stage2')
IMAGES = Path('docs/stage2-dqn/images')

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
# Categorical slots in fixed order.
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
VARIANT_COLORS = {'dqn': SLOTS[0], 'double': SLOTS[1], 'dueling': SLOTS[2], 'nstep3': SLOTS[3], 'all': SLOTS[6]}
VARIANT_LABELS = {'dqn': 'DQN', 'double': '+ Double', 'dueling': '+ Dueling', 'nstep3': '+ 3-step',
                  'all': 'Double + Dueling + 3-step'}
ARCH_COLORS = {'dqn': SLOTS[0], 'cnn-noaug': SLOTS[1], 'mlp': SLOTS[2], 'mlp-noaug': SLOTS[3]}
ARCH_LABELS = {'dqn': 'CNN + symmetry aug.', 'cnn-noaug': 'CNN, no aug.', 'mlp': 'MLP + symmetry aug.',
               'mlp-noaug': 'MLP, no aug.'}
SEQUENTIAL = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#2a78d6', '#0d366b'])

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK_2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': '#c3c2b7', 'font.size': 10, 'axes.titlesize': 10.5, 'axes.titlelocation': 'left',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.8, 'lines.linewidth': 2, 'legend.frameon': False,
})


def load(experiment):
    """{variant: [runs]} for an experiment; run files are named <variant>-seed<k>.json."""
    groups = defaultdict(list)
    for p in sorted((RUNS / experiment).glob('*.json')):
        groups[p.stem.rsplit('-seed', 1)[0]].append(json.loads(p.read_text()))
    return groups


def series(runs, metric):
    x = np.array([c['steps'] for c in runs[0]['curve']])
    return x, np.array([[c[metric] for c in r['curve']] for r in runs])


def band(ax, x, ys, color, label, **kwargs):
    ax.plot(x, ys.mean(0), color=color, label=label, **kwargs)
    ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.15, lw=0)


def steps_axis(ax):
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v / 1e6:g}M' if v >= 1e6 else f'{v / 1e3:g}k'))
    ax.set_xlabel('Self-play moves (both players)')


def pct_axis(ax, lo=0, hi=1):
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{100 * v:.0f}%'))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fmt(values, digits=2):
    values = np.asarray(values, float)
    return f'{values.mean():.{digits}f} ± {values.std():.{digits}f}'


def unbeatable(c):
    return c['worst_case_black'] == 0 and c['worst_case_white'] == 0


def first_step(run, metric, threshold):
    """First checkpoint where `metric` reaches `threshold` (None if never)."""
    return next((c['steps'] for c in run['curve'] if c[metric] >= threshold), None)


def fmt_steps(steps):
    steps = [s for s in steps if s is not None]
    return f'{np.median(steps) / 1e3:,.0f}k' if steps else '—'


# ---------------------------------------------------------------- tic-tac-toe


def tictactoe():
    groups = load('tictactoe')
    print('\n### Tic-tac-toe (exact metrics)\n')
    print('| Variant | Final optimal-move rate | Unbeatable at the end | Checkpoints unbeatable | '
          'Best optimal-move rate | Minutes |')
    print('|---|---:|---:|---:|---:|---:|')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for v in ('dqn', 'all'):
        runs = groups[v]
        last = [r['curve'][-1] for r in runs]
        share = [np.mean([unbeatable(c) for c in r['curve']]) for r in runs]
        best = [max(c['optimal_move_rate'] for c in r['curve']) for r in runs]
        print(f'| {VARIANT_LABELS[v]} | {fmt([c["optimal_move_rate"] for c in last], 3)} '
              f'| {sum(unbeatable(c) for c in last)}/{len(runs)} | {fmt(share)} | {fmt(best, 3)} '
              f'| {fmt([r["seconds"] / 60 for r in runs], 1)} |')
        for i, r in enumerate(runs):
            x = [c['steps'] for c in r['curve']]
            kw = dict(color=VARIANT_COLORS[v], lw=1.4, alpha=0.9, label=VARIANT_LABELS[v] if i == 0 else None)
            axes[0].plot(x, [c['optimal_move_rate'] for c in r['curve']], **kw)
            offset = 0.03 * (i + (3 if v == 'all' else 0)) - 0.075  # keep the six step lines apart
            axes[1].plot(x, [unbeatable(c) + offset for c in r['curve']], drawstyle='steps-post', **kw)
    axes[0].axhline(0.989, color=MUTED, ls=':', lw=1.2)
    axes[0].text(310_000, 0.986, 'tabular Q-learning\n(Stage 1, 50k games)', ha='right', va='top', fontsize=8.5,
                 color=INK_2)
    axes[0].set_title('Optimal moves (all 4,520 positions), one line per seed')
    axes[0].set_ylabel('Share of positions')
    pct_axis(axes[0], 0.8, 1.005)
    axes[1].set_title('Unbeatable at this checkpoint? (lines offset per seed)')
    axes[1].set_yticks([0, 1], ['no', 'yes'])
    axes[1].set_ylim(-0.2, 1.2)
    for ax in axes:
        steps_axis(ax)
    axes[0].legend(loc='lower right')
    save(fig, 'tictactoe.png')


# ---------------------------------------------------------------- omok6 ablation


def ablation():
    groups = load('ablation')
    print('\n### omok6 ablation (final checkpoint, mean ± std over 3 seeds)\n')
    print('| Variant | Score vs random | Win as black vs heuristic | Win as white vs heuristic '
          '| Moves to first 70% black win (median) | Best black win | Minutes |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for v in VARIANT_COLORS:
        runs = groups[v]
        last = [r['curve'][-1] for r in runs]
        best = [max(c['win_black_vs_heuristic'] for c in r['curve']) for r in runs]
        print(f'| {VARIANT_LABELS[v]} | {fmt([c["score_vs_random"] for c in last])} '
              f'| {fmt([c["win_black_vs_heuristic"] for c in last])} | {fmt([c["win_white_vs_heuristic"] for c in last])} '
              f'| {fmt_steps([first_step(r, "win_black_vs_heuristic", 0.7) for r in runs])} | {fmt(best)} '
              f'| {fmt([r["seconds"] / 60 for r in runs], 1)} |')
        x, win = series(runs, 'win_black_vs_heuristic')
        band(axes[0], x, win, VARIANT_COLORS[v], VARIANT_LABELS[v])
        _, q = series(runs, 'q_mean')
        axes[1].plot(x, q.mean(0), color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
    axes[0].set_title('omok6: win rate as Black against the heuristic (50 games)')
    axes[0].set_ylabel('Win rate')
    pct_axis(axes[0], -0.02, 1.02)
    axes[1].set_title('Mean Q(s, a) of sampled moves')
    axes[1].set_ylabel('Q-value')
    axes[1].axhline(0, color=MUTED, lw=0.8)
    for ax in axes:
        steps_axis(ax)
    axes[0].legend(loc='lower right', fontsize=9)
    save(fig, 'ablation-omok6.png')


# ---------------------------------------------------------------- omok6 architecture


def architecture():
    groups = load('architecture') | {'dqn': load('ablation')['dqn']}
    print('\n### omok6 architecture and augmentation (final checkpoint)\n')
    print('| Network | Score vs random | Win as black vs heuristic | Moves to first 70% black win (median) | Minutes |')
    print('|---|---:|---:|---:|---:|')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for v in ARCH_COLORS:
        runs = groups[v]
        last = [r['curve'][-1] for r in runs]
        print(f'| {ARCH_LABELS[v]} | {fmt([c["score_vs_random"] for c in last])} '
              f'| {fmt([c["win_black_vs_heuristic"] for c in last])} '
              f'| {fmt_steps([first_step(r, "win_black_vs_heuristic", 0.7) for r in runs])} '
              f'| {fmt([r["seconds"] / 60 for r in runs], 1)} |')
        x, s = series(runs, 'score_vs_random')
        band(axes[0], x, s, ARCH_COLORS[v], ARCH_LABELS[v])
        x, w = series(runs, 'win_black_vs_heuristic')
        band(axes[1], x, w, ARCH_COLORS[v], ARCH_LABELS[v])
    axes[0].set_title('omok6: score against random (100 games)')
    axes[0].set_ylabel('Score')
    pct_axis(axes[0], 0.4, 1.02)
    axes[1].set_title('omok6: win rate as Black against the heuristic')
    axes[1].set_ylabel('Win rate')
    pct_axis(axes[1], -0.02, 1.02)
    for ax in axes:
        steps_axis(ax)
    axes[0].legend(loc='lower right', fontsize=9)
    save(fig, 'architecture-omok6.png')


# ---------------------------------------------------------------- omok9


def omok9():
    groups = load('omok9')
    print('\n### omok9 (final checkpoint)\n')
    print('| Variant | Score vs random | Score vs heuristic | Win as black vs heuristic | Win as white vs heuristic '
          '| Best score vs heuristic | Minutes |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for v in ('dqn', 'all'):
        runs = groups[v]
        last = [r['curve'][-1] for r in runs]
        best = [max(c['score_vs_heuristic'] for c in r['curve']) for r in runs]
        print(f'| {VARIANT_LABELS[v]} | {fmt([c["score_vs_random"] for c in last])} '
              f'| {fmt([c["score_vs_heuristic"] for c in last])} | {fmt([c["win_black_vs_heuristic"] for c in last])} '
              f'| {fmt([c["win_white_vs_heuristic"] for c in last])} | {fmt(best)} '
              f'| {fmt([r["seconds"] / 60 for r in runs], 1)} |')
        for ax, color in zip(axes, ('black', 'white')):
            x, w = series(runs, f'win_{color}_vs_heuristic')
            band(ax, x, w, VARIANT_COLORS[v], VARIANT_LABELS[v])
    axes[0].set_title('omok9: win rate as Black against the heuristic')
    axes[1].set_title('omok9: win rate as White against the heuristic')
    axes[0].set_ylabel('Win rate')
    for ax in axes:
        pct_axis(ax, -0.02, 1.02)
        steps_axis(ax)
    axes[0].legend(loc='upper left', fontsize=9)
    save(fig, 'omok9.png')


# ---------------------------------------------------------------- what the network sees


def q_heatmap():
    """A DQN (black) vs heuristic game on omok9: the final position, and the Q-values at one decision."""
    path = sorted((RUNS / 'omok9').glob('all-seed0.pt'))[0]
    agent = DQNAgent.load(path)
    game = play_game(make_env('omok9'), agent, HeuristicAgent(0))
    moves = game.get_move_history()
    winner = {0: 'Draw', 1: 'Black (DQN) wins', 2: 'White (heuristic) wins'}[game.get_winner()]

    # the DQN's decision a few moves before the end
    k = max(0, len(moves) - 5)
    k -= k % 2  # a black (DQN) move
    env = make_env('omok9')
    for m in moves[:k]:
        env.move(m)
    q = agent.q_values(env)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    draw_board(axes[0], 9, moves)
    axes[0].set_title(f'DQN (black) vs heuristic (white), omok9\n{winner} in {len(moves)} moves', fontsize=10)
    draw_board(axes[1], 9, moves[:k], numbers=True, highlight_last=False)
    legal = np.isfinite(q)
    vmin, vmax = q[legal].min(), q[legal].max()
    for a in np.flatnonzero(legal):
        y, x = divmod(a, 9)
        color = SEQUENTIAL((q[a] - vmin) / (vmax - vmin + 1e-9))
        axes[1].add_patch(plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, color=color, alpha=0.85, zorder=1.5))
        if q[a] >= np.sort(q[legal])[-3]:
            axes[1].text(x, y, f'{q[a]:+.2f}', ha='center', va='center', fontsize=7, color='white', zorder=3)
    best = int(np.argmax(q))
    axes[1].add_patch(plt.Circle(divmod(best, 9)[::-1], 0.5, facecolor='none', edgecolor='#e34948', lw=2, zorder=4))
    axes[1].set_title(f'Q-values before Black\'s move {k + 1} (darker = higher; top 3 labeled,\n'
                      f'red ring = chosen move; range {vmin:+.2f} to {vmax:+.2f})', fontsize=10)
    save(fig, 'q-values-omok9.png')


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    for name, fn in (('tictactoe', tictactoe), ('ablation', ablation), ('architecture', architecture),
                     ('omok9', omok9)):
        if (RUNS / name).exists():
            fn()
    if (RUNS / 'omok9').exists():
        q_heatmap()
    commits = {json.loads(p.read_text()).get('commit') for p in RUNS.glob('*/*.json')}
    print(f'\ncommits: {commits}')


if __name__ == '__main__':
    main()
