"""Summarize the Stage 1 runs: print Markdown tables and save figures to docs/stage1-tabular/images/.

    uv run python scripts/stage1_tabular.py   # produces runs/stage1/**/*.json
    uv run python scripts/plot_stage1.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from omok_rl.agents.tabular import TabularAgent
from omok_rl.envs import make_env

RUNS = Path('runs/stage1')
IMAGES = Path('docs/stage1-tabular/images')
METHODS = ('mc', 'td', 'sarsa', 'q-learning')
LABELS = {'mc': 'Monte Carlo', 'td': 'TD(0)', 'sarsa': 'Sarsa', 'q-learning': 'Q-learning'}

# Reference palette (light mode): one fixed categorical slot per method.
SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
COLORS = {'mc': '#2a78d6', 'td': '#eb6834', 'sarsa': '#1baf7a', 'q-learning': '#4a3aa7'}
DIVERGING = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK_2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': '#c3c2b7', 'font.size': 10, 'axes.titlesize': 10.5, 'axes.titlelocation': 'left',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.8, 'lines.linewidth': 2, 'legend.frameon': False,
})


def load(experiment):
    return [json.loads(p.read_text()) for p in sorted((RUNS / experiment).glob('*.json'))]


def group(runs, *keys):
    """Runs grouped by the given config keys."""
    groups = defaultdict(list)
    for r in runs:
        groups[tuple(r['config'][k] for k in keys)].append(r)
    return groups


def series(runs, metric):
    """x (games) and a (seeds, checkpoints) array of `metric`."""
    x = np.array([c['games'] for c in runs[0]['curve']])
    return x, np.array([[c[metric] for c in r['curve']] for r in runs])


def unbeatable(checkpoint):
    return checkpoint['worst_case_black'] == 0 and checkpoint['worst_case_white'] == 0


def games_to_unbeatable(run):
    """First checkpoint from which the agent stays unbeatable to the end (None if never)."""
    result = None
    for c in run['curve']:
        if unbeatable(c):
            result = result or c['games']
        else:
            result = None
    return result


def median_games(reached):
    """Median games until unbeatable over the seeds that got there."""
    return f'{int(np.median(reached)):,}' if reached else '—'


def band(ax, x, ys, color, label, **kwargs):
    ax.plot(x, ys.mean(0), color=color, label=label, **kwargs)
    ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.15, lw=0)


def log_games_axis(ax):
    ax.set_xscale('log')
    ax.set_xlabel('Self-play training games')


def pct_axis(ax, lo=0, hi=1):
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{100 * v:.0f}%'))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fmt(values, digits=3):
    values = np.asarray(values, float)
    return f'{values.mean():.{digits}f} ± {values.std():.{digits}f}'


# ---------------------------------------------------------------- experiment 1


def methods():
    groups = group(load('methods'), 'method')
    print('\n### Experiment 1: methods (final checkpoint, mean ± std over seeds)\n')
    print('| Method | Optimal-move rate | Unbeatable seeds | Games until unbeatable (median of those seeds) '
          '| Score vs random | Table size | Train time (s) |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for m in METHODS:
        runs = groups[(m,)]
        last = [r['curve'][-1] for r in runs]
        g2u = [games_to_unbeatable(r) for r in runs]
        reached = [g for g in g2u if g is not None]
        print(f'| {LABELS[m]} | {fmt([c["optimal_move_rate"] for c in last])} '
              f'| {sum(unbeatable(c) for c in last)}/{len(runs)} '
              f'| {median_games(reached)} '
              f'| {fmt([c["score_vs_random"] for c in last])} '
              f'| {fmt([c["table_size"] for c in last], 0)} | {fmt([r["train_seconds"] for r in runs], 1)} |')

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for m in METHODS:
        runs = groups[(m,)]
        x, opt = series(runs, 'optimal_move_rate')
        band(axes[0], x, opt, COLORS[m], LABELS[m])
        share = np.array([[unbeatable(c) for c in r['curve']] for r in runs]).mean(0)
        axes[1].plot(x, share, color=COLORS[m], label=LABELS[m], drawstyle='steps-post')
        _, loss = series(runs, 'loss_vs_random')
        band(axes[2], x, loss, COLORS[m], LABELS[m])
    axes[0].set_title('Optimal moves (all 4,520 positions)')
    axes[0].set_ylabel('Share of positions')
    pct_axis(axes[0], 0.3, 1)
    axes[1].set_title('Unbeatable by a perfect opponent (both colors)')
    axes[1].set_ylabel('Share of seeds')
    pct_axis(axes[1], -0.03, 1.03)
    axes[2].set_title('Losses against random (200 games)')
    axes[2].set_ylabel('Loss rate')
    pct_axis(axes[2], 0, 0.3)
    for ax in axes:
        log_games_axis(ax)
    axes[0].legend(loc='lower right')
    save(fig, 'methods-learning-curves.png')


# ---------------------------------------------------------------- experiment 2


def symmetry():
    runs = load('methods') + load('symmetry')
    groups = group([r for r in runs if r['config']['method'] in ('td', 'q-learning')], 'method', 'symmetry')
    print('\n### Experiment 2: symmetry (final checkpoint)\n')
    print('| Method | Symmetry | Table size | Optimal-move rate | Unbeatable seeds | Games until unbeatable (median of those seeds) |')
    print('|---|---|---:|---:|---:|---:|')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for (m, sym), rs in sorted(groups.items(), key=lambda kv: (METHODS.index(kv[0][0]), not kv[0][1])):
        last = [r['curve'][-1] for r in rs]
        g2u = [games_to_unbeatable(r) for r in rs]
        reached = [g for g in g2u if g is not None]
        print(f'| {LABELS[m]} | {"on" if sym else "off"} | {fmt([c["table_size"] for c in last], 0)} '
              f'| {fmt([c["optimal_move_rate"] for c in last])} | {sum(unbeatable(c) for c in last)}/{len(rs)} '
              f'| {median_games(reached)} |')
        style = dict(ls='-' if sym else '--')
        label = f'{LABELS[m]}, symmetry {"on" if sym else "off"}'
        x, size = series(rs, 'table_size')
        band(axes[0], x, size, COLORS[m], label, **style)
        x, opt = series(rs, 'optimal_move_rate')
        band(axes[1], x, opt, COLORS[m], label, **style)
    axes[0].set_title('Table size')
    axes[0].set_ylabel('Entries')
    axes[1].set_title('Optimal moves (all 4,520 positions)')
    axes[1].set_ylabel('Share of positions')
    pct_axis(axes[1], 0.3, 1)
    for ax in axes:
        log_games_axis(ax)
    axes[0].legend(loc='upper left', fontsize=9)
    save(fig, 'symmetry.png')


# ---------------------------------------------------------------- experiment 3


def epsilon():
    runs = [r for r in load('methods') + load('epsilon') if r['config']['method'] in ('sarsa', 'q-learning')]
    groups = group(runs, 'method', 'epsilon')
    eps_values = sorted({e for _, e in groups})
    print('\n### Experiment 3: exploration rate (final checkpoint)\n')
    print('| Method | ε | Optimal-move rate | Unbeatable seeds | Score vs random | Table size |')
    print('|---|---:|---:|---:|---:|---:|')
    fig, ax = plt.subplots(figsize=(7, 3.8))
    offsets = {'sarsa': -0.12, 'q-learning': 0.12}
    for m in ('sarsa', 'q-learning'):
        means = []
        for i, e in enumerate(eps_values):
            last = [r['curve'][-1] for r in groups[(m, e)]]
            opt = np.array([c['optimal_move_rate'] for c in last])
            means.append(opt.mean())
            n_unbeatable = sum(unbeatable(c) for c in last)
            if n_unbeatable < len(last):  # call out exploitable policies; all others are unbeatable
                ax.annotate(f'but only {n_unbeatable}/{len(last)} seeds\nare unbeatable', xy=(i + offsets[m], opt.min() - 0.004),
                            xytext=(i + offsets[m] - 0.35, opt.min() - 0.09), fontsize=8.5, color=INK_2, ha='center',
                            arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8))
            ax.scatter(np.full(len(opt), i + offsets[m]), opt, color=COLORS[m], s=18, alpha=0.5, lw=0)
            print(f'| {LABELS[m]} | {e} | {fmt(opt)} | {sum(unbeatable(c) for c in last)}/{len(last)} '
                  f'| {fmt([c["score_vs_random"] for c in last])} | {fmt([c["table_size"] for c in last], 0)} |')
        ax.plot(np.arange(len(eps_values)) + offsets[m], means, color=COLORS[m], marker='o', ms=8,
                label=f'{LABELS[m]} ({"off" if m == "q-learning" else "on"}-policy)')
    ax.set_xticks(range(len(eps_values)), [f'ε = {e}' for e in eps_values])
    ax.set_xlabel('Exploration rate during self-play')
    ax.set_ylabel('Optimal-move rate after 50k games')
    ax.set_title('On-policy vs off-policy learning under more exploration')
    ax.grid(axis='x', visible=False)
    pct_axis(ax, 0.7, 1.01)
    ax.legend(loc='lower right')
    save(fig, 'epsilon.png')


# ---------------------------------------------------------------- experiment 4


def scaling():
    runs = load('scaling')
    if not runs:
        return
    print('\n### Experiment 4: scaling to omok6 (Q-learning)\n')
    print('| Games | Table size | Known states in eval | Score vs random | Score vs heuristic |')
    print('|---:|---:|---:|---:|---:|')
    x, size = series(runs, 'table_size')
    _, known = series(runs, 'known_state_rate')
    _, vs_random = series(runs, 'score_vs_random')
    _, vs_heuristic = series(runs, 'score_vs_heuristic')
    for i in range(0, len(x), 4) if len(x) > 8 else range(len(x)):
        print(f'| {x[i]:,} | {size[:, i].mean():,.0f} | {known[:, i].mean():.1%} '
              f'| {vs_random[:, i].mean():.3f} | {vs_heuristic[:, i].mean():.3f} |')
    print(f'| {x[-1]:,} | {size[:, -1].mean():,.0f} | {known[:, -1].mean():.1%} '
          f'| {vs_random[:, -1].mean():.3f} | {vs_heuristic[:, -1].mean():.3f} |')

    ttt = [r for r in load('methods') if r['config']['method'] == 'q-learning']
    tx, tsize = series(ttt, 'table_size')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    band(axes[0], x, size, COLORS['q-learning'], 'omok6 (6x6, 4 in a row)')
    band(axes[0], tx, tsize, COLORS['q-learning'], 'tic-tac-toe', ls=':')
    axes[0].set_yscale('log')
    axes[0].set_title('Q-table size keeps growing on omok6')
    axes[0].set_ylabel('Entries (log scale)')
    axes[0].legend(loc='upper left')
    band(axes[1], x, known, '#2a78d6', 'Known states in evaluation games')
    band(axes[1], x, vs_random, '#1baf7a', 'Score vs random')
    band(axes[1], x, vs_heuristic, '#eb6834', 'Score vs heuristic')
    axes[1].set_title('omok6: coverage and playing strength')
    axes[1].set_ylabel('Share / score')
    pct_axis(axes[1], -0.02, 1.02)
    axes[1].legend(loc='upper left', fontsize=9)
    for ax in axes:
        log_games_axis(ax)
    save(fig, 'scaling-omok6.png')


# ---------------------------------------------------------------- opening values


def opening_values():
    """X's first-move values learned by each method (seed 0) next to the true values (all draws)."""
    env = make_env('tictactoe')
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, m in zip(axes, METHODS):
        agent = TabularAgent.load(RUNS / 'agents' / f'tictactoe-{m}-eps0.1-sym-seed0.pkl')
        actions, values = agent.action_values(env)
        grid = np.zeros(9)
        grid[actions] = values
        ax.imshow(grid.reshape(3, 3), cmap=DIVERGING, vmin=-1, vmax=1)
        for a, v in zip(actions, values):
            y, x = divmod(a, 3)
            ax.text(x, y, f'{v:+.2f}', ha='center', va='center', fontsize=11, color=INK)
        ax.set_title(LABELS[m])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Learned value of X's first move (true value: 0 everywhere, a draw)", x=0.01, ha='left', fontsize=11)
    save(fig, 'opening-values.png')


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    methods()
    symmetry()
    epsilon()
    scaling()
    opening_values()
    commits = {json.loads(p.read_text())['commit'] for p in RUNS.glob('*/*.json')}
    print(f'\ncommits: {commits}')


if __name__ == '__main__':
    main()
