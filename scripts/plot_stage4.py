"""Summarize the Stage 4 runs: print Markdown tables and save figures to docs/stage4-search/images/.

    uv run python scripts/stage4_search.py   # produces runs/stage4/**/*.json
    uv run python scripts/plot_stage4.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from stage4_search import (ABLATIONS, BLOCK_CELL, BLOCK_MOVES, BOARDS, MAX_NODES, VARIANTS, make_mcts)  # noqa: E402

from omok_rl.envs import make_env  # noqa: E402
from omok_rl.symmetry import canonical  # noqa: E402
from omok_rl.viz import draw_board  # noqa: E402

RUNS = Path('runs/stage4')
IMAGES = Path('docs/stage4-search/images')

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
VARIANT_COLORS = dict(zip(VARIANTS, ['#898781', SLOTS[3], SLOTS[4], SLOTS[0], SLOTS[2], SLOTS[6], SLOTS[1]]))
VARIANT_LABELS = {
    'uct-random': 'UCT + random rollouts (pure MCTS)',
    'uct-heuristic': 'UCT + heuristic rollouts',
    'hprior-random': 'heuristic prior + random rollouts',
    'hprior-heuristic': 'heuristic prior + heuristic rollouts',
    'nprior-random': 'PPO prior + random rollouts',
    'nprior-value': 'PPO prior + PPO value (no rollouts)',
    'hprior-value': 'heuristic prior + PPO value',
}
ALPHABETA_COLOR = SLOTS[5]
ABLATION_LABELS = {'all': 'all speed-ups', 'no-alphabeta': 'without alpha-beta cutoffs', 'no-table': 'without the table',
                   'no-symmetry': 'without symmetry', 'no-ordering': 'without move ordering', 'no-threats': 'without threat rules',
                   'plain': 'plain negamax (none)'}
SEQUENTIAL = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#2a78d6', '#0d366b'])
# Stages 2-3, final networks, score vs the heuristic on omok9 (mean over 3 seeds; same evaluation protocol)
REFERENCES = {'DQN (Stage 2)': 0.57, 'PPO, ent_coef 0.03 (Stage 3)': 0.42}
# tic-tac-toe optimal-move rate of earlier stages
TICTACTOE_REFERENCES = {'Monte Carlo (Stage 1)': 0.990, 'DQN (Stage 2)': 0.997, 'PPO, ent_coef 0.1 (Stage 3)': 0.966}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK_2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': '#c3c2b7', 'font.size': 10, 'axes.titlesize': 10.5, 'axes.titlelocation': 'left',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.8, 'lines.linewidth': 2, 'legend.frameon': False,
})


def load(experiment: str) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted((RUNS / experiment).glob('*.json'))}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def pct_axis(ax, lo=0, hi=1):
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{100 * v:.0f}%'))


def sims_axis(ax, sims):
    ax.set_xscale('log')
    ax.set_xticks(sims, [f'{s:,}' for s in sims])
    ax.minorticks_off()
    ax.set_xlabel('Simulations per move (log scale)')


def fmt_nodes(r):
    return f'{r["nodes"]:,}' if r['value'] is not None else f'> {MAX_NODES:,}'


# ---------------------------------------------------------------- exact solving


def solving():
    runs = load('solve')
    names = {'tictactoe': 'tic-tac-toe (3x3, 3)', '4x4k3': '4x4, 3 in a row', '4x4k4': '4x4, 4 in a row',
             '5x5k4': '5x5, 4 in a row', 'omok6': '`omok6` (6x6, 4 in a row)'}
    value_names = {1.0: 'first player wins', 0.0: 'draw', -1.0: 'second player wins', None: 'unknown'}
    print('\n### Exact values (all speed-ups)\n')
    print('| Board | Value | Positions searched | Table entries | Seconds |')
    print('|---|---|---:|---:|---:|')
    for board in BOARDS:
        r = runs.get(f'{board}-all')
        if r:
            print(f'| {names[board]} | {value_names[r["value"]]} | {fmt_nodes(r)} | {r["table"]:,} | {r["seconds"]:.1f} |')

    print('\n### Positions searched with one speed-up switched off\n')
    shown = [b for b in BOARDS if b != '4x4k3']
    print('| Solver | ' + ' | '.join(names[b] for b in shown) + ' |')
    print('|---|' + '---:|' * len(shown))
    for ablation in ABLATIONS:
        cells = [fmt_nodes(runs[f'{b}-{ablation}']) if f'{b}-{ablation}' in runs else '' for b in shown]
        print(f'| {ABLATION_LABELS[ablation]} | ' + ' | '.join(cells) + ' |')

    fig, ax = plt.subplots(figsize=(11, 4))
    width = 0.8 / len(ABLATIONS)
    colors = dict(zip(ABLATIONS, [SLOTS[0], SLOTS[1], SLOTS[2], SLOTS[3], SLOTS[4], SLOTS[6], MUTED]))
    for k, ablation in enumerate(ABLATIONS):
        for i, b in enumerate(shown):
            r = runs.get(f'{b}-{ablation}')
            if not r:
                continue
            done = r['value'] is not None
            ax.bar(i + (k - len(ABLATIONS) / 2 + 0.5) * width, r['nodes'], width, color=colors[ablation],
                   hatch=None if done else '///', edgecolor=SURFACE if done else INK_2, lw=0.5,
                   label=ablation if i == 0 else None)
    ax.axhline(MAX_NODES, color=INK_2, lw=1, ls=':')
    ax.text(len(shown) - 0.5, MAX_NODES * 1.3, 'node limit', ha='right', fontsize=8.5, color=INK_2)
    ax.set_yscale('log')
    ax.set_ylim(50, 3e9)  # room for the legend above the bars
    ax.set_xticks(range(len(shown)), [names[b].replace('`', '') for b in shown])
    ax.set_ylabel('Positions searched (log scale)')
    ax.set_title('Exact solving: positions searched with each speed-up switched off (hatched = not solved within the limit)')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [ABLATION_LABELS[x] for x in labels], loc='upper left', fontsize=8.5, ncols=4)
    ax.grid(axis='x', visible=False)
    save(fig, 'solve-nodes.png')

    first = runs.get('omok6-first-moves')
    if first:
        size = 6
        values = {int(a): v for a, v in first['moves'].items()}
        print('\n### omok6 first moves\n')
        print('| First moves (one symmetry class per row) | Value for Black | Positions searched | Seconds |')
        print('|---|---|---:|---:|')
        classes = defaultdict(list)  # symmetric first moves share one search
        for a in sorted(values):
            board = np.zeros(size * size, np.uint8)
            board[a] = 1
            classes[canonical(board, size)[0]].append(a)
        for cells in classes.values():
            v = values[cells[0]]
            names_ = ', '.join(f'{"ABCDEF"[c % size]}{size - c // size}' for c in cells)
            nodes = f'{v["nodes"]:,}' if v['value'] is not None else f'> {MAX_NODES:,} (limit)'
            print(f'| {names_} | {value_names[v["value"]]} | {nodes} | {v["seconds"]:.0f} |')
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        draw_board(ax, size, [])
        cmap = {1.0: SLOTS[2], 0.0: SLOTS[3], -1.0: '#e34948', None: '#c3c2b7'}
        for a, v in values.items():
            y, x = divmod(a, size)
            ax.add_patch(plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, color=cmap[v['value']], alpha=0.85, zorder=1.5))
        present = sorted({v['value'] for v in values.values()}, key=lambda v: -2 if v is None else -v)
        ax.legend(handles=[Patch(color=cmap[v], label=value_names[v]) for v in present], loc='upper center',
                  bbox_to_anchor=(0.5, -0.08), ncols=2, fontsize=8.5)
        ax.set_title('omok6: exact value of each first move for Black', fontsize=10)
        save(fig, 'omok6-first-moves.png')


# ---------------------------------------------------------------- MCTS on tic-tac-toe


def tictactoe():
    runs = load('tictactoe')
    groups = defaultdict(list)
    for r in runs.values():
        groups[r['simulations']].append(r)
    if not groups:
        return
    sims = sorted(groups)
    rate = np.array([[r['optimal_move_rate'] for r in groups[s]] for s in sims])
    ms = np.array([np.mean([r['ms_per_move'] for r in groups[s]]) for s in sims])
    print('\n### MCTS (UCT + random rollouts) on tic-tac-toe: optimal moves over all 4,520 positions\n')
    print('| Simulations per move | Optimal-move rate | ms per move |')
    print('|---:|---:|---:|')
    for s, r, m in zip(sims, rate, ms):
        print(f'| {s:,} | {r.mean():.3f} ± {r.std():.3f} | {m:.1f} |')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sims, rate.mean(1), color=SLOTS[0], marker='o', label='MCTS, UCT + random rollouts')
    ax.fill_between(sims, rate.min(1), rate.max(1), color=SLOTS[0], alpha=0.15, lw=0)
    for (label, v), ls in zip(TICTACTOE_REFERENCES.items(), ('--', '-.', ':')):
        ax.axhline(v, color=MUTED, lw=1.2, ls=ls, label=f'{label}: {100 * v:.1f}%')
    sims_axis(ax, sims)
    ax.set_ylabel('Share of positions with an optimal move')
    ax.set_title('Tic-tac-toe: MCTS without any learning vs the trained agents of Stages 1-3')
    pct_axis(ax, 0.85, 1.005)
    ax.legend(loc='lower right', fontsize=8.5)
    save(fig, 'tictactoe-mcts.png')


# ---------------------------------------------------------------- blocking one threat


def blocking():
    runs = load('block')
    if not runs:
        return
    by_variant = defaultdict(list)
    for r in runs.values():
        by_variant[r['variant']].append(r)
    print('\n### Blocking a single four on omok9 (20 searches per point)\n')
    sims = sorted({r['simulations'] for r in runs.values()})
    print('| Variant | ' + ' | '.join(f'{s:,}' for s in sims) + ' |')
    print('|---|' + '---:|' * len(sims))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for v, rs in by_variant.items():
        rs.sort(key=lambda r: r['simulations'])
        print(f'| {VARIANT_LABELS[v]} | ' + ' | '.join(f'{r["block_rate"]:.2f}' for r in rs) + ' |')
        x = [r['simulations'] for r in rs]
        axes[0].plot(x, [r['block_rate'] for r in rs], marker='o', color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
        share = np.array([r['visit_share'] for r in rs])
        axes[1].plot(x, share.mean(1), marker='o', color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
        axes[1].fill_between(x, share.min(1), share.max(1), color=VARIANT_COLORS[v], alpha=0.15, lw=0)
    legal = next(iter(runs.values()))['legal_moves']
    axes[1].axhline(1 / legal, color=MUTED, lw=1, ls=':')
    axes[1].text(sims[-1], 1.3 / legal, f'uniform (1/{legal})', ha='right', fontsize=8.5, color=INK_2)
    axes[0].set_title('Share of searches that play the only block')
    axes[0].set_ylabel('Block rate (20 seeds)')
    axes[1].set_title('Share of the root visits that go to the block')
    axes[1].set_ylabel('Visit share')
    axes[1].set_yscale('log')
    for ax in axes:
        sims_axis(ax, sims)
    pct_axis(axes[0], -0.02, 1.02)
    axes[0].legend(loc='center right', fontsize=8.5)
    save(fig, 'block-threat.png')

    # where the visits go: pure MCTS vs a heuristic prior, same position, 1,600 simulations
    env = make_env('omok9')
    for m in BLOCK_MOVES:
        env.move(m)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, variant in zip(axes, ('uct-random', 'hprior-random')):
        agent = make_mcts(variant, 1600, seed=0)
        played = agent.act(env)
        moves, share = agent.root_policy()
        draw_board(ax, 9, BLOCK_MOVES, numbers=True, highlight_last=False)
        for a, p in zip(moves, share):
            y, x = divmod(int(a), 9)
            ax.add_patch(plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, color=SEQUENTIAL(min(1.0, p / share.max())),
                                       alpha=0.85, zorder=1.5))
        by, bx = divmod(BLOCK_CELL, 9)
        ax.text(bx, by, f'{share[moves == BLOCK_CELL][0]:.2f}', ha='center', va='center', fontsize=7.5,
                color='white', zorder=3)
        py, px = divmod(played, 9)
        ax.add_patch(plt.Circle((px, py), 0.5, facecolor='none', edgecolor='#e34948', lw=2, zorder=4))
        ax.set_title(f'{VARIANT_LABELS[variant]}, 1,600 simulations\nshare of root visits (darker = more); '
                     f'red ring = move played', fontsize=9.5)
    save(fig, 'block-visits.png')


# ---------------------------------------------------------------- omok9 strength


def omok9():
    runs = load('omok9')
    if not runs:
        return
    print('\n### omok9: score against the heuristic (100 games, 50 random 4-move openings x 2 colors)\n')
    print('| Player | Simulations / depth | Score | Win as Black | Win as White | Draws | Seconds per move |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for v in VARIANTS:
        rs = sorted((r for k, r in runs.items() if k.rsplit('-', 1)[0] == v), key=lambda r: int(r['a'].split('/')[-1]))
        if not rs:
            continue
        sims = [int(r['a'].split('/')[-1]) for r in rs]
        for s, r in zip(sims, rs):
            print(f'| {VARIANT_LABELS[v]} | {s:,} | {r["score"]:.2f} | {r["win_black"]:.2f} | {r["win_white"]:.2f} '
                  f'| {r["draws"]:.2f} | {r["seconds_per_move_a"]:.2f} |')
        scores = [r['score'] for r in rs]
        axes[0].plot(sims, scores, marker='o', color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
        axes[1].plot([r['seconds_per_move_a'] for r in rs], scores, marker='o', color=VARIANT_COLORS[v])
    ab = sorted((r for k, r in runs.items() if k.startswith('alphabeta')), key=lambda r: int(r['a'].split('/')[-1]))
    for r in ab:
        print(f'| alpha-beta, width 8 | depth {r["a"].split("/")[-1]} | {r["score"]:.2f} | {r["win_black"]:.2f} '
              f'| {r["win_white"]:.2f} | {r["draws"]:.2f} | {r["seconds_per_move_a"]:.3f} |')
    if ab:
        xs = [r['seconds_per_move_a'] for r in ab]
        axes[1].plot(xs, [r['score'] for r in ab], marker='s', color=ALPHABETA_COLOR, label='alpha-beta (depth 1-5)')
        for x, r in zip(xs, ab):
            axes[1].annotate(r['a'].split('/')[-1], (x, r['score']), textcoords='offset points', xytext=(0, 6),
                             ha='center', fontsize=8, color=ALPHABETA_COLOR)
    for ax in axes:
        for (label, v), ls in zip(REFERENCES.items(), ('--', ':')):
            ax.axhline(v, color=MUTED, lw=1.2, ls=ls, label=f'{label}: {v:.2f}' if ax is axes[1] else None)
        ax.axhline(0.5, color=INK_2, lw=0.8)
        pct_axis(ax, -0.02, 1.02)
        ax.set_ylabel('Score vs heuristic (win 1, draw ½)')
    sims_axis(axes[0], sorted({int(r['a'].split('/')[-1]) for k, r in runs.items() if not k.startswith('alphabeta')}))
    axes[0].set_title('omok9: MCTS strength vs simulations')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Seconds per move (log scale, 8 searches sharing the CPU)')
    axes[1].set_title('Strength vs thinking time, including alpha-beta')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[1].legend(loc='upper left', fontsize=8)
    save(fig, 'omok9-strength.png')


def versus():
    runs = load('versus')
    if not runs:
        return
    names = {'mcts/hprior-heuristic/1600': 'MCTS, heuristic prior + rollouts, 1,600 sims',
             'mcts/nprior-value/1600': 'MCTS, PPO prior + value, 1,600 sims',
             'alphabeta/4': 'alpha-beta, depth 4', 'ppo': 'PPO (Stage 3, greedy)', 'dqn': 'DQN (Stage 2, greedy)'}
    print('\n### omok9 head-to-head (100 games, 50 random 4-move openings x 2 colors; score of player A)\n')
    print('| Player A | Player B | Score of A | A wins as Black | A wins as White | Draws |')
    print('|---|---|---:|---:|---:|---:|')
    for r in runs.values():
        print(f'| {names[r["a"]]} | {names[r["b"]]} | {r["score"]:.2f} | {r["win_black"]:.2f} | {r["win_white"]:.2f} '
              f'| {r["draws"]:.2f} |')


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    for fn in (solving, tictactoe, blocking, omok9, versus):
        fn()
    commits = {json.loads(p.read_text()).get('commit') for p in RUNS.glob('*/*.json')}
    print(f'\ncommits: {commits}')


if __name__ == '__main__':
    main()
