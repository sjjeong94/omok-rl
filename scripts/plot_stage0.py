"""Summarize the Stage 0 runs: print Markdown tables and save figures to docs/stage0-foundations/images/.

    uv run python scripts/stage0_baselines.py   # produces runs/stage0/*.json
    uv run python scripts/plot_stage0.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from omok_rl.envs import make_env
from omok_rl.viz import draw_board

RUNS = Path('runs/stage0')
IMAGES = Path('docs/stage0-foundations/images')
ENVS = ['tictactoe', 'omok6', 'omok9', 'freestyle15', 'renju15']
ENV_LABELS = {
    'tictactoe': 'Tic-tac-toe (3x3, 3)',
    'omok6': 'omok6 (6x6, 4)',
    'omok9': 'omok9 (9x9, 5)',
    'freestyle15': 'freestyle15 (15x15, 5)',
    'renju15': 'renju15 (15x15, 5, Renju)',
}
BLACK, WHITE = 1, 2

# Reference palette (light mode): categorical slots 1-2, muted ink for draws.
SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
C_BLACK, C_WHITE, C_DRAW = '#2a78d6', '#eb6834', '#c3c2b7'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK_2, 'xtick.color': MUTED, 'ytick.color': INK_2,
    'axes.edgecolor': GRID, 'font.size': 10,
})


def load(a, b, env):
    return json.loads((RUNS / f'{a}-vs-{b}-{env}.json').read_text())


def outcome_rates(run):
    """Black win, draw and white win rates over all games of a run (colors, not agents)."""
    winners = np.array([g['winner'] for g in run['games']])
    return (winners == BLACK).mean(), (winners == 0).mean(), (winners == WHITE).mean()


def a_rates(run, color):
    """Agent A's win / draw / loss rates when A plays `color`."""
    games = [g for g in run['games'] if g['a_color'] == color]
    win = np.mean([g['winner'] == color for g in games])
    draw = np.mean([g['winner'] == 0 for g in games])
    return win, draw, max(0.0, 1 - win - draw)


def pct(x):
    return f'{100 * x:.1f}%'


def table_color_outcomes(a, b, envs):
    print(f'\n### {a} vs {b}: outcome by color\n')
    print('| Env | Games | Black wins | Draws | White wins | Mean length (moves) |')
    print('|---|---:|---:|---:|---:|---:|')
    for env in envs:
        run = load(a, b, env)
        lengths = [len(g['moves']) for g in run['games']]
        rates = ' | '.join(pct(r) for r in outcome_rates(run))
        print(f'| `{env}` | {len(run["games"])} | {rates} | {np.mean(lengths):.1f} |')


def table_agent_results(pairs):
    print('\n### Agent A results (win / draw / loss)\n')
    print('| Agent A | Agent B | Env | Games | A as black | A as white | A score |')
    print('|---|---|---|---:|---|---|---:|')
    for a, b, env in pairs:
        run = load(a, b, env)
        by_color = ['/'.join(f'{100 * r:.0f}' for r in a_rates(run, c)) for c in (BLACK, WHITE)]
        wins = np.mean([g['winner'] == g['a_color'] for g in run['games']])
        draws = np.mean([g['winner'] == 0 for g in run['games']])
        print(f'| {a} | {b} | `{env}` | {len(run["games"])} | {by_color[0]} | {by_color[1]} | {wins + 0.5 * draws:.3f} |')


def table_speed():
    print('\n### Speed of random self-play (single process)\n')
    print('| Env | Moves / second | Games / second |')
    print('|---|---:|---:|')
    for env in ENVS:
        run = load('random', 'random', env)
        moves = sum(len(g['moves']) for g in run['games'])
        print(f'| `{env}` | {moves / run["seconds"]:,.0f} | {len(run["games"]) / run["seconds"]:,.0f} |')


def figure_first_move_advantage():
    """Stacked outcome bars (black win / draw / white win) for random and heuristic self-play."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, agent in zip(axes, ('random', 'heuristic')):
        runs = [load(agent, agent, env) for env in ENVS]
        rates = np.array([outcome_rates(r) for r in runs])
        y = np.arange(len(ENVS))[::-1]
        left = np.zeros(len(ENVS))
        for i, (label, color) in enumerate((('Black wins', C_BLACK), ('Draw', C_DRAW), ('White wins', C_WHITE))):
            ax.barh(y, rates[:, i], left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2, label=label)
            for yi, l, r in zip(y, left, rates[:, i]):
                if r >= 0.08:
                    ax.text(l + r / 2, yi, f'{100 * r:.0f}%', ha='center', va='center', fontsize=8.5,
                            color='white' if color != C_DRAW else INK)
            left += rates[:, i]
        n = len(runs[0]['games'])
        ax.set_title(f'{agent} vs {agent} ({n} games per env)', fontsize=10.5, loc='left', color=INK)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'])
        ax.axvline(0.5, color=MUTED, lw=0.8, ls=':', zorder=0)
        ax.tick_params(length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(np.arange(len(ENVS))[::-1], [ENV_LABELS[e] for e in ENVS])
    axes[0].set_xlabel('Share of games')
    axes[1].set_xlabel('Share of games')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Who wins when both sides play the same way?', x=0.01, y=1.1, ha='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(IMAGES / 'first-move-advantage.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def figure_game_length():
    """Distribution of random self-play game length as a fraction of the board.
    Tic-tac-toe is left out: its lengths are only 5-9 moves, too discrete for a density plot."""
    envs = ENVS[1:]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    data = []
    for env in envs:
        run = load('random', 'random', env)
        cells = make_env(env).size ** 2
        data.append([len(g['moves']) / cells for g in run['games']])
    y = np.arange(len(envs))[::-1]
    parts = ax.violinplot(data, positions=y, orientation='horizontal', widths=0.75, showmedians=True, showextrema=False)
    for body in parts['bodies']:
        body.set_facecolor(C_BLACK)
        body.set_edgecolor(C_BLACK)
        body.set_alpha(0.35)
    parts['cmedians'].set_color(INK)
    for yi, d in zip(y, data):
        ax.text(1.02, yi, f'median {100 * np.median(d):.0f}%', va='center', fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [ENV_LABELS[e] for e in envs])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'])
    ax.set_xlabel('Game length (share of board cells filled)')
    ax.set_title('Random self-play: how full is the board when the game ends?', loc='left', fontsize=11)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGES / 'random-game-length.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def figure_example_games():
    """Final positions of a heuristic win over random and a pretrained win over heuristic."""
    examples = [
        ('heuristic', 'random', 'omok9', 'heuristic (black) vs random (white)'),
        ('heuristic', 'pretrained0', 'renju15', 'heuristic (black) vs pretrained0 (white)'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={'width_ratios': [9, 15]})
    for ax, (a, b, env, title) in zip(axes, examples):
        run = load(a, b, env)
        # first game where A is black and the outcome is typical for the match (A wins vs random, loses vs pretrained)
        want = BLACK if b == 'random' else WHITE
        game = next(g for g in run['games'] if g['a_color'] == BLACK and g['winner'] == want)
        size = make_env(env).size
        draw_board(ax, size, game['moves'])
        winner = 'Black' if game['winner'] == BLACK else 'White'
        ax.set_title(f'{title}\n{env}, {winner} wins in {len(game["moves"])} moves', fontsize=10, loc='left')
    fig.tight_layout()
    fig.savefig(IMAGES / 'example-games.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    table_color_outcomes('random', 'random', ENVS)
    table_color_outcomes('heuristic', 'heuristic', ENVS)
    table_agent_results(
        [('heuristic', 'random', env) for env in ENVS]
        + [('heuristic', m, env) for m in ('pretrained0', 'pretrained1') for env in ENVS[3:]]
        + [('pretrained0', 'pretrained1', env) for env in ENVS[3:]]
    )
    table_speed()
    commits = {json.loads(path.read_text())['commit'] for path in RUNS.glob('*.json')}
    print(f'\ncommits: {commits}')
    figure_first_move_advantage()
    figure_game_length()
    figure_example_games()


if __name__ == '__main__':
    main()
