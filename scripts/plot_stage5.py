"""Summarize the Stage 5 runs: print Markdown tables and save figures to docs/stage5-alphazero/images/.

    uv run python scripts/stage5_alphazero.py   # training runs: runs/stage5/<experiment>/<run>.json
    uv run python scripts/stage5_eval.py        # evaluations:   runs/stage5/eval/*.json
    uv run python scripts/plot_stage5.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from stage5_alphazero import OMOK9_ABLATIONS  # noqa: E402

RUNS = Path('runs/stage5')
PILOT = Path('runs/stage5-pilot')
IMAGES = Path('docs/stage5-alphazero/images')

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
# omok9 score vs the heuristic of the strongest players of earlier stages (same evaluation protocol)
REFERENCES = {'Stage 4: MCTS, heuristic prior + rollouts, 3,200 sims': 0.77, 'Stage 2: DQN': 0.57, 'Stage 3: PPO': 0.42}
# tic-tac-toe optimal-move rate of earlier stages
TICTACTOE_REFERENCES = {'Stage 2: DQN': 0.997, 'Stage 1: Monte Carlo': 0.990,
                        'Stage 4: MCTS with random rollouts, 50 simulations': 0.975}

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK_2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': '#c3c2b7', 'font.size': 10, 'axes.titlesize': 10.5, 'axes.titlelocation': 'left',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.8, 'lines.linewidth': 2, 'legend.frameon': False,
})


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_run(experiment: str, name: str) -> dict | None:
    """A finished run's record, or the curve of an unfinished one parsed from its log."""
    record = load_json(RUNS / experiment / f'{name}.json')
    if record is None and (RUNS / experiment / f'{name}.log').exists():
        record = {'curve': parse_log(RUNS / experiment / f'{name}.log'), 'unfinished': True}
    return record


def parse_log(path: Path) -> list[dict]:
    curve = []
    for line in path.read_text().splitlines():
        if line.startswith('gen='):
            point = {}
            for kv in line.split():
                k, v = kv.split('=')
                point[k] = v == 'True' if v in ('True', 'False') else float(v)
            curve.append(point)
    return curve


def series(curve: list[dict], key: str, x: str = 'gen') -> tuple[np.ndarray, np.ndarray]:
    pts = [(p[x], p[key]) for p in curve if key in p]
    return np.array([a for a, _ in pts]), np.array([b for _, b in pts])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMAGES / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def pct_axis(ax, lo=0, hi=1):
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{100 * v:.0f}%'))


def references(ax, refs=REFERENCES, fmt='{:.2f}'):
    for (label, v), ls in zip(refs.items(), ('--', '-.', ':')):
        ax.axhline(v, color=MUTED, lw=1.2, ls=ls, label=f'{label}: {fmt.format(v)}')


# ---------------------------------------------------------------- tic-tac-toe


def tictactoe():
    ev = load_json(RUNS / 'eval' / 'tictactoe.json')
    run = load_run('tictactoe', 'main-seed0')
    if not ev or not run:
        return
    print('\n### Tic-tac-toe: exact metrics per generation\n')
    print('| Generation | Self-play games | Optimal moves (raw) | Optimal moves (search) | Worst case B / W (raw) | Worst case B / W (search) |')
    print('|---:|---:|---:|---:|---|---|')
    games = {p['gen']: p['games'] for p in run['curve']}
    name = {1.0: 'win', 0.0: 'draw', -1.0: 'loss'}
    for p in ev['curve']:
        print(f'| {p["gen"]} | {games[p["gen"]]:,} | {p["optimal_move_rate_raw"]:.1%} | {p["optimal_move_rate_search"]:.1%} | '
              f'{name[p["worst_case_black_raw"]]} / {name[p["worst_case_white_raw"]]} | '
              f'{name[p["worst_case_black_search"]]} / {name[p["worst_case_white_search"]]} |')
    fig, ax = plt.subplots(figsize=(8, 4))
    x = [games[p['gen']] for p in ev['curve']]
    ax.plot(x, [p['optimal_move_rate_search'] for p in ev['curve']], color=SLOTS[0], marker='o',
            label=f'AlphaZero, with search ({ev["simulations"]} simulations)')
    ax.plot(x, [p['optimal_move_rate_raw'] for p in ev['curve']], color=SLOTS[1], marker='o',
            label='AlphaZero, raw policy (no search)')
    references(ax, TICTACTOE_REFERENCES, '{:.1%}')
    ax.set_xlabel('Self-play games')
    ax.set_ylabel('Share of positions with an optimal move')
    ax.set_title(f'Tic-tac-toe: optimal moves over all {ev["positions"]:,} positions during AlphaZero training')
    pct_axis(ax, 0.6, 1.005)
    ax.legend(loc='lower right', fontsize=8.5)
    save(fig, 'tictactoe-optimal-moves.png')


# ---------------------------------------------------------------- omok9: pilots, main run, ablations

PILOTS = {  # pilot name: (label, color)
    'omok9-reuse4': ('reuse 4, buffer 250k, temp_moves 8 (first settings)', MUTED),
    'reuse10': ('reuse 10, buffer 100k, temp_moves 8', SLOTS[3]),
    'reuse10-temp4': ('reuse 10, buffer 100k, temp_moves 4 (final settings)', SLOTS[0]),
}


def pilots():
    curves = {}
    for name in PILOTS:
        path = PILOT / name / 'main-seed0.log' if name == 'omok9-reuse4' else PILOT / f'{name}.log'
        if path.exists():
            curves[name] = parse_log(path)
    if not curves:
        return
    print('\n### omok9 pilots (20 generations)\n')
    print('| Settings | Gen 5 | Gen 10 | Gen 15 | Gen 20 | Self-play length at gen 10 | Black wins at gen 10 |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for name, curve in curves.items():
        scores = dict(zip(*series(curve, 'score_vs_heuristic_search')))
        g10 = next(p for p in curve if p['gen'] == 10)
        cells = ' | '.join(f'{scores[g]:.2f}' if g in scores else '–' for g in (5, 10, 15, 20))
        print(f'| {PILOTS[name][0]} | {cells} | {g10["game_length"]:.1f} | {g10["black_wins"]:.0%} |')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, curve in curves.items():
        label, color = PILOTS[name]
        axes[0].plot(*series(curve, 'score_vs_heuristic_search'), color=color, marker='o', label=label)
        axes[1].plot(*series(curve, 'game_length'), color=color, label=label)
        axes[2].plot(*series(curve, 'black_wins'), color=color, label=label)
    axes[0].set_title('Score vs the heuristic (with search)')
    axes[0].set_ylim(0, 1)
    axes[1].set_title('Self-play game length (moves)')
    axes[2].set_title('Self-play games won by Black')
    pct_axis(axes[2])
    for ax in axes:
        ax.set_xlabel('Generation (480 self-play games each)')
    axes[0].legend(loc='upper left', fontsize=8)
    save(fig, 'omok9-pilots.png')


def omok9_main():
    run = load_run('omok9', 'main-seed0')
    if not run:
        return
    curve = run['curve']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax = axes[0]
    for key, color, label in (('score_vs_heuristic_search', SLOTS[0], 'AlphaZero, with search (200 simulations)'),
                              ('score_vs_heuristic_raw', SLOTS[1], 'AlphaZero, raw policy (no search)')):
        ax.plot(*series(curve, key, 'games'), color=color, marker='o', ms=4, label=label)
    references(ax)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Self-play games')
    ax.set_ylabel('Score vs the heuristic')
    ax.set_title('omok9: score against the heuristic (100 games, 4-move random openings)')
    ax.legend(loc='lower right', fontsize=8)
    ax = axes[1]
    x, length = series(curve, 'game_length', 'games')
    ax.plot(x, length, color=SLOTS[2], label='game length (moves, left axis)')
    ax.set_ylabel('Moves per self-play game')
    ax.set_xlabel('Self-play games')
    twin = ax.twinx()
    twin.plot(*series(curve, 'black_wins', 'games'), color=INK_2, lw=1.4, label='won by Black')
    twin.plot(*series(curve, 'draws', 'games'), color=SLOTS[3], lw=1.4, label='draws')
    pct_axis(twin)
    twin.grid(False)
    twin.spines['right'].set_visible(True)
    ax.set_title('omok9: self-play games')
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='upper center', bbox_to_anchor=(0.5, -0.17), ncols=3, fontsize=8)
    save(fig, 'omok9-learning.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(*series(curve, 'policy_loss'), color=SLOTS[0], label='policy loss (cross-entropy with pi)')
    axes[0].plot(*series(curve, 'policy_kl'), color=SLOTS[6], label='KL(pi || p)')
    axes[0].set_title('Policy head')
    axes[1].plot(*series(curve, 'value_loss'), color=SLOTS[1], label='value loss (z - v)^2')
    axes[1].set_title('Value head')
    for ax in axes:
        ax.set_xlabel('Generation')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8.5)
    save(fig, 'omok9-losses.png')

    print('\n### omok9 main run\n')
    print('| Generation | Self-play games | Score vs heuristic (search) | Score vs heuristic (raw) | Game length | Draws | Hours |')
    print('|---:|---:|---:|---:|---:|---:|---:|')
    for p in curve:
        if 'score_vs_heuristic_search' in p:
            print(f'| {p["gen"]:.0f} | {p["games"]:,.0f} | {p["score_vs_heuristic_search"]:.2f} | '
                  f'{p["score_vs_heuristic_raw"]:.2f} | {p["game_length"]:.1f} | {p["draws"]:.0%} | {p["seconds"] / 3600:.2f} |')


ABLATION_LABELS = {'main': 'main (temp_moves 4, reuse 10, 200 simulations)', 'temp8': 'temp_moves 8',
                   'reuse4': 'reuse 4', 'sims50': '50 simulations per move', 'sims100': '100 simulations per move', 'no-symmetry': 'no symmetries',
                   'gating': 'gating (AlphaGo Zero)'}


def ablations():
    runs = {'main': load_run('omok9', 'main-seed0')} | {a: load_run('omok9', f'{a}-seed0') for a in OMOK9_ABLATIONS}
    runs = {k: v for k, v in runs.items() if v}
    if len(runs) < 2:
        return
    horizon = 30
    colors = dict(zip(ABLATION_LABELS, [INK, SLOTS[3], SLOTS[1], SLOTS[2], SLOTS[5], SLOTS[4], SLOTS[6]]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for name, run in runs.items():
        curve = [p for p in run['curve'] if p['gen'] <= horizon]
        kw = dict(color=colors[name], marker='o', ms=3.5, lw=2.4 if name == 'main' else 1.6, label=ABLATION_LABELS[name])
        axes[0].plot(*series(curve, 'score_vs_heuristic_search'), **kw)
        x, y = series(curve, 'score_vs_heuristic_search', 'seconds')
        axes[1].plot(x / 3600, y, **kw)
    axes[0].set_xlabel('Generation (480 self-play games each)')
    axes[1].set_xlabel('Wall-clock hours')
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.set_ylabel('Score vs the heuristic (with search)')
    axes[0].set_title('omok9 ablations: one change each from the main run')
    axes[1].set_title('The same runs against time')
    axes[0].legend(loc='lower right', fontsize=8)
    save(fig, 'omok9-ablations.png')

    print('\n### omok9 ablations (score vs heuristic with 200 simulations)\n')
    print('| Run | Gen 10 | Gen 20 | Gen 30 | Raw policy, gen 30 | Hours for 30 generations |')
    print('|---|---:|---:|---:|---:|---:|')
    for name, run in runs.items():
        scores = dict(zip(*series(run['curve'], 'score_vs_heuristic_search')))
        raw = dict(zip(*series(run['curve'], 'score_vs_heuristic_raw')))
        hours = next((p['seconds'] / 3600 for p in run['curve'] if p['gen'] == horizon), float('nan'))
        cells = ' | '.join(f'{scores[g]:.2f}' if g in scores else '–' for g in (10, 20, 30))
        print(f'| {ABLATION_LABELS[name]} | {cells} | {raw.get(30, float("nan")):.2f} | {hours:.2f} |')
    gating = runs.get('gating')
    if gating:
        accepted = [p for p in gating['curve'] if p.get('accepted')]
        print(f'\nGating: {len(accepted)} of {len(gating["curve"])} new networks accepted '
              f'(generations {", ".join(str(int(p["gen"])) for p in accepted)})')


# ---------------------------------------------------------------- evaluations


def elo():
    r = load_json(RUNS / 'eval' / 'omok9-elo.json')
    if not r:
        return
    gens = [(int(n[4:]), e, lo, hi) for n, e, lo, hi in zip(r['names'], r['elo'], r['elo_low'], r['elo_high'])
            if n.startswith('gen-')]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = [g for g, *_ in gens]
    ax.plot(x, [e for _, e, _, _ in gens], color=SLOTS[0], marker='o', label='AlphaZero generation (200 simulations)')
    ax.fill_between(x, [lo for *_, lo, _ in gens], [hi for *_, hi in gens], color=SLOTS[0], alpha=0.15, lw=0,
                    label='95% bootstrap interval')
    ax.axhline(0, color=MUTED, ls='--', lw=1.2, label='heuristic (anchored at 0)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Elo')
    ax.set_title(f'omok9: Elo of the saved generations (round-robin, {r["games_per_pair"]} games per pair)')
    ax.legend(loc='lower right', fontsize=8.5)
    save(fig, 'omok9-elo.png')
    print('\n### omok9 Elo\n')
    print('| Player | Elo | 95% interval |')
    print('|---|---:|---|')
    for n, e, lo, hi in zip(r['names'], r['elo'], r['elo_low'], r['elo_high']):
        print(f'| {n} | {e:.0f} | {lo:.0f} to {hi:.0f} |')


def versus_table(name: str, title: str):
    r = load_json(RUNS / 'eval' / f'{name}.json')
    if not r:
        return None
    print(f'\n### {title}\n')
    print('| A | B | A score | A wins as black | A wins as white | Draws | Moves per game |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for m in r['matches']:
        print(f'| {m["a"]} | {m["b"]} | {m["score"]:.2f} | {m["win_black"]:.0%} | {m["win_white"]:.0%} | '
              f'{m["draws"]:.0%} | {m["game_length"]:.1f} |')
    return r


def simulations():
    r = load_json(RUNS / 'eval' / 'omok9-simulations.json')
    if not r:
        return
    sims = [int(m['a'].rsplit('-', 1)[1]) for m in r['matches']]
    score = [m['score'] for m in r['matches']]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = [max(s, 12) for s in sims]  # put the raw policy (0) on the log axis
    ax.plot(x, score, color=SLOTS[0], marker='o', label=f'final AlphaZero network vs {r["matches"][0]["b"]}')
    ax.set_xscale('log')
    ax.set_xticks(x, ['raw' if s == 0 else f'{s:,}' for s in sims])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Simulations per move (log scale)')
    ax.set_ylabel('Score')
    ax.set_title('omok9: the same network with more search (100 games per point)')
    ax.legend(loc='lower right', fontsize=8.5)
    save(fig, 'omok9-simulations.png')


LARGE = {'freestyle15/scratch-seed0': ('freestyle15, from scratch', MUTED),
         'freestyle15/transfer-seed0': ('freestyle15, starting from the omok9 network', SLOTS[0]),
         'renju15/transfer-seed0': ('renju15, starting from the freestyle15 network', SLOTS[1])}


def large_boards():
    runs = {k: load_run(*k.split('/')) for k in LARGE}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return
    zero = load_json(RUNS / 'eval' / 'freestyle15-zero-shot.json')
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for name, run in runs.items():
        label, color = LARGE[name]
        axes[0].plot(*series(run['curve'], 'score_vs_heuristic_search'), color=color, marker='o', ms=4, label=label)
        axes[1].plot(*series(run['curve'], 'game_length'), color=color, label=label)
        axes[2].plot(*series(run['curve'], 'black_wins'), color=color, label=label)
    if zero:
        search = next(m for m in zero['matches'] if m['a'].endswith('-200'))
        axes[0].axhline(search['score'], color=SLOTS[0], ls=':', lw=1.2,
                        label=f'omok9 network before any 15x15 training: {search["score"]:.2f}')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Score vs the heuristic (with search)')
    axes[1].set_title('Self-play game length (moves)')
    axes[2].set_title('Self-play games won by Black')
    pct_axis(axes[2])
    for ax in axes:
        ax.set_xlabel('Generation (480 self-play games each)')
    axes[0].legend(loc='upper right', fontsize=7.5)
    save(fig, 'large-boards.png')
    print('\n### 15x15 runs (score vs heuristic every 5 generations: search / raw)\n')
    print('| Run | ' + ' | '.join(f'Gen {g}' for g in (5, 10, 20, 30)) + ' | Self-play length at gen 30 | Black wins at gen 30 | Minutes |')
    print('|---|' + '---:|' * 7)
    for name, run in runs.items():
        pts = {p['gen']: p for p in run['curve']}
        cells = ' | '.join(f'{pts[g]["score_vs_heuristic_search"]:.2f} / {pts[g]["score_vs_heuristic_raw"]:.2f}' for g in (5, 10, 20, 30))
        last = run['curve'][-1]
        print(f'| {LARGE[name][0]} | {cells} | {last["game_length"]:.1f} | {last["black_wins"]:.0%} | {last["seconds"] / 60:.0f} |')


def network_views():
    """The network's prior (no search) on the empty board and on Stage 4's blocking test, across generations."""
    import torch

    from omok_rl.envs import make_env
    from omok_rl.nets import AlphaZeroNet, NetEvaluator
    from omok_rl.viz import draw_board
    from stage4_search import BLOCK_CELL, BLOCK_MOVES

    gens = [g for g in (5, 15, 30, 60) if (RUNS / 'omok9' / 'main-seed0' / f'gen-{g:04d}.pt').exists()]
    if not gens:
        return
    positions = [('empty board, Black to move', []), ('White threatens five at the ringed cell, Black to move', BLOCK_MOVES)]
    fig, axes = plt.subplots(2, len(gens), figsize=(3.1 * len(gens), 6.6), squeeze=False)
    print('\n### Network prior on two positions\n')
    print('| Generation | Value of the empty board (Black) | Prior of the blocking move | Value after the threat (Black) |')
    print('|---:|---:|---:|---:|')
    for col, g in enumerate(gens):
        ckpt = torch.load(RUNS / 'omok9' / 'main-seed0' / f'gen-{g:04d}.pt', map_location='cpu', weights_only=False)
        net = AlphaZeroNet(ckpt['in_channels'], ckpt['config']['channels'], ckpt['config']['blocks'])
        net.load_state_dict(ckpt['state_dict'])
        evaluator = NetEvaluator(net.eval(), 'cpu', 9, symmetries=False)
        row_values = []
        for row, (label, moves) in enumerate(positions):
            env = make_env('omok9')
            for m in moves:
                env.move(m)
            probs, values = evaluator(env.get_observation()[None], env.get_legal_mask()[None])
            p, v = probs[0], float(values[0])
            row_values.append((p, v))
            ax = axes[row, col]
            draw_board(ax, 9, moves, numbers=False, highlight_last=False)
            for a in np.flatnonzero(p > 0.1 * p.max()):  # shade relative to the most likely move
                y, x = divmod(int(a), 9)
                ax.add_patch(plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, color=SLOTS[0], alpha=0.85 * p[a] / p.max(),
                                           lw=0, zorder=1.5))
                if p[a] >= 0.05:
                    ax.text(x, y, f'{100 * p[a]:.0f}', ha='center', va='center', fontsize=7, color='white', zorder=3)
            if moves:
                y, x = divmod(BLOCK_CELL, 9)
                ax.add_patch(plt.Circle((x, y), 0.48, fill=False, ec='#e34948', lw=1.8, zorder=4))
            ax.set_title(f'gen {g}: v = {v:+.2f}', fontsize=9.5)
            if col == 0:
                ax.set_ylabel(label, fontsize=8.5)
        (p0, v0), (p1, v1) = row_values
        print(f'| {g} | {v0:+.2f} | {p1[BLOCK_CELL]:.0%} | {v1:+.2f} |')
    fig.suptitle('omok9: the raw network (no search). Shading: prior relative to the top move, numbers: prior in % (if ≥ 5%), v: value for the player to move', x=0.01,
                 ha='left', fontsize=10.5)
    save(fig, 'omok9-network.png')


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    tictactoe()
    pilots()
    omok9_main()
    ablations()
    elo()
    simulations()
    versus_table('omok9-simulations', 'omok9: more search with the final network')
    versus_table('omok9-previous-stages', 'omok9: against the best players of Stages 2-4')
    versus_table('omok9-ablations', 'omok9: ablations (generation 30) against the main run at generation 30, 200 simulations')
    network_views()
    large_boards()
    versus_table('freestyle15-zero-shot', 'freestyle15: the omok9 network before any 15x15 training')
    versus_table('freestyle15-final', 'freestyle15: final networks')
    versus_table('renju15-final', 'renju15: final network')


if __name__ == '__main__':
    main()
