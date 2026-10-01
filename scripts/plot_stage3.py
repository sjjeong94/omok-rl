"""Summarize the Stage 3 runs: print Markdown tables and save figures to docs/stage3-pg/images/.

    uv run python scripts/stage3_pg.py   # produces runs/stage3/**/*.json and *.pt
    uv run python scripts/plot_stage3.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from omok_rl.agents import HeuristicAgent, make_agent
from omok_rl.agents.policy import PolicyAgent
from omok_rl.arena import evaluate, play_game
from omok_rl.envs import make_env
from omok_rl.viz import draw_board

RUNS = Path('runs/stage3')
DQN_RUNS = Path('runs/stage2')
IMAGES = Path('docs/stage3-pg/images')

SURFACE, INK, INK_2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
ALGO_COLORS = {'reinforce': SLOTS[3], 'reinforce-baseline': SLOTS[1], 'a2c': SLOTS[2], 'ppo': SLOTS[0],
               'dqn': MUTED, 'pool': SLOTS[6], 'a2c-16mb': SLOTS[4], 'noclip': SLOTS[5]}
ALGO_LABELS = {'reinforce': 'REINFORCE', 'reinforce-baseline': 'REINFORCE + baseline', 'a2c': 'A2C', 'ppo': 'PPO',
               'dqn': 'DQN, Double + Dueling + 3-step (Stage 2)', 'pool': 'PPO + opponent pool',
               'a2c-16mb': 'A2C, 16 minibatches', 'noclip': 'PPO without clipping'}
SEQUENTIAL = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#2a78d6', '#0d366b'])
ENT_COLORS = dict(zip(('ent0', 'ent0.003', 'ent0.01', 'ent0.03', 'ent0.1'), SLOTS))

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'text.color': INK, 'axes.labelcolor': INK_2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': '#c3c2b7', 'font.size': 10, 'axes.titlesize': 10.5, 'axes.titlelocation': 'left',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.8, 'lines.linewidth': 2, 'legend.frameon': False,
})


def load(experiment, root=RUNS):
    """{variant: [runs]} for an experiment; run files are named <variant>-seed<k>.json."""
    groups = defaultdict(list)
    for p in sorted((root / experiment).glob('*.json')):
        groups[p.stem.rsplit('-seed', 1)[0]].append(json.loads(p.read_text()) | {'path': p.with_suffix('')})
    return groups


def series(runs, metric, x='steps'):
    n = min(len(r['curve']) for r in runs)
    xs = np.array([c[x] for c in runs[0]['curve'][:n]])
    return xs, np.array([[c[metric] for c in r['curve'][:n]] for r in runs])


def band(ax, x, ys, color, label, **kwargs):
    ax.plot(x, ys.mean(0), color=color, label=label, **kwargs)
    ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.15, lw=0)


def steps_axis(ax, label='Self-play moves (both players)'):
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v / 1e6:g}M' if v >= 1e6 else f'{v / 1e3:g}k'))
    ax.set_xlabel(label)


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


def first_step(run, metric, threshold):
    return next((c['steps'] for c in run['curve'] if c[metric] >= threshold), None)


def fmt_steps(steps):
    steps = [s for s in steps if s is not None]
    return f'{np.median(steps) / 1e6:.1f}M ({len(steps)}/3)' if steps else 'never'


def unbeatable(c):
    return c['worst_case_black'] == 0 and c['worst_case_white'] == 0


def last_k(run, metric, k=5):
    return [c[metric] for c in run['curve'][-k:]]


def jitter(run, metric='score_vs_heuristic', k=10):
    """Mean absolute change of `metric` between consecutive checkpoints, over the last k."""
    return float(np.mean(np.abs(np.diff(last_k(run, metric, k)))))


# ---------------------------------------------------------------- tic-tac-toe


def tictactoe():
    groups = load('tictactoe')
    groups['ent0.01'] = groups.pop('ppo')
    dqn = load('tictactoe', DQN_RUNS)['all']
    print('\n### Tic-tac-toe (exact metrics, final checkpoint)\n')
    print('| Run | Final optimal-move rate | Best optimal-move rate | Unbeatable at the end | Checkpoints unbeatable '
          '| Final entropy (nats) |')
    print('|---|---:|---:|---:|---:|---:|')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    variants = [('a2c', 'A2C (ent_coef 0.01)', SLOTS[2])] + [
        (f'ent{e}', f'PPO, ent_coef {e}', ENT_COLORS[f'ent{e}']) for e in ('0', '0.01', '0.03', '0.1')]
    for v, label, color in variants:
        runs = groups.get(v)
        if not runs:
            continue
        last = [r['curve'][-1] for r in runs]
        share = [np.mean([unbeatable(c) for c in r['curve']]) for r in runs]
        best = [max(c['optimal_move_rate'] for c in r['curve']) for r in runs]
        print(f'| {label} | {fmt([c["optimal_move_rate"] for c in last], 3)} | {fmt(best, 3)} '
              f'| {sum(unbeatable(c) for c in last)}/{len(runs)} | {fmt(share)} | {fmt([c["entropy"] for c in last])} |')
        x, y = series(runs, 'optimal_move_rate')
        band(axes[0], x, y, color, label)
        x, y = series(runs, 'entropy')
        band(axes[1], x, y, color, label)
    x, y = series(dqn, 'optimal_move_rate')
    axes[0].plot(x, y.mean(0), color=MUTED, ls='--', lw=1.5, label='DQN (Stage 2)')
    axes[0].set_title('Tic-tac-toe: optimal moves of the greedy policy (all 4,520 positions)')
    axes[0].set_ylabel('Share of positions')
    pct_axis(axes[0], 0.6, 1.005)
    axes[1].set_title('Policy entropy of the moves played in training')
    axes[1].set_ylabel('Entropy (nats)')
    for ax in axes:
        steps_axis(ax)
    axes[0].legend(loc='lower right', fontsize=8.5)
    save(fig, 'tictactoe.png')


# ---------------------------------------------------------------- omok9: algorithms


def algorithms():
    groups = load('algorithms')
    dqn = load('omok9', DQN_RUNS)['all']
    print('\n### omok9 algorithms (final checkpoint, mean ± std over 3 seeds)\n')
    print('| Algorithm | Score vs random | Score vs heuristic | Win as Black | Win as White | Best score vs heuristic '
          '| Moves to first 0.5 vs heuristic | Gradient steps | Minutes |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for v in ('dqn', *ALGO_LABELS):
        runs = dqn if v == 'dqn' else groups.get(v)
        if not runs:
            continue
        last = [r['curve'][-1] for r in runs]
        best = [max(c['score_vs_heuristic'] for c in r['curve']) for r in runs]
        grad = [c['updates'] * r['config'].get('epochs', 1) * r['config'].get('minibatches', 1)
                for c, r in zip(last, runs)]
        if v in groups or v == 'dqn':
            print(f'| {ALGO_LABELS[v]} | {fmt([c["score_vs_random"] for c in last])} '
                  f'| {fmt([c["score_vs_heuristic"] for c in last])} | {fmt([c["win_black_vs_heuristic"] for c in last])} '
                  f'| {fmt([c["win_white_vs_heuristic"] for c in last])} | {fmt(best)} '
                  f'| {fmt_steps([first_step(r, "score_vs_heuristic", 0.5) for r in runs])} '
                  f'| {np.mean(grad):,.0f} | {fmt([r["seconds"] / 60 for r in runs], 0)} |')
        if v == 'pool':
            continue
        x, y = series(runs, 'score_vs_heuristic')
        band(axes[0], x, y, ALGO_COLORS[v], ALGO_LABELS[v], ls='--' if v == 'dqn' else '-')
        if v != 'dqn':
            x, y = series(runs, 'entropy')
            band(axes[1], x, y, ALGO_COLORS[v], ALGO_LABELS[v])
    axes[0].set_title('omok9: score against the heuristic (100 games, random openings)')
    axes[0].set_ylabel('Score (win 1, draw ½)')
    pct_axis(axes[0], -0.02, 1.02)
    axes[1].set_title('Policy entropy of the moves played in training')
    axes[1].set_ylabel('Entropy (nats)')
    for ax in axes:
        steps_axis(ax)
    axes[0].legend(loc='upper left', fontsize=8.5)
    save(fig, 'algorithms-omok9.png')

    # diagnostics of the learning signal
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for v in ('reinforce', 'reinforce-baseline', 'a2c', 'ppo'):
        if v not in groups:
            continue
        x, y = series(groups[v], 'explained_variance')
        band(axes[0], x, y, ALGO_COLORS[v], ALGO_LABELS[v])
        x, y = series(groups[v], 'game_length')
        band(axes[1], x, y, ALGO_COLORS[v], ALGO_LABELS[v])
    axes[0].set_title('Explained variance of returns by V(s)')
    axes[0].set_ylabel('1 − Var[G − V] / Var[G]')
    axes[1].set_title('Average self-play game length')
    axes[1].set_ylabel('Moves per game')
    for ax in axes:
        steps_axis(ax)
    axes[0].legend(loc='lower right', fontsize=9)
    save(fig, 'diagnostics-omok9.png')


# ---------------------------------------------------------------- omok9: what makes PPO work


def updates():
    groups = load('updates') | {v: load('algorithms')[v] for v in ('a2c', 'ppo')}
    print('\n### omok9 update rules (final checkpoint)\n')
    print('| Update | Gradient steps per batch | Uses of each sample | Score vs heuristic | Best score vs heuristic '
          '| Final entropy (nats) | Mean approx. KL |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for v in ('a2c', 'a2c-16mb', 'noclip', 'ppo'):
        runs = groups.get(v)
        if not runs:
            continue
        c = runs[0]['config']
        last = [r['curve'][-1] for r in runs]
        best = [max(x['score_vs_heuristic'] for x in r['curve']) for r in runs]
        kl = [np.mean([x['approx_kl'] for x in r['curve']]) for r in runs]
        print(f'| {ALGO_LABELS[v]} | {c["epochs"] * c["minibatches"]} | {c["epochs"]} '
              f'| {fmt([x["score_vs_heuristic"] for x in last])} | {fmt(best)} | {fmt([x["entropy"] for x in last])} '
              f'| {np.mean(kl):.3g} |')
        x, y = series(runs, 'score_vs_heuristic')
        band(axes[0], x, y, ALGO_COLORS[v], ALGO_LABELS[v])
        x, y = series(runs, 'approx_kl')
        band(axes[1], x, np.maximum(y, 1e-6), ALGO_COLORS[v], ALGO_LABELS[v])
    axes[0].set_title('omok9: score against the heuristic')
    axes[0].set_ylabel('Score (win 1, draw ½)')
    pct_axis(axes[0], -0.02, 1.02)
    axes[1].set_title('How far each batch moves the policy (approx. KL, log scale)')
    axes[1].set_ylabel('KL(π_old ‖ π) per move (nats)')
    axes[1].set_yscale('log')
    for ax in axes:
        steps_axis(ax)
    axes[0].legend(loc='upper left', fontsize=9)
    save(fig, 'updates-omok9.png')


# ---------------------------------------------------------------- omok9: entropy coefficient


def entropy_sweep():
    groups = load('entropy') | {'ent0.01': load('algorithms')['ppo']}
    print('\n### omok9 PPO entropy coefficient (final checkpoint)\n')
    print('| ent_coef | Score vs heuristic | Best score vs heuristic | Final entropy (nats) | Last-10 jitter |')
    print('|---:|---:|---:|---:|---:|')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for v in sorted(groups, key=lambda k: float(k[3:])):
        runs = groups[v]
        last = [r['curve'][-1] for r in runs]
        best = [max(c['score_vs_heuristic'] for c in r['curve']) for r in runs]
        print(f'| {v[3:]} | {fmt([c["score_vs_heuristic"] for c in last])} | {fmt(best)} '
              f'| {fmt([c["entropy"] for c in last])} | {fmt([jitter(r) for r in runs])} |')
        label = f'ent_coef = {v[3:]}'
        x, y = series(runs, 'score_vs_heuristic')
        band(axes[0], x, y, ENT_COLORS[v], label)
        x, y = series(runs, 'entropy')
        band(axes[1], x, y, ENT_COLORS[v], label)
    axes[0].set_title('omok9 PPO: score against the heuristic')
    axes[0].set_ylabel('Score (win 1, draw ½)')
    pct_axis(axes[0], -0.02, 1.02)
    axes[1].set_title('Policy entropy of the moves played in training')
    axes[1].set_ylabel('Entropy (nats)')
    axes[1].set_yscale('log')
    for ax in axes:
        steps_axis(ax)
    axes[0].legend(loc='upper left', fontsize=9)
    save(fig, 'entropy-omok9.png')


# ---------------------------------------------------------------- omok9: league and head-to-head


PAST = (1_500_000, 3_000_000, 4_500_000)  # moves at which earlier checkpoints are picked as opponents


def checkpoint_near(run, steps):
    """The saved checkpoint of a run closest to `steps` moves."""
    return min(Path(run['path']).glob('steps-*.pt'), key=lambda p: abs(int(p.stem[6:]) - steps))


def head_to_head(a, b, games=200, openings=4, seed=0):
    return evaluate(make_agent(str(a)), make_agent(str(b)), 'omok9', games, openings, seed=seed)


def league():
    groups = {'ppo': load('algorithms')['ppo'], 'pool': load('league')['pool']}
    if not groups['pool']:
        return
    print('\n### omok9 self-play vs opponent pool (final checkpoint)\n')
    print('| Training opponents | Score vs heuristic | Last-10 jitter | Final vs own past checkpoints (mean of 4) |')
    print('|---|---:|---:|---:|')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    past_scores = {}
    for v, runs in groups.items():
        last = [r['curve'][-1] for r in runs]
        # how the final network does against its own earlier checkpoints
        rows = []
        for r in runs:
            final = Path(f"{r['path']}.pt")
            picks = [checkpoint_near(r, s) for s in PAST] + [checkpoint_near(r, r['curve'][-2]['steps'])]
            rows.append([head_to_head(final, p, games=100).score() for p in picks])
        past_scores[v] = np.array(rows)
        print(f'| {ALGO_LABELS[v]} | {fmt([c["score_vs_heuristic"] for c in last])} '
              f'| {fmt([jitter(r) for r in runs])} | {fmt(past_scores[v].mean(1))} |')
        x, y = series(runs, 'score_vs_heuristic')
        band(axes[0], x, y, ALGO_COLORS[v], 'PPO, pure self-play' if v == 'ppo' else ALGO_LABELS[v])
    width = 0.38
    labels = ['1.5M', '3M', '4.5M', 'previous (−250k)']
    for k, (v, s) in enumerate(past_scores.items()):
        xs = np.arange(4) + (k - 0.5) * width
        axes[1].bar(xs, s.mean(0), width, color=ALGO_COLORS[v], yerr=s.std(0), capsize=3,
                    label='PPO, pure self-play' if v == 'ppo' else ALGO_LABELS[v])
    axes[1].axhline(0.5, color=MUTED, lw=1, ls=':')
    axes[1].set_xticks(np.arange(4), labels)
    axes[1].set_xlabel('Opponent: own checkpoint after … moves')
    axes[1].set_title('Final network vs its own past checkpoints (100 games each)')
    axes[1].set_ylabel('Score of the final network')
    pct_axis(axes[1], 0, 1)
    axes[0].set_title('omok9: score against the heuristic')
    axes[0].set_ylabel('Score (win 1, draw ½)')
    pct_axis(axes[0], -0.02, 1.02)
    steps_axis(axes[0])
    axes[0].legend(loc='upper left', fontsize=9)
    axes[1].legend(loc='upper right', fontsize=9)
    save(fig, 'league-omok9.png')


def versus_dqn():
    """Final PPO networks vs final Stage 2 DQN networks (same seeds), 200 games each, random 4-move openings."""
    ppo = load('algorithms')['ppo']
    dqn = load('omok9', DQN_RUNS)['all']
    print('\n### omok9 head-to-head: PPO vs DQN (Double + Dueling + 3-step), final networks\n')
    print('| Pair | PPO score | PPO win as Black | PPO win as White | Draws |')
    print('|---|---:|---:|---:|---:|')
    for p, d in zip(ppo, dqn):
        r = head_to_head(f"{p['path']}.pt", f"{d['path']}.pt")
        print(f'| {p["path"].name} vs {d["path"].name} | {r.score():.2f} | {r.rate("win", 1):.2f} '
              f'| {r.rate("win", 2):.2f} | {r.rate("draw"):.2f} |')


def policy_heatmap():
    """A PPO (black) vs heuristic game on omok9: the final position, and pi(a | s) and V(s) at one decision."""
    agent = PolicyAgent.load(RUNS / 'algorithms' / 'ppo-seed0.pt')
    game = play_game(make_env('omok9'), agent, HeuristicAgent(0))
    moves = game.get_move_history()
    winner = {0: 'Draw', 1: 'Black (PPO) wins', 2: 'White (heuristic) wins'}[game.get_winner()]

    k = max(0, len(moves) - 5)
    k -= k % 2  # a black (PPO) move
    env = make_env('omok9')
    for m in moves[:k]:
        env.move(m)
    probs, value = agent.evaluate(env)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    draw_board(axes[0], 9, moves)
    axes[0].set_title(f'PPO (black) vs heuristic (white), omok9\n{winner} in {len(moves)} moves', fontsize=10)
    draw_board(axes[1], 9, moves[:k], numbers=True, highlight_last=False)
    top = np.sort(probs)[-3]
    for a in np.flatnonzero(probs > 0.002):
        y, x = divmod(a, 9)
        color = SEQUENTIAL(min(1.0, probs[a] / probs.max()))
        axes[1].add_patch(plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, color=color, alpha=0.85, zorder=1.5))
        if probs[a] >= top:
            axes[1].text(x, y, f'{probs[a]:.2f}', ha='center', va='center', fontsize=7, color='white', zorder=3)
    best = int(np.argmax(probs))
    axes[1].add_patch(plt.Circle(divmod(best, 9)[::-1], 0.5, facecolor='none', edgecolor='#e34948', lw=2, zorder=4))
    axes[1].set_title(f'pi(a | s) before Black\'s move {k + 1}: V(s) = {value:+.2f}\n(darker = more likely, cells with p > 0.002 shaded, '
                      f'top 3 labeled; red ring = chosen move)', fontsize=10)
    save(fig, 'policy-omok9.png')
    print(f'\npolicy figure: {winner} in {len(moves)} moves, decision before move {k + 1}, V = {value:+.2f}, '
          f'top moves {[(int(a), round(float(probs[a]), 3)) for a in np.argsort(probs)[::-1][:3]]}')


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    for name, fn in (('tictactoe', tictactoe), ('algorithms', algorithms), ('updates', updates), ('entropy', entropy_sweep),
                     ('league', league)):
        if (RUNS / name).exists():
            fn()
    if (RUNS / 'algorithms').exists():
        versus_dqn()
        policy_heatmap()
    commits = {json.loads(p.read_text()).get('commit') for p in RUNS.glob('*/*.json')}
    print(f'\ncommits: {commits}')


if __name__ == '__main__':
    main()
