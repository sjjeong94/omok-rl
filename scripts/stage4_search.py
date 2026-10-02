"""Stage 4 experiments: exact alpha-beta solving, and MCTS / alpha-beta players on tic-tac-toe and omok9.

Each job writes runs/stage4/<experiment>/<name>.json and is skipped if that file exists. Jobs run on the CPU,
several at a time (the policy network of Stage 3 is evaluated on the CPU too). Summarize with scripts/plot_stage4.py.

    uv run python scripts/stage4_search.py [--only omok9] [--workers 8]
"""

import argparse
import json
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

import numpy as np
from omok.env import BoardGame

OUT = Path('runs/stage4')
PPO = 'runs/stage3/entropy/ent0.03-seed0.pt'  # Stage 3's best policy network (PPO, ent_coef 0.03)
DQN = 'runs/stage2/omok9/all-seed0.pt'  # Stage 2's best omok9 Q-network (Double + Dueling + 3-step)

# ---------------------------------------------------------------- exact solving

BOARDS = {'tictactoe': (3, 3), '4x4k3': (4, 3), '4x4k4': (4, 4), '5x5k4': (5, 4), 'omok6': (6, 4)}
ABLATIONS = {  # each speed-up switched off on its own, plus plain negamax with nothing
    'all': {}, 'no-alphabeta': dict(alphabeta=False), 'no-table': dict(table=False), 'no-symmetry': dict(symmetry=False),
    'no-ordering': dict(ordering=False), 'no-threats': dict(threats=False),
    'plain': dict(alphabeta=False, table=False, symmetry=False, ordering=False, threats=False),
}
MAX_NODES = 10_000_000  # ~5 minutes and up to ~2 GB; a search that needs more is reported as unfinished


def mnk(size: int, win: int) -> BoardGame:
    return type(f'Game{size}x{size}k{win}', (BoardGame,), {'size': size, 'win': win})()


def solve(board: str, ablation: str) -> dict:
    from omok_rl.agents.alphabeta import SearchLimit, Solver, SolverOptions

    solver = Solver(SolverOptions(**ABLATIONS[ablation]), max_nodes=MAX_NODES)
    start = time.perf_counter()
    try:
        value = solver.value(mnk(*BOARDS[board]))
    except SearchLimit:
        value = None
    return dict(board=board, ablation=ablation, value=value, nodes=solver.nodes, table=len(solver.table),
                seconds=time.perf_counter() - start)


def first_moves(board: str) -> dict:
    """Exact value of every first move (one search per symmetry class, each with its own table and node limit)."""
    from omok_rl.agents.alphabeta import SearchLimit, Solver
    from omok_rl.symmetry import canonical

    size, win = BOARDS[board]
    classes = {}
    for a in range(size * size):
        b = np.zeros(size * size, np.uint8)
        b[a] = 1
        classes.setdefault(canonical(b, size)[0], []).append(a)
    values = {}
    for cells in classes.values():
        env = mnk(size, win)
        env.move(cells[0])
        solver, start = Solver(max_nodes=MAX_NODES), time.perf_counter()
        try:
            value = -solver.value(env)  # for Black, who just moved
        except SearchLimit:
            value = None
        for a in cells:
            values[a] = dict(value=value, nodes=solver.nodes, seconds=time.perf_counter() - start)
    return dict(board=board, moves=values)


# ---------------------------------------------------------------- MCTS on tic-tac-toe (exact metric)

TICTACTOE_SIMS = (10, 20, 50, 100, 200, 500, 1000)


def tictactoe(simulations: int, seed: int) -> dict:
    from omok_rl.agents.mcts import MCTSAgent
    from omok_rl.agents.minimax import MinimaxSolver, all_positions
    from omok_rl.envs import make_env

    positions = all_positions(make_env('tictactoe'))
    solver = MinimaxSolver()
    agent = MCTSAgent(simulations, seed=seed)
    ok = [agent.act(p) in solver.optimal_actions(p.clone()) for p in positions]
    return dict(simulations=simulations, seed=seed, optimal_move_rate=float(np.mean(ok)), positions=len(ok),
                ms_per_move=1e3 * agent.seconds / agent.searches)


# ---------------------------------------------------------------- blocking a single threat on omok9

# Black has 36, 0, 1, 80; White has 37-40 in row 4 and wins at 41 unless Black blocks there.
BLOCK_MOVES = [36, 37, 0, 38, 1, 39, 80, 40]
BLOCK_CELL = 41
BLOCK_SIMS = (100, 200, 400, 800, 1600, 3200, 6400, 12800)
BLOCK_SEEDS = 20


def block(variant: str, simulations: int) -> dict:
    from omok_rl.envs import make_env

    env = make_env('omok9')
    for m in BLOCK_MOVES:
        env.move(m)
    blocks, shares = 0, []
    for seed in range(BLOCK_SEEDS):
        agent = make_mcts(variant, simulations, seed)
        blocks += agent.act(env) == BLOCK_CELL
        moves, share = agent.root_policy()
        shares.append(float(share[moves == BLOCK_CELL][0]))
    return dict(variant=variant, simulations=simulations, block_rate=blocks / BLOCK_SEEDS, visit_share=shares,
                legal_moves=int(env.get_legal_mask().sum()))


# ---------------------------------------------------------------- players on omok9

VARIANTS = {  # name: (evaluation, prior)
    'uct-random': ('random', None),  # pure MCTS
    'uct-heuristic': ('heuristic', None),
    'hprior-random': ('random', 'heuristic'),
    'hprior-heuristic': ('heuristic', 'heuristic'),
    'nprior-random': ('random', 'net'),
    'nprior-value': ('value', 'net'),  # AlphaZero-style search with Stage 3's network, no training
    'hprior-value': ('value', 'heuristic'),
}
OMOK9_SIMS = (100, 200, 400, 800, 1600, 3200)
ALPHABETA_DEPTHS = (1, 2, 3, 4, 5)
GAMES, OPENINGS = 100, 4  # as in Stages 2-3: 50 random 4-move openings, each played once per color


def make_mcts(variant: str, simulations: int, seed: int):
    from omok_rl.agents.mcts import MCTSAgent
    from omok_rl.agents.policy import PolicyAgent

    evaluation, prior = VARIANTS[variant]
    net = PolicyAgent.load(PPO) if prior == 'net' or evaluation == 'value' else None
    return MCTSAgent(simulations, evaluation=evaluation, prior=prior, net=net, seed=seed, name=f'{variant}-{simulations}')


class Timed:
    """Wraps an agent and measures the time it spends choosing moves."""

    def __init__(self, agent):
        self.agent, self.name, self.seconds, self.moves = agent, agent.name, 0.0, 0

    def act(self, env):
        start = time.perf_counter()
        move = self.agent.act(env)
        self.seconds += time.perf_counter() - start
        self.moves += 1
        return move


def make_player(spec: str, seed: int = 0):
    """'mcts/<variant>/<simulations>', 'alphabeta/<depth>', 'ppo', 'dqn' or 'heuristic'."""
    from omok_rl.agents import AlphaBetaAgent, HeuristicAgent, make_agent

    kind, *args = spec.split('/')
    if kind == 'mcts':
        return make_mcts(args[0], int(args[1]), seed)
    if kind == 'alphabeta':
        return AlphaBetaAgent(depth=int(args[0]), seed=seed)
    if kind == 'heuristic':
        return HeuristicAgent(seed)
    return make_agent({'ppo': PPO, 'dqn': DQN}[kind])


def match(a: str, b: str) -> dict:
    """`GAMES` games of player a against player b on omok9, from random openings (seed 0, as Stages 2-3)."""
    from omok.env import PLAYER_BLACK, PLAYER_WHITE

    from omok_rl.arena import evaluate

    agent_a, agent_b = Timed(make_player(a, seed=0)), Timed(make_player(b, seed=1000))
    r = evaluate(agent_a, agent_b, 'omok9', GAMES, OPENINGS, seed=0)
    return dict(a=a, b=b, score=r.score(), win_black=r.rate('win', PLAYER_BLACK), win_white=r.rate('win', PLAYER_WHITE),
                draws=r.rate('draw'), game_length=float(np.mean([len(g.moves) for g in r.games])),
                seconds_per_move_a=agent_a.seconds / agent_a.moves, seconds_per_move_b=agent_b.seconds / agent_b.moves,
                games=[dict(a_color=g.a_color, winner=g.winner, moves=g.moves) for g in r.games])


# ---------------------------------------------------------------- job list


def jobs():
    out = []
    for board in BOARDS:
        for ablation in ABLATIONS:
            out.append(('solve', f'{board}-{ablation}', solve, (board, ablation)))
    out.append(('solve', 'omok6-first-moves', first_moves, ('omok6',)))
    for sims in TICTACTOE_SIMS:
        for seed in range(3):
            out.append(('tictactoe', f'uct-random-{sims}-seed{seed}', tictactoe, (sims, seed)))
    for variant in ('uct-random', 'uct-heuristic', 'hprior-random'):
        for sims in BLOCK_SIMS:
            out.append(('block', f'{variant}-{sims}', block, (variant, sims)))
    for variant in VARIANTS:
        for sims in OMOK9_SIMS:
            out.append(('omok9', f'{variant}-{sims}', match, (f'mcts/{variant}/{sims}', 'heuristic')))
    for depth in ALPHABETA_DEPTHS:
        out.append(('omok9', f'alphabeta-{depth}', match, (f'alphabeta/{depth}', 'heuristic')))
    for a, b in (('mcts/hprior-heuristic/1600', 'alphabeta/4'), ('mcts/hprior-heuristic/1600', 'ppo'),
                 ('mcts/hprior-heuristic/1600', 'dqn'), ('mcts/nprior-value/1600', 'ppo'), ('mcts/nprior-value/1600', 'dqn'),
                 ('alphabeta/4', 'ppo'), ('alphabeta/4', 'dqn'), ('dqn', 'ppo')):
        out.append(('versus', f'{a.replace("/", "_")}--{b.replace("/", "_")}', match, (a, b)))
    return out


def git_commit() -> str:
    git = lambda *args: subprocess.run(['git', *args], capture_output=True, text=True).stdout.strip()
    return git('rev-parse', '--short', 'HEAD') + ('-dirty' if git('status', '--porcelain') else '')


def run(experiment, name, fn, args, commit):
    import torch

    torch.set_num_threads(1)
    path = OUT / experiment / f'{name}.json'
    if path.exists():
        return f'{experiment}/{name}: exists, skipped'
    start = time.perf_counter()
    result = fn(*args) | {'commit': commit, 'wall_seconds': time.perf_counter() - start}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    summary = {k: v for k, v in result.items() if isinstance(v, (int, float, str)) and k not in ('commit',)}
    return f'{experiment}/{name} ({result["wall_seconds"] / 60:.1f} min): {summary}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--only', help='run only this experiment (solve, tictactoe, block, omok9, versus)')
    args = parser.parse_args()

    todo = [j for j in jobs() if args.only in (None, j[0])]
    commit = git_commit()
    with mp.get_context('spawn').Pool(args.workers, maxtasksperchild=1) as pool:  # fresh process per job: solver tables can be large
        results = [pool.apply_async(run, (*job, commit)) for job in todo]
        for r in results:
            print(r.get(), flush=True)


if __name__ == '__main__':
    main()
