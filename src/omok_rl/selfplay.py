"""Playing games for AlphaZero (Stage 5): self-play, and matches where AlphaZero players search all their games at once.

This module doesn't import torch: it runs in the CPU worker processes, which get their network evaluations from the
GPU process through `omok_rl.inference`.
"""

import numpy as np
from omok.env import PLAYER_BLACK, PLAYER_NONE, PLAYER_WHITE

from omok_rl import inference
from omok_rl.agents import make_agent
from omok_rl.agents.alphazero import AlphaZeroAgent
from omok_rl.arena import Game, random_opening
from omok_rl.envs import make_env
from omok_rl.puct import PUCT, drive, make_search


def match_openings(env_name: str, n_games: int, plies: int, seed: int | None) -> list[list[int]]:
    """The openings of `arena.evaluate` with the same arguments: each random opening is used for one pair of games."""
    rng = np.random.default_rng(seed)
    openings = []
    for i in range(n_games):
        if i % 2 == 0:
            opening = random_opening(env_name, plies, rng) if plies else []
        openings.append(opening)
    return openings


def play_games(env_name: str, openings: list[list[int]], blacks: list, whites: list) -> list:
    """Play one game per opening to the end; agents with `act_batch` choose the moves of all their games at once."""
    envs = [make_env(env_name) for _ in openings]
    for env, opening in zip(envs, openings):
        for pos in opening:
            env.move(pos)
    active = [g for g, env in enumerate(envs) if not env.is_done()]
    while active:
        groups = {}
        for g in active:
            agent = blacks[g] if envs[g].get_player() == PLAYER_BLACK else whites[g]
            groups.setdefault(id(agent), (agent, []))[1].append(g)
        for agent, games in groups.values():
            batch = [envs[g] for g in games]
            moves = agent.act_batch(batch) if hasattr(agent, 'act_batch') else [agent.act(e) for e in batch]
            for env, pos in zip(batch, moves):
                if env.move(pos) == -1:
                    raise RuntimeError(f'{agent.name} played an illegal move {pos} after {env.get_move_history()}')
        active = [g for g in active if not envs[g].is_done()]
    return envs


def play_match(agent_a, agent_b, env_name: str, openings: list[list[int]], first: int = 0) -> list[Game]:
    """Game i (counting from `first`) has A as black if i is even, as in `arena.evaluate`."""
    a_black = [(first + g) % 2 == 0 for g in range(len(openings))]
    blacks = [agent_a if b else agent_b for b in a_black]
    whites = [agent_b if b else agent_a for b in a_black]
    envs = play_games(env_name, openings, blacks, whites)
    return [Game(PLAYER_BLACK if b else PLAYER_WHITE, env.get_winner(), env.get_move_history())
            for b, env in zip(a_black, envs)]


def self_play(search: PUCT, env_name: str, n_games: int, simulations: int, temp_moves: int, rng: np.random.Generator,
              leaves: int = 1, fast_simulations: int = 0, full_prob: float = 1.0, reuse_tree: bool = False) -> dict:
    """Play `n_games` self-play games at once. Returns the training samples and per-game statistics.

    Every move is chosen by a search with root noise (Dirichlet for PUCT, Gumbel for `puct.Gumbel`); the first
    `temp_moves` moves of a PUCT game are drawn in proportion to the visits.
    With `fast_simulations` > 0 (**playout cap randomization**, KataGo), only a share `full_prob` of the moves get the
    full search of `simulations`, with noise, and become training samples; the others get a cheap search of
    `fast_simulations` without noise, which only serves to play the game on. With `reuse_tree`, the subtree of the
    move played is kept as the next root (PUCT only)."""
    return drive(self_play_steps(search, env_name, n_games, simulations, temp_moves, rng, leaves, fast_simulations,
                                 full_prob, reuse_tree), search.evaluator)


def self_play_steps(search: PUCT, env_name: str, n_games: int, simulations: int, temp_moves: int,
                    rng: np.random.Generator, leaves: int = 1, fast_simulations: int = 0, full_prob: float = 1.0,
                    reuse_tree: bool = False):
    """`self_play` as a generator of network requests (see `PUCT.search_steps`)."""
    envs = [make_env(env_name) for _ in range(n_games)]
    size = envs[0].size
    records = [[] for _ in envs]  # (obs, pi, player) per position
    kept = [None] * n_games  # subtrees for tree reuse
    obs, pis, zs, lengths, winners, full_moves = [], [], [], [], [], 0
    active = list(range(n_games))
    while active:
        full = [fast_simulations <= 0 or rng.random() < full_prob for _ in active]
        sims = [simulations if f else fast_simulations for f in full]
        roots = yield from search.search_steps([envs[g] for g in active], sims, noise=full, leaves=leaves,
                                               roots=[kept[g] for g in active] if reuse_tree else None)
        for g, root, f in zip(active, roots, full):
            env = envs[g]
            if f:
                records[g].append((env.get_observation().astype(np.uint8), search.target(root, size), env.get_player()))
                full_moves += 1
            move = search.choose(root, rng, sample=len(env.get_move_history()) < temp_moves)
            kept[g] = root.children[int(np.flatnonzero(root.moves == move)[0])] if reuse_tree else None
            env.move(move)
            if env.is_done():
                winner = env.get_winner()
                for o, pi, player in records[g]:
                    obs.append(o)
                    pis.append(pi)
                    zs.append(0 if winner == PLAYER_NONE else 1 if winner == player else -1)
                lengths.append(len(env.get_move_history()))
                winners.append(winner)
                records[g] = kept[g] = None
        active = [g for g in active if not envs[g].is_done()]
    planes = envs[0].get_observation().shape
    return dict(obs=np.stack(obs) if obs else np.zeros((0, *planes), np.uint8),
                pi=np.stack(pis).astype(np.float16) if pis else np.zeros((0, size * size), np.float16),
                z=np.array(zs, np.int8), lengths=np.array(lengths), winners=np.array(winners),
                full_moves=np.array([full_moves]))


# ---------------------------------------------------------------------------- worker processes


def make_player(spec: dict | str, seed: int):
    """An agent by name, or an AlphaZero player from `dict(net=<server network id>, simulations=..., c_puct=...)`."""
    if isinstance(spec, str):
        return make_agent(spec, seed=seed)
    return AlphaZeroAgent(evaluator=inference.evaluator(spec['net']), simulations=spec['simulations'],
                          c_puct=spec['c_puct'], seed=seed, name=spec.get('name', spec['net']),
                          search=spec.get('search', 'puct'), considered=spec.get('considered', 16))


def search_from_config(config: dict, evaluator=None, seed: int | None = None) -> PUCT:
    keys = ('c_puct', 'dirichlet_alpha', 'dirichlet_eps', 'considered', 'c_visit', 'c_scale')
    return make_search(config.get('search', 'puct'), evaluator, seed=seed, **{k: config[k] for k in keys if k in config})


def self_play_task(net_id: str, config: dict, n_games: int, seed: int) -> dict:
    """Self-play in a worker; `config` is `asdict(AZConfig)`. The games are split into two groups whose searches take
    turns: one group's leaves are on the GPU while the other group is searched."""
    groups = [len(g) for g in np.array_split(np.arange(n_games), inference.SLOTS) if len(g)]
    rng = np.random.default_rng(seed)
    steps = [self_play_steps(search_from_config(config, seed=seed + k), config['env'], n, config['simulations'],
                             config['temp_moves'], rng, config['leaves'], config.get('fast_simulations', 0),
                             config.get('full_prob', 1.0), config.get('reuse_tree', False))
             for k, n in enumerate(groups)]
    results = inference.drive_interleaved(steps, net_id)
    return {k: np.concatenate([r[k] for r in results]) for k in results[0]}


def match_task(spec_a, spec_b, env_name: str, openings: list[list[int]], first: int, seed: int) -> list[Game]:
    np.random.seed(seed)  # the pretrained agents draw from the global random state
    return play_match(make_player(spec_a, seed), make_player(spec_b, seed + 1000), env_name, openings, first)
