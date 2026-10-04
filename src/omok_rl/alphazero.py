"""AlphaZero (Stage 5): self-play with PUCT search, and a network trained to predict the search's result.

One **generation** of the loop:

1. Self-play: `games_per_gen` games, split among `workers` processes that each play their games all at once (one tree
   per game, leaves of all trees evaluated in one GPU batch, see `omok_rl.puct`). Every move is chosen by a search of
   `simulations` simulations with Dirichlet noise at the root. For the first `temp_moves` moves of a game the move is
   drawn in proportion to the visit counts, afterwards the most visited move is played. Each position is recorded with
   the visit distribution pi as policy target; when the game ends, every position gets the result z (+1 / 0 / -1)
   for the player to move there.
2. Replay buffer: the newest `buffer_size` positions.
3. Training: minibatches from the buffer, each position shown through a random one of the 8 symmetries, minimizing

       (z - v)^2 - sum_a pi(a) log p(a)  (+ weight decay)

   The number of steps is set so that each position is used `reuse` times on average while it is in the buffer.
4. Optionally (`gating`, as in AlphaGo Zero): the new network plays the best one so far, and only replaces it as the
   self-play network if it scores at least `gate_threshold`. AlphaZero dropped this step and always uses the newest.

Every `eval_every` generations, the network plays the heuristic with search (`eval_simulations`) and without
(the raw policy), and is saved to `<out>/gen-XXXX.pt`.

    uv run python -m omok_rl.alphazero --env omok9 --out runs/stage5/omok9/main
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from omok.env import PLAYER_BLACK, PLAYER_NONE, PLAYER_WHITE

from omok_rl import inference, selfplay
from omok_rl.arena import MatchResult
from omok_rl.envs import make_env
from omok_rl.nets import AlphaZeroNet, NetEvaluator, permute_planes
from omok_rl.selfplay import match_openings
from omok_rl.symmetry import permutations


@dataclass
class AZConfig:
    env: str = 'omok9'
    init: str = ''  # warm start from this checkpoint (any board size; extra input planes start at zero)
    # network
    channels: int = 64
    blocks: int = 6
    # search
    simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 10.0  # total concentration, split over the legal moves: alpha = 10 / legal moves
    dirichlet_eps: float = 0.25
    temp_moves: int = 4  # moves sampled in proportion to the visits at the start of each game
    leaves: int = 1  # leaves per tree per network batch in self-play (virtual loss when > 1)
    symmetries: bool = True  # evaluate each leaf through a random symmetry
    # self-play
    games_per_gen: int = 480
    workers: int = 12  # CPU search processes (the GPU work is done by the main process)
    # learning
    buffer_size: int = 100_000  # positions
    batch_size: int = 512
    reuse: float = 10.0  # average number of times each position is trained on
    lr: float = 1e-3
    weight_decay: float = 1e-4
    augment: bool = True  # train on a random symmetry of each position
    # gating (AlphaGo Zero)
    gating: bool = False
    gate_games: int = 100
    gate_threshold: float = 0.55
    # run
    generations: int = 60
    eval_every: int = 5
    eval_games: int = 100
    eval_openings: int = 4
    eval_simulations: int = 200
    seed: int = 0


# ---------------------------------------------------------------------------- worker processes


class Workers:
    """CPU processes that play batched games, with their networks evaluated on the GPU by the main process.

    Register networks in `evaluators` (id -> `puct.NetEvaluator`); players refer to them by id."""

    def __init__(self, n: int, max_batch: int = 1024):
        self.n = n
        self.server = inference.Server(n, max_batch)
        self.evaluators = self.server.evaluators

    def self_play(self, net_id: str, config: AZConfig, seed: int) -> list[dict]:
        per_worker = [len(games) for games in np.array_split(np.arange(config.games_per_gen), self.n)]
        return self.server.run(selfplay.self_play_task, [(net_id, asdict(config), n, seed * 1000 + k)
                                                 for k, n in enumerate(per_worker) if n])

    def match(self, spec_a, spec_b, env_name: str, n_games: int, opening_plies: int, seed: int,
              names: tuple[str, str] = ('a', 'b')) -> MatchResult:
        """Like `arena.evaluate` (same openings and colors), with the games split among the workers."""
        return self.matches([(spec_a, spec_b, names)], env_name, n_games, opening_plies, seed)[0]

    def matches(self, pairings: list[tuple], env_name: str, n_games: int, opening_plies: int, seed: int,
                games_per_task: int | None = None) -> list[MatchResult]:
        """Several matches `(spec_a, spec_b, (name_a, name_b))` at once, all from the same openings."""
        openings = match_openings(env_name, n_games, opening_plies, seed)
        per_task = games_per_task or math.ceil(n_games * len(pairings) / self.n)
        chunks = [c for c in np.array_split(np.arange(n_games), math.ceil(n_games / per_task)) if len(c)]
        tasks, owners = [], []
        for p, (spec_a, spec_b, _) in enumerate(pairings):
            for k, c in enumerate(chunks):
                tasks.append((spec_a, spec_b, env_name, [openings[g] for g in c], int(c[0]), seed + k))
                owners.append(p)
        results = [MatchResult(*names, env_name) for _, _, names in pairings]
        for p, games in zip(owners, self.server.run(selfplay.match_task, tasks)):
            results[p].games += games
        return results

    def close(self):
        self.server.close()


def player(net_id: str, config: AZConfig, simulations: int | None = None) -> dict:
    """Spec of an AlphaZero player for `Workers.match`."""
    return dict(net=net_id, simulations=config.eval_simulations if simulations is None else simulations,
                c_puct=config.c_puct)


# ---------------------------------------------------------------------------- learning


class ReplayBuffer:
    """The newest `capacity` positions: observation planes (0/1, stored as uint8), pi and z."""

    def __init__(self, capacity: int, obs_shape: tuple, seed: int):
        self.obs = np.zeros((capacity, *obs_shape), np.uint8)
        self.pi = np.zeros((capacity, obs_shape[1] * obs_shape[2]), np.float16)
        self.z = np.zeros(capacity, np.int8)
        self.capacity, self.size, self.next = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs: np.ndarray, pi: np.ndarray, z: np.ndarray):
        for start in range(0, len(z), self.capacity):  # in pieces, in case one batch is larger than the buffer
            o, p, r = obs[start:start + self.capacity], pi[start:start + self.capacity], z[start:start + self.capacity]
            idx = (self.next + np.arange(len(r))) % self.capacity
            self.obs[idx], self.pi[idx], self.z[idx] = o, p, r
            self.next = (self.next + len(r)) % self.capacity
            self.size = min(self.size + len(r), self.capacity)

    def sample(self, n: int):
        idx = self.rng.integers(self.size, size=n)
        return self.obs[idx], self.pi[idx], self.z[idx]

    def state(self) -> dict:
        return dict(obs=self.obs[:self.size], pi=self.pi[:self.size], z=self.z[:self.size], next=self.next)

    def load(self, state: dict):
        n = len(state['z'])
        self.obs[:n], self.pi[:n], self.z[:n] = state['obs'], state['pi'], state['z']
        self.size, self.next = n, state['next']


def legal_from_obs(obs: torch.Tensor) -> torch.Tensor:
    """Legal moves of a batch of observations, (B, size*size): empty cells that are not forbidden (plane 4, Renju)."""
    occupied = obs[:, 0] + obs[:, 1] + (obs[:, 4] if obs.shape[1] > 4 else 0)
    return (occupied == 0).flatten(1)


def learn(net, optimizer, buffer: ReplayBuffer, config: AZConfig, steps: int, device, perms) -> dict:
    """`steps` gradient steps on minibatches from the buffer. Returns the mean losses."""
    net.train()
    stats = []
    for _ in range(steps):
        obs, pi, z = buffer.sample(config.batch_size)
        obs = torch.as_tensor(obs, device=device).float()
        pi = torch.as_tensor(pi, device=device).float()
        z = torch.as_tensor(z, device=device).float()
        if config.augment:  # one random symmetry per position, applied to the planes and to pi alike
            p = perms[torch.as_tensor(buffer.rng.integers(8, size=len(z)), device=device)]
            obs = permute_planes(obs, p)
            pi = pi.gather(1, p)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits, value = net(obs)
        mask = legal_from_obs(obs)
        log_p = torch.log_softmax(logits.float().masked_fill(~mask, -torch.inf), 1).masked_fill(~mask, 0)
        policy_loss = -(pi * log_p).sum(1).mean()
        value_loss = ((value.float() - z) ** 2).mean()
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():  # the policy loss minus the target's own entropy: KL(pi || p) >= 0
            target_entropy = -(pi * pi.clamp_min(1e-12).log()).sum(1).mean()
            stats.append(torch.stack([policy_loss, value_loss, policy_loss - target_entropy]))
    net.eval()
    return dict(zip(('policy_loss', 'value_loss', 'policy_kl'), torch.stack(stats).mean(0).tolist())) if stats else {}


# ---------------------------------------------------------------------------- the loop


def save_checkpoint(path: Path, config: AZConfig, net: AlphaZeroNet, size: int, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'kind': 'alphazero', 'config': asdict(config), 'in_channels': net.in_channels, 'size': size,
                'state_dict': net.state_dict()} | extra, path)


def train(config: AZConfig, out: Path, device: str | None = None, log=print) -> dict:
    """Run the AlphaZero loop. Saves `<out>.json` (learning curve), `<out>.pt` (final network), `<out>/gen-*.pt`
    (the network at every evaluation) and `<out>/resume.pt`, from which an interrupted run continues."""
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.manual_seed(config.seed)
    env = make_env(config.env)
    obs_shape, size = env.get_observation().shape, env.size
    net = AlphaZeroNet(obs_shape[0], config.channels, config.blocks).to(device)
    if config.init:
        net.load_transfer(torch.load(config.init, map_location=device, weights_only=False)['state_dict'])
    net.eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    buffer = ReplayBuffer(config.buffer_size, obs_shape, config.seed)
    perms = torch.as_tensor(permutations(size)[0], device=device)
    best = AlphaZeroNet(obs_shape[0], config.channels, config.blocks).to(device).eval()  # self-play network (gating)
    best.load_state_dict(net.state_dict())

    gen, curve, seconds, games, positions = 0, [], 0.0, 0, 0
    resume = out / 'resume.pt'
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        net.load_state_dict(state['net'])
        best.load_state_dict(state['best'])
        optimizer.load_state_dict(state['optimizer'])
        buffer.load(state['buffer'])
        buffer.rng.bit_generator.state = state['rng']
        gen, curve, seconds, games, positions = (state[k] for k in ('gen', 'curve', 'seconds', 'games', 'positions'))
        log(f'resumed after generation {gen}')

    workers = Workers(config.workers)
    workers.evaluators['net'] = NetEvaluator(net, device, size, config.symmetries, config.seed)
    workers.evaluators['best'] = NetEvaluator(best, device, size, config.symmetries, config.seed + 1)
    try:
        while gen < config.generations:
            gen += 1
            start = time.perf_counter()
            results = workers.self_play('best' if config.gating else 'net', config, config.seed * 10_000 + gen)
            for r in results:
                buffer.add(r['obs'], r['pi'], r['z'])
            new = sum(len(r['z']) for r in results)
            lengths = np.concatenate([r['lengths'] for r in results])
            winners = np.concatenate([r['winners'] for r in results])
            games += len(lengths)
            positions += new
            self_play_seconds = time.perf_counter() - start

            steps = math.ceil(new * config.reuse / config.batch_size)
            stats = learn(net, optimizer, buffer, config, steps, device, perms)
            point = {'gen': gen, 'games': games, 'positions': positions, 'buffer': buffer.size, 'steps': steps,
                     'game_length': float(lengths.mean()), 'black_wins': float(np.mean(winners == PLAYER_BLACK)),
                     'draws': float(np.mean(winners == PLAYER_NONE))} | stats
            point['self_play_seconds'] = self_play_seconds
            point['train_seconds'] = time.perf_counter() - start - self_play_seconds

            if config.gating:
                r = workers.match(player('net', config), player('best', config), config.env, config.gate_games,
                                  config.eval_openings, seed=gen)
                point['gate_score'] = r.score()
                point['accepted'] = r.score() >= config.gate_threshold
                if point['accepted']:
                    best.load_state_dict(net.state_dict())

            if gen % config.eval_every == 0 or gen == config.generations:
                net_id = 'best' if config.gating else 'net'
                for key, sims in (('search', config.eval_simulations), ('raw', 0)):
                    r = workers.match(player(net_id, config, sims), 'heuristic', config.env, config.eval_games,
                                      config.eval_openings, seed=config.seed)
                    point[f'score_vs_heuristic_{key}'] = r.score()
                    point[f'win_black_vs_heuristic_{key}'] = r.rate('win', PLAYER_BLACK)
                    point[f'win_white_vs_heuristic_{key}'] = r.rate('win', PLAYER_WHITE)
                save_checkpoint(out / f'gen-{gen:04d}.pt', config, best if config.gating else net, size, gen=gen)
            seconds += time.perf_counter() - start
            point['seconds'] = seconds
            curve.append(point)
            log(' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in point.items()))

            if gen % config.eval_every == 0:
                torch.save({'net': net.state_dict(), 'best': best.state_dict(), 'optimizer': optimizer.state_dict(),
                            'buffer': buffer.state(), 'rng': buffer.rng.bit_generator.state, 'gen': gen, 'curve': curve,
                            'seconds': seconds, 'games': games, 'positions': positions}, resume)
    finally:
        workers.close()

    result = {'config': asdict(config), 'curve': curve, 'seconds': seconds, 'device': str(device)}
    Path(f'{out}.json').write_text(json.dumps(result))
    save_checkpoint(Path(f'{out}.pt'), config, best if config.gating else net, size, gen=gen)
    resume.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(AZConfig):
        if f.type is bool or f.type == 'bool':
            parser.add_argument(f'--{f.name.replace("_", "-")}', action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(f'--{f.name.replace("_", "-")}', type=type(f.default), default=f.default)
    parser.add_argument('--out', type=Path, required=True, help='save <out>.json, <out>.pt and <out>/gen-*.pt')
    args = vars(parser.parse_args())
    out = args.pop('out')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    train(AZConfig(**args), out, log=lambda *a: print(*a, flush=True))


if __name__ == '__main__':
    main()
