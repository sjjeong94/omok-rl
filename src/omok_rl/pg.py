"""Policy gradients by self-play (Stage 3): REINFORCE, A2C and PPO in one loop.

The loop, repeated until `total_steps` moves have been played:

1. Collect: play complete games in `num_envs` parallel boards until at least `batch_moves` learner moves are
   recorded. Moves are sampled from pi(a | s) with illegal moves masked out, for all boards in one batch.
   Every move is recorded from the point of view of the player who made it, as one trajectory per player and
   game; the opponent's replies are part of that player's environment. The reward is the final result for that
   player (+1 win, -1 loss, 0 draw) on their last move.
2. Advantages: for each trajectory, A_t = sum_k (gamma * lambda)^k delta_{t+k} with
   delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) (GAE). REINFORCE uses lambda = 1, i.e. A_t = G_t - V(s_t),
   or A_t = G_t without a baseline.
3. Learn: minimize  policy loss + vf_coef * (V(s) - G)^2 / 2 - ent_coef * entropy, where the policy loss is
   - REINFORCE and A2C: -log pi(a | s) * A, one gradient step on the whole batch;
   - PPO: -min(rho * A, clip(rho, 1 - clip, 1 + clip) * A) with rho = pi(a | s) / pi_old(a | s),
     several epochs of minibatches.

Opponents: in pure self-play both players are the current network and both trajectories are learned from.
With `pool_prob > 0`, that share of games is played against a frozen past version of the network (a snapshot
every `snapshot_every` updates, the last `pool_size` kept; one snapshot, drawn uniformly, per batch), and only
the learner's moves are learned from.

    uv run python -m omok_rl.pg --env omok9 --algo ppo
"""

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from omok.env import PLAYER_BLACK, PLAYER_NONE, PLAYER_WHITE

from omok_rl.agents.policy import PolicyAgent
from omok_rl.dqn import Evaluator
from omok_rl.envs import make_env
from omok_rl.nets import PolicyValueNet
from omok_rl.symmetry import permutations

ALGOS = ('reinforce', 'a2c', 'ppo')


@dataclass
class PGConfig:
    env: str = 'omok9'
    # network
    channels: int = 64
    blocks: int = 3
    # learning
    algo: str = 'ppo'  # 'reinforce', 'a2c' or 'ppo'
    baseline: bool = True  # reinforce only: subtract V(s) from the return (the critic is trained either way)
    gamma: float = 1.0
    gae_lambda: float = 0.95  # a2c and ppo; reinforce always uses Monte Carlo returns (lambda = 1)
    lr: float = 3e-4
    batch_moves: int = 4096  # learner moves (in complete games) per update
    epochs: int = 0  # passes over each batch; 0 = algorithm default (ppo 4, others 1)
    minibatches: int = 0  # gradient steps per epoch; 0 = algorithm default (ppo 4, others 1)
    clip: float = 0.2  # ppo only
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    norm_adv: bool = True  # normalize advantages per batch (never for reinforce without a baseline)
    max_grad_norm: float = 1.0
    augment: bool = True  # every move is chosen on a randomly rotated/reflected board
    # opponents
    pool_prob: float = 0.0  # share of games against a frozen past version instead of the current network
    pool_size: int = 20
    snapshot_every: int = 10  # updates between snapshots added to the pool
    # run
    total_steps: int = 4_000_000  # moves played (both players)
    num_envs: int = 64
    eval_every: int = 100_000
    eval_games: int = 100
    eval_openings: int = 4  # random opening moves per pair of evaluation games (see arena.evaluate)
    seed: int = 0

    def __post_init__(self):
        if self.algo not in ALGOS:
            raise ValueError(f'unknown algo {self.algo!r}, choose from {ALGOS}')

    # The defaults are resolved when used, not stored, so that dataclasses.replace(config, algo=...) picks
    # the new algorithm's default instead of copying the old one's.
    @property
    def n_epochs(self) -> int:
        return self.epochs or (4 if self.algo == 'ppo' else 1)

    @property
    def n_minibatches(self) -> int:
        return self.minibatches or (4 if self.algo == 'ppo' else 1)


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log pi(a | s) over all cells; illegal cells get -inf (probability 0)."""
    return torch.log_softmax(logits.float().masked_fill(~mask, -torch.inf), dim=1)


def entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Entropy of each row of `log_probs`, ignoring the -inf (illegal) cells."""
    return -(log_probs.exp() * log_probs.nan_to_num(neginf=0.0)).sum(1)


def advantages(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    """GAE(gamma, lambda) for one complete trajectory (no bootstrap after the last step)."""
    adv = np.zeros(len(rewards), np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(rewards) else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv


class Trajectory:
    """One player's moves in one game."""

    def __init__(self):
        self.obs, self.mask, self.action, self.log_prob, self.value = [], [], [], [], []

    def __len__(self):
        return len(self.action)


def finish(trajectories: dict[int, Trajectory], winner: int, config: PGConfig) -> list[dict]:
    """Turn the trajectories of a finished game into training samples (one dict of arrays per player)."""
    out = []
    for color, traj in trajectories.items():
        if not len(traj):
            continue
        outcome = 0.0 if winner == PLAYER_NONE else 1.0 if winner == color else -1.0
        rewards = np.zeros(len(traj), np.float32)
        rewards[-1] = outcome
        values = np.array(traj.value, np.float32)
        lam = 1.0 if config.algo == 'reinforce' else config.gae_lambda
        adv = advantages(rewards, values, config.gamma, lam)
        out.append(dict(obs=np.stack(traj.obs), mask=np.stack(traj.mask), action=np.array(traj.action),
                        log_prob=np.array(traj.log_prob, np.float32), adv=adv, ret=adv + values))
    return out


class Collector:
    """Plays games in parallel boards and returns complete trajectories."""

    def __init__(self, config: PGConfig, net: PolicyValueNet, device: torch.device, rng: np.random.Generator):
        self.config, self.net, self.device, self.rng = config, net, device, rng
        self.envs = [make_env(config.env) for _ in range(config.num_envs)]
        self.size = self.envs[0].size
        self.perms = permutations(self.size)[0]
        self.pool: list[PolicyValueNet] = []
        self.steps = self.games = 0
        self.lengths = []

    def add_snapshot(self):
        self.pool.append(copy.deepcopy(self.net).eval().requires_grad_(False))
        self.pool = self.pool[-self.config.pool_size:]

    def new_game(self, i):
        """Reset board i and pick its opponent: None (self-play) or (pool index, learner color)."""
        self.envs[i].reset()
        self.trajs[i] = {PLAYER_BLACK: Trajectory(), PLAYER_WHITE: Trajectory()}
        if self.pool and self.rng.random() < self.config.pool_prob:
            self.opponent[i] = (self.batch_opponent, PLAYER_BLACK if self.rng.random() < 0.5 else PLAYER_WHITE)
        else:
            self.opponent[i] = None

    @torch.inference_mode()
    def sample(self, net, obs, mask):
        """Sample one move per row; returns (actions, log-probs, values) as numpy arrays."""
        logits, value = net(torch.as_tensor(obs, device=self.device))
        log_probs = masked_log_probs(logits, torch.as_tensor(mask, device=self.device))
        action = torch.multinomial(log_probs.exp(), 1)[:, 0]
        return (action.cpu().numpy(), log_probs.gather(1, action[:, None])[:, 0].cpu().numpy(),
                value.float().cpu().numpy())

    def collect(self) -> list[dict]:
        """Play complete games until at least `batch_moves` learner moves are recorded."""
        n = len(self.envs)
        self.trajs, self.opponent = [None] * n, [None] * n
        # one snapshot per batch, so that all pool games share one forward pass per move
        self.batch_opponent = int(self.rng.integers(len(self.pool))) if self.pool else None
        for i in range(n):
            self.new_game(i)
        active = list(range(n))
        samples, recorded = [], 0
        while active:
            obs = np.stack([self.envs[i].get_observation() for i in active])
            mask = np.stack([self.envs[i].get_legal_mask() for i in active])
            if self.config.augment:  # look at every board through a random symmetry
                p = self.perms[self.rng.integers(8, size=len(active))]
                c = obs.shape[1]
                obs = np.take_along_axis(obs.reshape(len(active), c, -1), p[:, None, :], 2).reshape(obs.shape)
                mask = np.take_along_axis(mask, p, 1)
            # who moves on each board: the learner (-1) or a pool network (its index)
            mover = np.array([-1 if self.opponent[i] is None or self.envs[i].get_player() == self.opponent[i][1]
                              else self.opponent[i][0] for i in active])
            action = np.zeros(len(active), np.int64)
            log_prob = np.zeros(len(active), np.float32)
            value = np.zeros(len(active), np.float32)
            for k in np.unique(mover):
                rows = np.flatnonzero(mover == k)
                net = self.net if k == -1 else self.pool[k]
                action[rows], log_prob[rows], value[rows] = self.sample(net, obs[rows], mask[rows])

            still_active = []
            for j, i in enumerate(active):
                env = self.envs[i]
                if mover[j] == -1:
                    traj = self.trajs[i][env.get_player()]
                    traj.obs.append(obs[j])
                    traj.mask.append(mask[j])
                    traj.action.append(action[j])
                    traj.log_prob.append(log_prob[j])
                    traj.value.append(value[j])
                    recorded += 1
                env.move(int(p[j][action[j]]) if self.config.augment else int(action[j]))
                self.steps += 1
                if not env.is_done():
                    still_active.append(i)
                    continue
                self.games += 1
                self.lengths.append(len(env.get_move_history()))
                samples += finish(self.trajs[i], env.get_winner(), self.config)
                if recorded < self.config.batch_moves:
                    self.new_game(i)
                    still_active.append(i)
            active = still_active
        return samples


def ppo_loss(net, config: PGConfig, obs, mask, action, old_log_prob, adv, ret):
    """Total loss and diagnostics for one minibatch."""
    logits, value = net(obs)
    log_probs = masked_log_probs(logits, mask)
    log_prob = log_probs.gather(1, action[:, None])[:, 0]
    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()
    if config.algo == 'ppo':
        policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - config.clip, 1 + config.clip) * adv).mean()
    else:
        policy_loss = -(log_prob * adv).mean()
    value_loss = 0.5 * ((value.float() - ret) ** 2).mean()
    ent = entropy(log_probs).mean()
    loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * ent
    with torch.no_grad():
        stats = torch.stack([policy_loss, value_loss, ent, ((ratio - 1) - log_ratio).mean(),
                             ((ratio - 1).abs() > config.clip).float().mean()])
    return loss, stats


STAT_NAMES = ('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_frac')


def learn(net, optimizer, config: PGConfig, samples: list[dict], device, rng) -> torch.Tensor:
    """Update the network on one batch. Returns the mean of the diagnostics over all gradient steps."""
    batch = {k: torch.as_tensor(np.concatenate([s[k] for s in samples]), device=device) for k in samples[0]}
    adv = batch['adv']
    if config.algo == 'reinforce' and not config.baseline:
        adv = batch['ret']  # the plain Monte Carlo return G_t
    elif config.norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    n = len(adv)
    stats = []
    for _ in range(config.n_epochs):
        order = torch.as_tensor(rng.permutation(n), device=device)
        for idx in order.chunk(config.n_minibatches):
            loss, s = ppo_loss(net, config, batch['obs'][idx], batch['mask'][idx], batch['action'][idx],
                               batch['log_prob'][idx], adv[idx], batch['ret'][idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
            optimizer.step()
            stats.append(s)
    with torch.no_grad():  # how much of the return's variance the value head explains
        var = batch['ret'].var()  # ret - adv is V(s) at collection time
        explained = 1 - batch['adv'].var() / var if var > 0 else torch.zeros((), device=device)
    return torch.cat([torch.stack(stats).mean(0), explained.reshape(1)])


def train(config: PGConfig, out: Path | None = None, device: str | None = None, log=print) -> dict:
    """Train a policy-value network by self-play. Saves `<out>.json` (learning curve), `<out>.pt` (final network)
    and `<out>/steps-*.pt` (the network at every evaluation)."""
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    env = make_env(config.env)
    in_channels, size = env.get_observation().shape[0], env.size
    net = PolicyValueNet(in_channels, config.channels, config.blocks).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    collector = Collector(config, net, device, rng)
    evaluator = Evaluator(config.env, config.eval_games, config.eval_openings, config.seed)
    agent = PolicyAgent(net, device=device)

    updates = learner_moves = 0
    stats, curve = [], []
    start = time.perf_counter()
    next_eval = config.eval_every
    while collector.steps < config.total_steps:
        samples = collector.collect()
        learner_moves += sum(len(s['action']) for s in samples)
        stats.append(learn(net, optimizer, config, samples, device, rng))
        updates += 1
        if config.pool_prob > 0 and updates % config.snapshot_every == 0:
            collector.add_snapshot()

        if collector.steps >= next_eval or collector.steps >= config.total_steps:
            next_eval += config.eval_every
            net.eval()
            metrics = evaluator(agent)
            net.train()
            mean = torch.stack(stats).mean(0).tolist()
            point = ({'steps': collector.steps, 'games': collector.games, 'updates': updates,
                      'learner_moves': learner_moves, 'game_length': float(np.mean(collector.lengths))}
                     | dict(zip(STAT_NAMES + ('explained_variance',), mean))
                     | {'pool': len(collector.pool), 'seconds': time.perf_counter() - start} | metrics)
            stats, collector.lengths = [], []
            curve.append(point)
            if out is not None:  # keep every evaluated network, so evaluation can be redone offline
                save_checkpoint(out / f'steps-{collector.steps:08d}.pt', config, in_channels, size, net)
            log(' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in point.items()))

    resolved = asdict(config) | {'epochs': config.n_epochs, 'minibatches': config.n_minibatches}
    result = {'config': resolved, 'curve': curve, 'seconds': time.perf_counter() - start, 'device': str(device)}
    if out is not None:
        # not out.with_suffix(): run names like 'ent0.03-seed0' contain a dot
        Path(f'{out}.json').write_text(json.dumps(result))
        save_checkpoint(Path(f'{out}.pt'), config, in_channels, size, net)
    return result


def save_checkpoint(path: Path, config: PGConfig, in_channels: int, size: int, net: torch.nn.Module):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'kind': 'policy', 'config': asdict(config), 'in_channels': in_channels, 'size': size,
                'state_dict': net.state_dict()}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(PGConfig):
        if f.type is bool or f.type == 'bool':
            parser.add_argument(f'--{f.name.replace("_", "-")}', action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(f'--{f.name.replace("_", "-")}', type=type(f.default), default=f.default)
    parser.add_argument('--out', type=Path, help='save <out>.json and <out>.pt')
    args = vars(parser.parse_args())
    out = args.pop('out')
    train(PGConfig(**args), out, log=lambda *a: print(*a, flush=True))


if __name__ == '__main__':
    main()
