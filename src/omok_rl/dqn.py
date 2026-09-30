"""Deep Q-learning by self-play (Stage 2).

The loop, repeated until `total_steps` moves have been played:

1. Collect: advance `num_envs` self-play games by one move each. Moves are epsilon-greedy with respect to the
   online network, evaluated for all games in one batch.
2. Store: every move becomes an n-step transition in the replay buffer (see `omok_rl.replay`).
3. Learn: every `train_every` moves, sample a batch, optionally apply a random board symmetry, and take one
   gradient step on the Huber loss between Q(s, a) and the target

       y = ret + sign * Q_target(s', a*),   a* = argmax_a' Q_target(s', a')  (DQN)
                                             a* = argmax_a' Q_online(s', a')  (Double DQN)

   over legal a' only. The target network is a copy of the online network, refreshed every `target_update` steps.

    uv run python -m omok_rl.dqn --env omok6 --double --dueling --n-step 3
"""

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omok.env import PLAYER_BLACK, PLAYER_NONE, PLAYER_WHITE

from omok_rl.agents import HeuristicAgent, RandomAgent
from omok_rl.agents.dqn import DQNAgent
from omok_rl.agents.minimax import MinimaxSolver, all_positions, optimal_move_rate, worst_case
from omok_rl.arena import evaluate
from omok_rl.envs import make_env
from omok_rl.nets import make_qnet
from omok_rl.replay import NStepWriter, ReplayBuffer
from omok_rl.symmetry import permutations


@dataclass
class DQNConfig:
    env: str = 'omok6'
    # network
    arch: str = 'cnn'  # 'cnn' or 'mlp'
    channels: int = 64
    blocks: int = 3
    dueling: bool = False
    # learning
    double: bool = False
    n_step: int = 1
    augment: bool = True  # random board symmetry for every sampled transition
    lr: float = 1e-3
    batch_size: int = 1024
    buffer_size: int = 200_000
    train_every: int = 64  # moves collected per gradient step (each move is sampled ~batch_size/train_every times)
    target_update: int = 250  # gradient steps between target network refreshes
    learning_starts: int = 10_000  # moves collected before learning begins
    # exploration: epsilon decays linearly from eps_start to eps_end over eps_decay of the run
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.5
    # run
    total_steps: int = 500_000  # self-play moves (both players)
    num_envs: int = 64
    eval_every: int = 25_000
    eval_games: int = 100
    eval_openings: int = 2  # random opening moves per pair of evaluation games (see arena.evaluate)
    seed: int = 0


def augment_batch(obs, action, next_obs, next_mask, perms, inverse):
    """Apply an independent random symmetry to every transition in the batch (on the device)."""
    b, c, s, _ = obs.shape
    codes = torch.randint(0, 8, (b,), device=obs.device)
    p = perms[codes]  # (B, n): transformed[i] = original[p[i]]
    gather = lambda x: x.flatten(2).gather(2, p[:, None, :].expand(-1, x.shape[1], -1)).view_as(x)
    return gather(obs), inverse[codes].gather(1, action[:, None])[:, 0], gather(next_obs), next_mask.gather(1, p)


class Evaluator:
    def __init__(self, env_name: str, games: int, openings: int, seed: int):
        self.env_name, self.games, self.openings, self.seed = env_name, games, openings, seed
        if env_name == 'tictactoe':  # exact metrics, as in Stage 1
            solver = MinimaxSolver()
            self.positions = all_positions(make_env('tictactoe'))
            self.optimal = [set(solver.optimal_actions(p)) for p in self.positions]

    def __call__(self, agent: DQNAgent) -> dict:
        metrics = {}
        for name, opponent in (('random', RandomAgent(self.seed + 1000)), ('heuristic', HeuristicAgent(self.seed + 1000))):
            r = evaluate(agent, opponent, self.env_name, self.games, self.openings, seed=self.seed)
            metrics[f'score_vs_{name}'] = r.score()
            metrics[f'win_black_vs_{name}'] = r.rate('win', PLAYER_BLACK)
            metrics[f'win_white_vs_{name}'] = r.rate('win', PLAYER_WHITE)
        if self.env_name == 'tictactoe':
            env = make_env('tictactoe')
            metrics['optimal_move_rate'] = optimal_move_rate(agent.greedy_actions, self.positions, self.optimal)
            metrics['worst_case_black'] = worst_case(agent.greedy_actions, env, PLAYER_BLACK)
            metrics['worst_case_white'] = worst_case(agent.greedy_actions, env, PLAYER_WHITE)
        return metrics


def train(config: DQNConfig, out: Path | None = None, device: str | None = None, log=print) -> dict:
    """Train a Q-network by self-play. Saves `<out>.json` (learning curve), `<out>.pt` (final network)
    and `<out>/steps-*.pt` (the network at every evaluation)."""
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = [make_env(config.env) for _ in range(config.num_envs)]
    obs_shape = envs[0].get_observation().shape
    in_channels, size = obs_shape[0], envs[0].size
    online = make_qnet(config.arch, in_channels, size, config.dueling, config.channels, config.blocks).to(device)
    target = copy.deepcopy(online).requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=config.lr, fused=device.type == 'cuda')
    buffer = ReplayBuffer(config.buffer_size, obs_shape, seed=config.seed)
    writers = [NStepWriter(buffer, config.n_step) for _ in envs]
    perms, inverse = (torch.as_tensor(x, device=device) for x in permutations(size))
    evaluator = Evaluator(config.env, config.eval_games, config.eval_openings, config.seed)
    agent = DQNAgent(online, device=device)

    steps = games = updates = 0
    losses, q_means, curve = [], [], []
    start = time.perf_counter()
    next_eval = config.eval_every
    obs = np.stack([e.get_observation() for e in envs])
    masks = np.stack([e.get_legal_mask() for e in envs])
    while steps < config.total_steps:
        # 1. collect one move in every game
        epsilon = max(config.eps_end, config.eps_start - (config.eps_start - config.eps_end)
                      * steps / (config.eps_decay * config.total_steps))
        with torch.inference_mode():
            q = online(torch.as_tensor(obs, device=device)).float().cpu().numpy()
        q[~masks] = -np.inf
        obs_before = obs.copy()
        for i, env in enumerate(envs):
            if rng.random() < epsilon:
                action = int(rng.choice(np.flatnonzero(masks[i])))
            else:
                action = int(np.argmax(q[i]))
            mover = env.get_player()
            env.move(action)
            done = env.is_done()
            reward = 1.0 if done and env.get_winner() == mover else 0.0
            obs[i], masks[i] = env.get_observation(), env.get_legal_mask()  # the next position, reused below
            writers[i].step(obs_before[i], action, obs[i], masks[i], done, reward)
            if done:
                games += 1
                env.reset()
                obs[i], masks[i] = env.get_observation(), env.get_legal_mask()
        steps += config.num_envs

        # 2. learn
        if len(buffer) >= config.learning_starts:
            while updates < (steps - config.learning_starts) // config.train_every:
                loss, q_mean = learn_step(buffer, online, target, optimizer, config, device, perms, inverse)
                losses.append(loss)
                q_means.append(q_mean)
                updates += 1
                if updates % config.target_update == 0:
                    target.load_state_dict(online.state_dict())

        # 3. evaluate
        if steps >= next_eval or steps >= config.total_steps:
            next_eval += config.eval_every
            online.eval()
            metrics = evaluator(agent)
            online.train()
            point = {'steps': steps, 'games': games, 'updates': updates, 'epsilon': epsilon,
                     'loss': torch.stack(losses).mean().item() if losses else None,
                     'q_mean': torch.stack(q_means).mean().item() if q_means else None,
                     'seconds': time.perf_counter() - start} | metrics
            losses, q_means = [], []
            curve.append(point)
            if out is not None:  # keep every evaluated network, so evaluation can be redone offline
                save_checkpoint(out / f'steps-{steps:08d}.pt', config, in_channels, size, online)
            log(' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in point.items()))

    result = {'config': asdict(config), 'curve': curve, 'seconds': time.perf_counter() - start, 'device': str(device)}
    if out is not None:
        out.with_suffix('.json').write_text(json.dumps(result))
        save_checkpoint(out.with_suffix('.pt'), config, in_channels, size, online)
    return result


def save_checkpoint(path: Path, config: DQNConfig, in_channels: int, size: int, net: torch.nn.Module):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'config': asdict(config), 'in_channels': in_channels, 'size': size, 'state_dict': net.state_dict()}, path)


def td_loss(online, target, double: bool, obs, action, ret, next_obs, next_mask, sign):
    """Huber loss between Q(s, a) and the (Double) DQN target, and the mean Q of the batch."""
    with torch.no_grad():
        q_next = target(next_obs).masked_fill(~next_mask, -torch.inf)
        if double:
            a_star = online(next_obs).masked_fill(~next_mask, -torch.inf).argmax(1, keepdim=True)
            v_next = q_next.gather(1, a_star)[:, 0]
        else:
            v_next = q_next.max(1).values
        v_next = torch.where(sign != 0, v_next, 0.0)  # finished games have no legal moves (and -inf)
        y = ret + sign * v_next
    q = online(obs).gather(1, action[:, None])[:, 0]
    return F.smooth_l1_loss(q, y), q.detach().mean()


def learn_step(buffer, online, target, optimizer, config, device, perms, inverse):
    """One gradient step. Returns the loss and mean Q as device tensors (no GPU sync)."""
    b = buffer.sample(config.batch_size)
    tensors = [torch.as_tensor(x, device=device)
               for x in (b.obs, b.action, b.ret, b.next_obs, b.next_mask, b.sign)]
    obs, action, ret, next_obs, next_mask, sign = tensors
    if config.augment:
        obs, action, next_obs, next_mask = augment_batch(obs, action, next_obs, next_mask, perms, inverse)
    loss, q_mean = td_loss(online, target, config.double, obs, action, ret, next_obs, next_mask, sign)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return loss.detach(), q_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(DQNConfig):
        if f.type is bool or f.type == 'bool':
            parser.add_argument(f'--{f.name.replace("_", "-")}', action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(f'--{f.name.replace("_", "-")}', type=type(f.default), default=f.default)
    parser.add_argument('--out', type=Path, help='save <out>.json and <out>.pt')
    args = vars(parser.parse_args())
    out = args.pop('out')
    train(DQNConfig(**args), out)


if __name__ == '__main__':
    main()
