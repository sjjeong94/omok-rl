import subprocess
import sys

import numpy as np
import torch
from torch import nn

from omok_rl.agents import make_agent
from omok_rl.agents.alphazero import AlphaZeroAgent
from omok_rl.alphazero import AZConfig, ReplayBuffer, legal_from_obs, save_checkpoint
from omok_rl.arena import evaluate
from omok_rl.envs import make_env
from omok_rl.nets import AlphaZeroNet, NetEvaluator, permute_planes
from omok_rl.puct import PUCT
from omok_rl.selfplay import match_openings, play_match, self_play
from omok_rl.symmetry import permutations


class Uniform(nn.Module):
    """Equal logits everywhere and value 0: the search alone must find the tactics."""

    def forward(self, x):
        return torch.zeros(len(x), x.shape[2] * x.shape[3]), torch.zeros(len(x))


class NearStones(nn.Module):
    """Logits that grow near the stones (a 3x3 blur, which commutes with every symmetry), to check that symmetries are undone."""

    def forward(self, x):
        h = nn.functional.avg_pool2d(x[:, 3:4] * 9 + x[:, 0:1] * 4 + x[:, 1:2], 3, 1, 1, count_include_pad=False)
        return h.flatten(1), x[:, 0].mean((1, 2))


def play(env, moves):
    for m in moves:
        env.move(m)
    return env


def puct(net=None, size=3, seed=0):
    return PUCT(NetEvaluator(net or Uniform(), 'cpu', size, symmetries=True, seed=seed), seed=seed)


def test_search_wins_and_blocks_with_an_uninformed_network():
    win = play(make_env('tictactoe'), [0, 3, 1, 4])  # X to move, wins at 2
    block = play(make_env('tictactoe'), [0, 4, 1])  # O to move, must block at 2
    for leaves in (1, 4):
        roots = puct().search([win, block], 200, leaves=leaves)
        assert [int(r.moves[np.argmax(r.n)]) for r in roots] == [2, 2]
        assert all(r.visits >= 200 and r.n.sum() == r.visits for r in roots)
    assert win.get_move_history() == [0, 3, 1, 4]  # the search left the games unchanged


def test_virtual_loss_spreads_leaves_over_the_tree():
    env = make_env('omok9')
    search = puct(size=9)
    root = search.search([env], 64, leaves=8)[0]
    assert search.evaluator.calls <= 1 + 64 // 4  # the root, then batches of up to 8 leaves
    assert root.visits == 64 and np.all(root.n >= 0) and np.allclose(root.w, 0)


def test_evaluator_undoes_the_symmetry():
    env = play(make_env('omok9'), [10, 40, 11])
    obs, mask = env.get_observation()[None], env.get_legal_mask()[None]
    plain = NetEvaluator(NearStones(), 'cpu', 9, symmetries=False)(obs, mask)[0][0]
    assert plain.max() > 3 * plain[mask[0]].min() and plain[~mask[0]].sum() == 0
    for seed in range(8):
        probs = NetEvaluator(NearStones(), 'cpu', 9, symmetries=True, seed=seed)(obs, mask)[0][0]
        assert np.allclose(probs, plain, atol=1e-6)


def test_permute_planes_matches_symmetries_of_the_board():
    env = play(make_env('omok9'), [10, 40, 11])
    obs = torch.as_tensor(env.get_observation())[None].repeat(8, 1, 1, 1)
    perms = torch.as_tensor(permutations(9)[0])
    out = permute_planes(obs, perms)
    for code in range(8):
        last = int(out[code, 3].flatten().argmax())
        assert int(perms[code][last]) == 11  # the transformed last-move cell maps back to the real one


def test_legal_from_obs_matches_env_mask():
    env = play(make_env('renju15'), [112, 113, 97, 98, 127, 128, 82])
    env.move(68)
    env.move(96)  # white plays, black to move: some forbidden points possible
    obs = torch.as_tensor(env.get_observation())[None]
    assert np.array_equal(legal_from_obs(obs)[0].numpy(), env.get_legal_mask())
    env9 = play(make_env('omok9'), [0, 1, 2])
    assert np.array_equal(legal_from_obs(torch.as_tensor(env9.get_observation())[None])[0].numpy(), env9.get_legal_mask())


def test_transfer_to_more_input_planes():
    small, large = AlphaZeroNet(4, 16, 1).eval(), AlphaZeroNet(5, 16, 1).eval()
    large.load_transfer(small.state_dict())
    env = play(make_env('omok9'), [40, 41])
    x4 = torch.as_tensor(env.get_observation())[None]
    x5 = torch.cat([x4, torch.zeros_like(x4[:, :1])], 1)
    for a, b in zip(small(x4), large(x5)):
        assert torch.allclose(a, b, atol=1e-6)


def test_replay_buffer_keeps_the_newest():
    buf = ReplayBuffer(5, (1, 2, 2), seed=0)
    for k in range(3):
        n = 3
        buf.add(np.full((n, 1, 2, 2), k, np.uint8), np.zeros((n, 4), np.float16), np.full(n, k, np.int8))
    assert buf.size == 5 and sorted(buf.z.tolist()) == [1, 1, 2, 2, 2]
    restored = ReplayBuffer(5, (1, 2, 2), seed=0)
    restored.load(buf.state())
    assert np.array_equal(restored.z, buf.z) and restored.next == buf.next


def test_self_play_targets():
    data = self_play(puct(), 'tictactoe', 6, simulations=20, temp_moves=2, rng=np.random.default_rng(0))
    n = len(data['z'])
    assert n == data['lengths'].sum() and data['obs'].shape == (n, 4, 3, 3)
    assert np.allclose(data['pi'].astype(np.float32).sum(1), 1, atol=1e-2)
    occupied = data['obs'][:, 0] + data['obs'][:, 1]
    assert np.all(data['pi'].astype(np.float32)[occupied.reshape(n, -1) > 0] == 0)  # no visits to occupied cells
    # the first position of each game is Black's: z is +1 if Black won
    first = np.r_[0, np.cumsum(data['lengths'])[:-1]]
    assert np.array_equal(data['z'][first], np.where(data['winners'] == 1, 1, np.where(data['winners'] == 2, -1, 0)))


def test_batched_match_matches_arena(tmp_path):
    net = AlphaZeroNet(4, 16, 1).eval()
    a = AlphaZeroAgent(net, 9, simulations=0, seed=0)
    openings = match_openings('omok9', 6, 4, seed=3)
    games = play_match(a, make_agent('heuristic', seed=1), 'omok9', openings)
    ref = evaluate(AlphaZeroAgent(net, 9, simulations=0, seed=0), make_agent('heuristic', seed=1), 'omok9', 6, 4, seed=3)
    assert [g.moves[:4] for g in games] == [g.moves[:4] for g in ref.games]
    assert [g.a_color for g in games] == [g.a_color for g in ref.games]
    path = tmp_path / 'az.pt'
    save_checkpoint(path, AZConfig(channels=16, blocks=1, eval_simulations=8), net, 9)
    agent = make_agent(str(path), seed=0)
    assert isinstance(agent, AlphaZeroAgent) and agent.simulations == 8
    assert make_env('omok9').get_legal_mask()[agent.act(make_env('omok9'))]


def test_worker_modules_do_not_import_torch():
    code = 'import sys, omok_rl.selfplay, omok_rl.inference; assert "torch" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True)
