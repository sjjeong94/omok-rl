import numpy as np
import torch
from torch import nn

from omok_rl.envs import make_env
from omok_rl.nets import NetEvaluator
from omok_rl.puct import PUCT, Gumbel, halving_schedule
from omok_rl.selfplay import self_play


class Uniform(nn.Module):
    """Equal logits everywhere and value 0: the search alone must find the tactics."""

    def forward(self, x):
        return torch.zeros(len(x), x.shape[2] * x.shape[3]), torch.zeros(len(x))


def play(env, moves):
    for m in moves:
        env.move(m)
    return env


def evaluator(size=3, seed=0):
    return NetEvaluator(Uniform(), 'cpu', size, symmetries=False, seed=seed)


def test_halving_schedule():
    s = halving_schedule(16, 200)
    assert len(s) == 200 and np.all(s[:16] == 0)
    counts = np.bincount(s)
    assert np.all(np.diff(counts) <= 0)  # fewer moves remain considered as visits grow
    assert counts[-1] == 2  # the final phase has two moves left
    assert list(halving_schedule(4, 8)) == [0, 0, 0, 0, 1, 1, 2, 2]
    assert list(halving_schedule(1, 3)) == [0, 1, 2]


def test_gumbel_wins_and_blocks_with_an_uninformed_network():
    win = play(make_env('tictactoe'), [0, 3, 1, 4])  # X to move, wins at 2
    block = play(make_env('tictactoe'), [0, 4, 1])  # O to move, must block at 2
    for seed in range(3):
        search = Gumbel(evaluator(), seed=seed)
        roots = search.search([win, block], 64, noise=True)
        assert [search.choose(r, np.random.default_rng(0)) for r in roots] == [2, 2]
        assert all(r.n.sum() == 64 for r in roots)
        for r in roots:  # the improved policy puts most of its weight on the right move
            pi = search.target(r, 3)
            assert abs(pi.sum() - 1) < 1e-5 and pi.argmax() == 2


def test_gumbel_target_without_visits_is_the_prior():
    search = Gumbel(evaluator(9), seed=0)
    env = play(make_env('omok9'), [40])
    root = search.search([env], 0)[0]
    assert np.allclose(search.target(root, 9)[root.moves], root.prior)


def test_gumbel_considers_only_a_few_root_moves():
    search = Gumbel(evaluator(9), considered=4, seed=0)
    root = search.search([make_env('omok9')], 20, noise=True)[0]
    assert (root.n > 0).sum() == 4 and root.n.sum() == 20


def test_per_position_simulations_and_tree_reuse():
    search = PUCT(evaluator(9), seed=0)
    envs = [make_env('omok9'), play(make_env('omok9'), [40])]
    roots = search.search(envs, [10, 30])
    assert [r.visits for r in roots] == [10, 30]
    i = int(np.argmax(roots[1].n))
    child = roots[1].children[i]
    before = child.visits
    envs[1].move(int(roots[1].moves[i]))
    calls = search.evaluator.calls
    new_root = search.search([envs[1]], 30, roots=[child])[0]
    assert new_root is child and new_root.visits == 30
    assert search.evaluator.calls - calls == 30 - before  # only the missing simulations, no root evaluation


def test_playout_cap_randomization_records_only_full_searches():
    rng = np.random.default_rng(0)
    data = self_play(PUCT(evaluator(), seed=0), 'tictactoe', 20, simulations=16, temp_moves=2, rng=rng,
                     fast_simulations=4, full_prob=0.25)
    n = len(data['z'])
    assert n == data['full_moves'].sum() and 0 < n < data['lengths'].sum() * 0.5
    assert np.allclose(data['pi'].astype(np.float32).sum(1), 1, atol=1e-2)


def test_gumbel_self_play_and_tree_reuse_self_play():
    for search, reuse in ((Gumbel(evaluator(), seed=0), False), (PUCT(evaluator(), seed=0), True)):
        data = self_play(search, 'tictactoe', 6, simulations=16, temp_moves=2, rng=np.random.default_rng(0),
                         reuse_tree=reuse)
        n = len(data['z'])
        assert n == data['lengths'].sum()
        occupied = data['obs'][:, 0] + data['obs'][:, 1]
        assert np.all(data['pi'].astype(np.float32)[occupied.reshape(n, -1) > 0] == 0)
        assert np.allclose(data['pi'].astype(np.float32).sum(1), 1, atol=1e-2)


def test_onnx_export_matches_torch_and_plays_in_the_web_ui(tmp_path):
    import omok

    from omok_rl.alphazero import AZConfig, save_checkpoint
    from omok_rl.nets import AlphaZeroNet
    from omok_rl.play import OnnxEvaluator, WebAgent, export_onnx

    torch.manual_seed(0)
    net = AlphaZeroNet(4, 16, 1)
    for p in net.parameters():  # move the BN statistics away from the defaults so they are exported for real
        if p.dim() == 1:
            p.data.uniform_(0.5, 1.5)
    net.eval()
    ckpt = tmp_path / 'az.pt'
    save_checkpoint(ckpt, AZConfig(channels=16, blocks=1), net, 9)
    onnx_path = export_onnx(ckpt, tmp_path / 'az.onnx', in_channels=5)  # an omok9 network for 15x15 Renju

    env = play(make_env('renju15'), [112, 113, 97])
    obs, mask = env.get_observation()[None], env.get_legal_mask()[None]
    large = AlphaZeroNet(5, 16, 1)
    large.load_transfer(net.state_dict())
    large.eval()
    obs8, mask8 = obs.repeat(8, 0), mask.repeat(8, 0)
    for symmetries in (False, True):  # the same seed draws the same symmetry for each position
        ref = NetEvaluator(large, 'cpu', 15, symmetries=symmetries, seed=1)(obs8, mask8)
        got = OnnxEvaluator(onnx_path, 15, symmetries=symmetries, seed=1)(obs8, mask8)
        assert np.allclose(got[0], ref[0], atol=1e-5) and np.allclose(got[1], ref[1], atol=1e-5)

    agent = WebAgent(OnnxEvaluator(onnx_path, 15, seed=0), simulations=16)
    game = omok.OmokGame(agent=agent, rule='renju')
    agent.bind(game.env)
    status = game.handle('move', {'pos': 112, 'reply': True, 'probs': True})
    assert len(status['moves']) == 2 and abs(sum(status['probs']) - 1) < 1e-3
    assert status['probs'][status['moves'][0]] == 0  # occupied cells get no probability
