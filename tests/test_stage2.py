import numpy as np
import torch

from omok_rl.agents import make_agent
from omok_rl.dqn import DQNConfig, augment_batch, train
from omok_rl.envs import make_env
from omok_rl.nets import make_qnet
from omok_rl.replay import NStepWriter, ReplayBuffer
from omok_rl.symmetry import permutations


def play(moves, n):
    """Play `moves` on tic-tac-toe through an NStepWriter and return the buffer."""
    env = make_env('tictactoe')
    buffer = ReplayBuffer(100, env.get_observation().shape)
    writer = NStepWriter(buffer, n)
    for a in moves:
        obs, mover = env.get_observation(), env.get_player()
        env.move(a)
        reward = 1.0 if env.is_done() and env.get_winner() == mover else 0.0
        writer.step(obs, a, env.get_observation(), env.get_legal_mask(), env.is_done(), reward)
    return buffer


def test_one_step_transitions():
    # X: 0, 1, 2 (wins on move 5); O: 3, 4
    b = play([0, 3, 1, 4, 2], n=1)
    assert len(b) == 5
    assert b.action[:5].tolist() == [0, 3, 1, 4, 2]
    assert b.sign[:4].tolist() == [-1, -1, -1, -1]  # bootstrap from the opponent's position, negated
    assert b.ret[4] == 1.0 and b.sign[4] == 0.0  # the winning move: reward, no bootstrap


def test_n_step_returns_alternate_sign_at_the_end():
    b = play([0, 3, 1, 4, 2], n=3)
    # moves 0, 1 get bootstrapped 3-step targets; the last 3 moves see the end of the game
    assert b.action[:5].tolist() == [0, 3, 1, 4, 2]
    assert b.sign[:2].tolist() == [-1, -1]  # (-1)^3
    assert b.ret[2:5].tolist() == [1.0, -1.0, 1.0]  # X's move 1 (+), O's move 4 (-), X's winning move 2 (+)
    assert b.sign[2:5].tolist() == [0, 0, 0]


def test_draw_gives_zero_returns():
    b = play([0, 1, 2, 4, 3, 5, 7, 6, 8], n=2)
    assert len(b) == 9
    assert np.all(b.ret[:9] == 0)


def test_augmentation_moves_stones_and_actions_together():
    env = make_env('omok6')
    for a in (0, 7, 13):
        env.move(a)
    obs = torch.as_tensor(env.get_observation())[None].repeat(64, 1, 1, 1)
    mask = torch.as_tensor(env.get_legal_mask())[None].repeat(64, 1)
    action = torch.full((64,), 20)
    perms, inverse = (torch.as_tensor(x) for x in permutations(6))
    o, a, no, m = augment_batch(obs, action, obs.clone(), mask, perms, inverse)
    for i in range(64):
        # the stones and the chosen action are transformed identically, so the cell stays empty and legal
        assert m[i, a[i]]
        assert o[i, 0].flatten()[a[i]] == 0 and o[i, 1].flatten()[a[i]] == 0
        assert o[i, :2].sum() == 3  # no stone lost
        assert torch.equal(o[i], no[i])


def test_cnn_works_on_any_board_size():
    net = make_qnet('cnn', 4, size=6)
    assert net(torch.zeros(2, 4, 6, 6)).shape == (2, 36)
    assert net(torch.zeros(2, 4, 9, 9)).shape == (2, 81)


def test_short_training_run_saves_a_playable_agent(tmp_path):
    config = DQNConfig(env='tictactoe', total_steps=2048, num_envs=32, learning_starts=512, eval_every=1024,
                       eval_games=4, double=True, dueling=True, n_step=3, channels=16, blocks=1)
    result = train(config, tmp_path / 'run', device='cpu', log=lambda *_: None)
    assert len(result['curve']) == 2
    agent = make_agent(str(tmp_path / 'run.pt'))
    env = make_env('tictactoe')
    assert env.get_legal_mask()[agent.act(env)]
