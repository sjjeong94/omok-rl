from dataclasses import replace

import numpy as np
import pytest
import torch
from omok.env import PLAYER_BLACK, PLAYER_NONE, PLAYER_WHITE

from omok_rl.agents import make_agent
from omok_rl.agents.policy import PolicyAgent
from omok_rl.envs import make_env
from omok_rl.nets import PolicyValueNet
from omok_rl.pg import PGConfig, Trajectory, advantages, entropy, finish, masked_log_probs, train


def test_gae_with_lambda_one_is_return_minus_value():
    rewards = np.array([0, 0, 1], np.float32)
    values = np.array([0.2, -0.5, 0.4], np.float32)
    adv = advantages(rewards, values, gamma=1.0, lam=1.0)
    assert np.allclose(adv + values, [1, 1, 1])  # G_t = final reward for every move


def test_gae_with_lambda_zero_is_one_step_td_error():
    rewards = np.array([0, 0, -1], np.float32)
    values = np.array([0.2, -0.5, 0.4], np.float32)
    adv = advantages(rewards, values, gamma=0.9, lam=0.0)
    assert np.allclose(adv, [0.9 * -0.5 - 0.2, 0.9 * 0.4 + 0.5, -1 - 0.4])


def make_traj(n):
    t = Trajectory()
    for k in range(n):
        t.obs.append(np.zeros((4, 3, 3), np.float32))
        t.mask.append(np.ones(9, bool))
        t.action.append(k)
        t.log_prob.append(0.0)
        t.value.append(0.0)
    return t


@pytest.mark.parametrize('winner, black, white', [(PLAYER_BLACK, 1, -1), (PLAYER_WHITE, -1, 1), (PLAYER_NONE, 0, 0)])
def test_each_player_gets_the_final_result_from_its_own_view(winner, black, white):
    config = PGConfig(env='tictactoe', algo='reinforce')
    samples = finish({PLAYER_BLACK: make_traj(3), PLAYER_WHITE: make_traj(2)}, winner, config)
    assert [s['ret'].tolist() for s in samples] == [[black] * 3, [white] * 2]


def test_opponent_moves_are_not_learned_from():
    config = PGConfig(env='tictactoe')
    samples = finish({PLAYER_BLACK: make_traj(3), PLAYER_WHITE: Trajectory()}, PLAYER_WHITE, config)
    assert len(samples) == 1 and len(samples[0]['action']) == 3


def test_algorithm_defaults_follow_replace():
    ppo = PGConfig(algo='ppo')
    assert (ppo.n_epochs, ppo.n_minibatches) == (4, 4)
    a2c = replace(ppo, algo='a2c')  # must not inherit PPO's 4 epochs x 4 minibatches
    assert (a2c.n_epochs, a2c.n_minibatches) == (1, 1)
    assert replace(ppo, algo='a2c', minibatches=16).n_minibatches == 16


def test_masked_policy_never_picks_illegal_moves():
    logits = torch.randn(4, 9)
    mask = torch.zeros(4, 9, dtype=torch.bool)
    mask[:, [2, 5]] = True
    log_probs = masked_log_probs(logits, mask)
    assert torch.allclose(log_probs.exp().sum(1), torch.ones(4))
    assert torch.all(log_probs.exp()[~mask] == 0)
    assert torch.all(entropy(log_probs) <= np.log(2) + 1e-6)  # only 2 legal moves
    assert torch.isfinite(entropy(log_probs)).all()


def test_policy_value_net_shapes_on_any_board_size():
    net = PolicyValueNet(4, channels=8, blocks=1)
    for size in (3, 9):
        logits, value = net(torch.zeros(2, 4, size, size))
        assert logits.shape == (2, size * size) and value.shape == (2,)


@pytest.mark.parametrize('algo, pool_prob', [('reinforce', 0.0), ('a2c', 0.0), ('ppo', 0.5)])
def test_short_training_run_saves_a_playable_agent(tmp_path, algo, pool_prob):
    config = PGConfig(env='tictactoe', algo=algo, total_steps=1500, num_envs=16, batch_moves=256, eval_every=750,
                      eval_games=4, eval_openings=0, channels=8, blocks=1, pool_prob=pool_prob, snapshot_every=1)
    result = train(config, tmp_path / 'run', device='cpu', log=lambda *_: None)
    assert len(result['curve']) == 2
    if pool_prob:
        assert result['curve'][-1]['pool'] > 0
    agent = make_agent(str(tmp_path / 'run.pt'))
    assert isinstance(agent, PolicyAgent)
    env = make_env('tictactoe')
    assert env.get_legal_mask()[agent.act(env)]
