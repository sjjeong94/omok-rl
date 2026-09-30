import numpy as np
import pytest
from omok.env import PLAYER_BLACK, PLAYER_WHITE

from omok_rl.agents import HeuristicAgent, MinimaxAgent, RandomAgent, make_agent
from omok_rl.agents.minimax import MinimaxSolver, all_positions, optimal_move_rate, worst_case
from omok_rl.agents.tabular import make_tabular
from omok_rl.arena import evaluate
from omok_rl.envs import make_env
from omok_rl.symmetry import canonical


def test_canonical_is_shared_by_symmetric_boards():
    keys = set()
    for corner in (0, 2, 6, 8):
        b = np.zeros(9, np.uint8)
        b[corner] = 1  # X in a corner
        keys.add(canonical(b, 3)[0])
    assert len(keys) == 1
    center = np.zeros(9, np.uint8)
    center[4] = 1
    assert canonical(center, 3)[0] not in keys


def test_canonical_action_map_is_consistent():
    rng = np.random.default_rng(0)
    for _ in range(50):
        board = rng.integers(0, 3, 9).astype(np.uint8)
        key, action_map = canonical(board, 3)
        canon = np.frombuffer(key, np.uint8)
        for a in range(9):  # the stone at a lands on action_map[a] in the canonical board
            assert canon[action_map[a]] == board[a]


def test_minimax_known_values():
    solver = MinimaxSolver()
    assert solver.value(make_env('tictactoe')) == 0.0  # perfect play draws
    assert len(all_positions(make_env('tictactoe'))) == 4520
    env = make_env('tictactoe')
    for pos in (0, 3, 1, 4):  # X X . / O O . / . . .  -> X wins at 2
        env.move(pos)
    assert solver.optimal_actions(env) == [2]


def test_minimax_never_loses():
    result = evaluate(MinimaxAgent(0), RandomAgent(0), 'tictactoe', 50)
    assert result.count('loss') == 0


def test_worst_case():
    solver = MinimaxSolver()
    perfect = lambda env: solver.optimal_actions(env.clone())
    everything = lambda env: [int(a) for a in np.flatnonzero(env.get_legal_mask())]
    env = make_env('tictactoe')
    assert worst_case(perfect, env, PLAYER_BLACK) == 0.0
    assert worst_case(perfect, env, PLAYER_WHITE) == 0.0
    assert worst_case(everything, env, PLAYER_BLACK) == -1.0
    assert worst_case(HeuristicAgent(0).greedy_actions, env, PLAYER_BLACK) == 0.0


def test_optimal_move_rate_of_perfect_player_is_one():
    solver = MinimaxSolver()
    positions = all_positions(make_env('tictactoe'))[:300]
    optimal = [set(solver.optimal_actions(p)) for p in positions]
    assert optimal_move_rate(lambda env: solver.optimal_actions(env.clone()), positions, optimal) == 1.0


@pytest.mark.parametrize('method', ['mc', 'td', 'sarsa', 'q-learning'])
def test_tabular_learns_to_beat_random(method):
    agent = make_tabular(method, seed=0)
    for _ in range(3000):
        agent.train_game('tictactoe', epsilon=0.1)
    assert evaluate(agent, RandomAgent(0), 'tictactoe', 100).score() > 0.8


def test_q_values_stay_in_range_and_winning_move_is_learned():
    agent = make_tabular('q-learning', seed=0)
    for _ in range(3000):
        agent.train_game('tictactoe', epsilon=0.3)
    assert all(np.abs(row).max() <= 1.0 for row in agent.table.values())
    env = make_env('tictactoe')
    for pos in (0, 3, 1, 4):
        env.move(pos)
    assert agent.greedy_actions(env) == [2]


def test_evaluation_does_not_grow_the_table():
    agent = make_tabular('q-learning', seed=0)
    for _ in range(100):
        agent.train_game('tictactoe', epsilon=0.1)
    size = len(agent.table)
    evaluate(agent, RandomAgent(0), 'tictactoe', 20)
    assert len(agent.table) == size


def test_save_and_load(tmp_path):
    agent = make_tabular('td', seed=0)
    for _ in range(200):
        agent.train_game('tictactoe', epsilon=0.1)
    path = tmp_path / 'td.pkl'
    agent.save(path)
    loaded = make_agent(str(path), seed=0)
    assert loaded.table == agent.table
