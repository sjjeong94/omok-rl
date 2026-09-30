import numpy as np
import pytest
from omok.env import PLAYER_BLACK, PLAYER_NONE, PLAYER_WHITE

from omok_rl.agents import HeuristicAgent, RandomAgent
from omok_rl.agents.heuristic import line_windows
from omok_rl.arena import evaluate, play_game
from omok_rl.envs import ENVS, make_env


@pytest.mark.parametrize('name', list(ENVS))
def test_random_games_finish(name):
    env = play_game(make_env(name), RandomAgent(0), RandomAgent(1))
    assert env.is_done()
    assert env.get_winner() in (PLAYER_NONE, PLAYER_BLACK, PLAYER_WHITE)


def test_tictactoe_win_and_draw():
    # X X X on the top row
    env = make_env('tictactoe')
    for pos in (0, 3, 1, 4, 2):
        env.move(pos)
    assert env.is_done() and env.get_winner() == PLAYER_BLACK

    # X O X / X O O / O X X -> full board, no line
    env = make_env('tictactoe')
    for pos in (0, 1, 2, 4, 3, 5, 7, 6, 8):
        env.move(pos)
    assert env.is_done() and env.get_winner() == PLAYER_NONE


def test_line_windows_count():
    assert len(line_windows(3, 3)) == 8  # 3 rows, 3 columns, 2 diagonals
    # 15x15, five in a row: 11*15 rows + 11*15 columns + 11*11 * 2 diagonals
    assert len(line_windows(15, 5)) == 2 * 11 * 15 + 2 * 11 * 11


def test_heuristic_takes_win():
    # Black has 0, 1, 2, 3 on the top row of omok9; White has 9..12 as well, Black to move.
    env = make_env('omok9')
    for pos in (0, 9, 1, 10, 2, 11, 3, 12):
        env.move(pos)
    assert HeuristicAgent(0).act(env) == 4


def test_heuristic_blocks_win():
    # White has four on the second row (9..12, open at 13); Black must block at 13.
    env = make_env('omok9')
    for pos in (0, 9, 20, 10, 40, 11, 60, 12):
        env.move(pos)
    assert HeuristicAgent(0).act(env) == 13


def test_heuristic_never_plays_forbidden():
    env = make_env('renju15')
    agent = HeuristicAgent(0)
    opponent = RandomAgent(0)
    while not env.is_done():
        mover = agent if env.get_player() == PLAYER_BLACK else opponent
        pos = mover.act(env)
        assert env.get_legal_mask()[pos]
        env.move(pos)


def test_evaluate_alternates_colors_and_counts():
    result = evaluate(HeuristicAgent(0), RandomAgent(0), 'omok6', n_games=10)
    assert [g.a_color for g in result.games[:4]] == [PLAYER_BLACK, PLAYER_WHITE] * 2
    assert sum(result.count(o) for o in ('win', 'draw', 'loss')) == 10
    assert result.count('win', PLAYER_BLACK) + result.count('win', PLAYER_WHITE) == result.count('win')


def test_seeded_agents_are_reproducible():
    games = [evaluate(RandomAgent(3), RandomAgent(4), 'omok6', n_games=4).games for _ in range(2)]
    assert games[0] == games[1]


def test_random_agent_only_plays_legal_moves():
    env = make_env('omok6')
    agent = RandomAgent(0)
    for _ in range(10):
        pos = agent.act(env)
        assert np.flatnonzero(env.get_legal_mask()).tolist().count(pos) == 1
        env.move(pos)
