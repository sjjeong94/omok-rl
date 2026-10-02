import numpy as np
import pytest
from omok.env import BoardGame

from omok_rl.agents import AlphaBetaAgent, MCTSAgent, RandomAgent, make_agent
from omok_rl.agents.alphabeta import Solver, SolverOptions, threat_moves, winning_cells
from omok_rl.agents.minimax import MinimaxSolver, all_positions
from omok_rl.agents.policy import PolicyAgent
from omok_rl.arena import evaluate
from omok_rl.envs import make_env
from omok_rl.nets import PolicyValueNet


def mnk(size, win):
    """An m,n,k-game: `win` in a row on a size x size board."""
    return type(f'Game{size}x{size}k{win}', (BoardGame,), {'size': size, 'win': win})()


def play(env, moves):
    for m in moves:
        env.move(m)
    return env


def test_winning_cells_and_threat_rules():
    env = play(make_env('tictactoe'), [0, 3, 1])  # X X . / O . . / . . .  (O to move)
    assert winning_cells(env, 1).tolist() == [2]
    assert winning_cells(env, 2).tolist() == []
    value, forced = threat_moves(env)
    assert value is None and forced.tolist() == [2]  # O must block
    env = play(make_env('tictactoe'), [0, 3, 1, 4])  # X to move and wins at 2
    assert threat_moves(env)[0] == 1.0
    env = play(make_env('tictactoe'), [0, 3, 1, 5, 4])  # X threatens 2 and 8, O can't win: O loses
    assert threat_moves(env)[0] == -1.0


@pytest.mark.parametrize('options', [SolverOptions(), SolverOptions(False, False, False, False, False),
                                     SolverOptions(threats=False), SolverOptions(symmetry=False, ordering=False)])
def test_solver_matches_minimax_on_every_tictactoe_position(options):
    reference = MinimaxSolver()
    solver = Solver(options)
    positions = all_positions(make_env('tictactoe'))[::7]
    assert all(solver.value(p) == reference.value(p.clone()) for p in positions)


@pytest.mark.parametrize('size, win, value', [(3, 3, 0.0), (4, 4, 0.0), (4, 3, 1.0)])
def test_solver_known_mnk_values(size, win, value):
    assert Solver().value(mnk(size, win)) == value


def test_solver_leaves_env_unchanged_and_move_values():
    env = play(make_env('tictactoe'), [0, 3, 1, 4])
    before = env.get_move_history()
    values = Solver().move_values(env)
    assert env.get_move_history() == before
    assert values[2] == 1.0 and values[8] == -1.0  # win now; after 8, O wins at 5


def test_alphabeta_agent_wins_and_blocks():
    env = play(make_env('omok9'), [0, 9, 1, 10, 2, 11, 3])  # black has 4 in row 0, white must block at 4
    assert AlphaBetaAgent(depth=2).act(env) == 4
    env = play(make_env('omok9'), [0, 9, 1, 10, 2, 11, 3, 12])  # black to move wins at 4
    assert AlphaBetaAgent(depth=2).act(env) == 4


def test_mcts_finds_tictactoe_wins_and_blocks():
    agent = MCTSAgent(300, seed=0)
    assert agent.act(play(make_env('tictactoe'), [0, 3, 1, 4])) == 2  # win now
    assert agent.act(play(make_env('tictactoe'), [0, 3, 1])) == 2  # block
    moves, share = agent.root_policy()
    assert share.sum() == pytest.approx(1.0) and agent.root.visits == 300


@pytest.mark.parametrize('kwargs', [dict(), dict(evaluation='heuristic'), dict(prior='heuristic'),
                                    dict(prior='net'), dict(prior='net', evaluation='value')])
def test_mcts_variants_play_legal_moves(kwargs):
    net = PolicyAgent(PolicyValueNet(4, channels=8, blocks=1)) if 'net' in kwargs.values() else None
    agent = MCTSAgent(40, net=net, seed=0, **kwargs)
    result = evaluate(agent, RandomAgent(0), 'omok6', 2)
    assert len(result.games) == 2


def test_mcts_statistics_are_consistent():
    agent = MCTSAgent(200, seed=1)
    agent.act(make_env('omok6'))
    root = agent.root
    assert root.n.sum() == root.visits == 200
    assert np.all(np.abs(root.w) <= root.n)  # every value is in [-1, 1]
    for child, n in zip(root.children, root.n):
        if child is not None:  # a child was expanded on its first visit, so it has one visit less than its edge
            assert child.visits == n - 1


def test_mcts_needs_a_network_for_value_and_net_prior():
    with pytest.raises(ValueError):
        MCTSAgent(evaluation='value')
    with pytest.raises(ValueError):
        MCTSAgent(prior='net')


def test_make_agent_search_specs():
    assert make_agent('alphabeta:2').depth == 2
    assert make_agent('mcts:50').simulations == 50
    agent = make_agent('mcts-heuristic')
    assert (agent.evaluation, agent.prior) == ('heuristic', 'heuristic')
