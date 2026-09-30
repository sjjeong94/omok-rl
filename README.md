# omok-rl

A project for studying Reinforcement Learning by building an Omok (Gomoku) agent step by step.
The game environment is the [`omok`](https://pypi.org/project/omok/0.1.0/) package, and the development environment is managed with [uv](https://docs.astral.sh/uv/).

> **Goal**: not to build a strong Omok AI right away, but to climb one step at a time —
> tabular RL → value function approximation → policy gradients → search (MCTS) → AlphaZero —
> and experience firsthand **why** each technique is needed.

---

## 1. Setup

```bash
uv sync                        # create .venv with Python 3.12, omok[fast], numpy, and dev tools
uv run pytest                  # run the tests
uv run omok-arena heuristic random --env omok9 --games 200   # play a match between two agents
uv run python -m omok          # play in the browser (http://127.0.0.1:8000)
```

`omok[fast]` compiles the Renju forbidden-move check with numba (~30x faster). PyTorch will be added in Stage 2 (`uv add torch`).

## 2. `omok` Package Cheat Sheet

| Feature | API | Notes |
|---|---|---|
| Create env | `omok.Omok(rule='renju' \| 'freestyle')` | 15x15, Renju by default |
| Gym style | `obs, info = env.reset()`, `obs, reward, terminated, truncated, info = env.step(a)` | `reward` is for the **player who just moved** (1 for a win, else 0) |
| Observation | `env.get_observation()` → `(5, 15, 15)` float32 | From the **player to move next**: own stones / opponent stones / black to play / last move / forbidden points |
| Legal moves | `env.get_legal_mask()` / `info['action_mask']` → `(225,)` bool | Excludes Renju forbidden points |
| Result | `env.is_done()`, `env.get_winner()` | 0: draw, 1: black, 2: white |
| For search | `env.clone()`, `env.move_back()`, `Omok.from_moves([...])` | `clone()` is ~6x faster than deepcopy |
| Augmentation | `omok.transforms.symmetries(obs, policy)` | 8 rotations/reflections |
| Benchmark opponent | `omok.OmokAgent(model_index=0 \| 1)` | Pretrained ONNX model, used for final evaluation |

**Smaller boards** — 15x15 is too large for the early stages, so subclass `BoardGame` and change only the board size and win length.

```python
from omok.env import BoardGame

class TicTacToe(BoardGame):   # 3x3, three in a row
    size, win = 3, 3

class Omok6(BoardGame):       # 6x6, four in a row (freestyle)
    size, win = 6, 4
```

> ⚠️ This is a two-player zero-sum game, so pay attention to the **sign and perspective of rewards**.
> The reward from `step()` is for the player who just moved, while the observation is for the player to move next.
> When learning values, remember to flip the sign in negamax form: `V(s) = -V(s')`.

---

## 3. Roadmap

Each stage follows **(1) new concepts → (2) implementation → (3) experiments/observations compared to the previous stage**.
Grow the board only when needed: `3x3 (3 in a row) → 6x6 (4 in a row) → 9x9 (5 in a row) → 15x15 freestyle → 15x15 renju`.

### Stage 0. Foundations — Environment & Evaluation Tools
- **Concepts**: MDPs, episodes, states/actions/rewards, two-player zero-sum games, self-play
- **Implementation**
  - `RandomAgent`, `HeuristicAgent` (simple rules such as "complete a five", "block an open four")
  - Common `Agent` interface: `act(env) -> action`
  - `arena.py`: play N games between two agents (alternating colors) and compute win rates
  - Establish **evaluation baselines** used in every later stage (vs Random, vs Heuristic)
- **Checkpoint**: Heuristic beats Random nearly 100% of the time

### Stage 1. Tabular RL — 3x3 Tic-Tac-Toe
- **Concepts**: value functions, Monte Carlo vs Temporal Difference, ε-greedy exploration, on-policy vs off-policy, afterstate values
- **Implementation**
  1. Monte Carlo value estimation (`dict[state] -> value`)
  2. TD(0) / Sarsa / Q-learning
  3. Learning via self-play (the tic-tac-toe example from Sutton & Barto, Chapter 1)
  4. Reducing the state space with the 8 symmetries
- **Experiments**: convergence speed vs learning rate and ε schedule; compare with exhaustive minimax to check whether the policy reaches "never lose"
- **Takeaway**: even at 6x6 the state space explodes → why function approximation is needed

### Stage 2. Value-Based Deep RL — DQN
- **Concepts**: function approximation, experience replay, target networks, the deadly triad, action masking
- **Implementation** (6x6 four-in-a-row → 9x9)
  1. Vanilla DQN (CNN, illegal moves masked with Q = `-inf`)
  2. Double DQN, Dueling DQN, N-step returns
  3. Prioritized Experience Replay (optional)
  4. Self-play opponents: play against frozen past versions → opponent pool
- **Experiments**: ablation of each improvement, learning curves (win rate vs Random/Heuristic)
- **Takeaway**: non-stationarity in self-play, difficulty of sparse rewards

### Stage 3. Policy Gradients — REINFORCE → A2C → PPO
- **Concepts**: policy gradient theorem, baselines and variance reduction, advantage, GAE, clipping, entropy bonus
- **Implementation** (9x9)
  1. REINFORCE (+ baseline)
  2. Actor-Critic (A2C) with parallel environments
  3. PPO (masked logits, GAE)
  4. Self-play against a pool of past checkpoints (a simple league)
- **Experiments**: sample efficiency and stability vs DQN, effect of the entropy coefficient on exploration

### Stage 4. Search — Monte Carlo Tree Search
- **Concepts**: Minimax/Alpha-Beta, multi-armed bandits, UCB1, UCT, rollout policies
- **Implementation**
  1. Minimax + Alpha-Beta (ground truth on small boards)
  2. Pure MCTS with random rollouts — using `env.clone()`
  3. Heuristic rollouts / using networks from Stages 2–3 for rollouts or value estimates
- **Experiments**: number of simulations vs playing strength; how strong search alone can get without learning

### Stage 5. AlphaZero — Combining Search and Learning
- **Concepts**: MCTS as policy improvement, PUCT, policy-value networks, Dirichlet noise, temperature
- **Implementation**
  1. ResNet policy-value network (input: the 5 planes from `get_observation()`)
  2. PUCT MCTS (batched inference, virtual loss)
  3. Self-play → replay buffer → training → evaluate new model (gating) loop
  4. 8x data augmentation with `omok.transforms.symmetries`
  5. Scale up: 9x9 → 15x15 freestyle → 15x15 renju
- **Experiments**: Elo curve across generations, games against `omok.OmokAgent`

### Stage 6. Advanced (Optional)
- Efficiency: multi-process self-play, ONNX/TensorRT inference, tree reuse
- Recent techniques: Gumbel AlphaZero (few simulations), KataGo-style auxiliary targets, MuZero (learned model)
- Opening diversity, analysis of black/white asymmetry under Renju
- Export the trained model to ONNX and hook it up to the `python -m omok` web UI

---

## 4. Project Structure (Planned)

```
omok-rl/
├── pyproject.toml
├── README.md
├── src/omok_rl/
│   ├── envs.py            # tictactoe, omok6, omok9, freestyle15, renju15
│   ├── agents/            # random, heuristic, minimax, tabular, (dqn, ppo, mcts, alphazero)
│   ├── symmetry.py        # canonical board keys under the 8 symmetries
│   ├── arena.py           # matches & win-rate evaluation (omok-arena CLI)
│   ├── viz.py             # board drawing for docs figures
│   └── utils/             # logging, checkpoints, replay buffer
├── scripts/               # per-stage training/evaluation scripts
│   ├── stage0_baselines.py
│   ├── plot_stage0.py
│   ├── stage1_tabular.py
│   ├── plot_stage1.py
│   ├── stage2_dqn.py
│   └── ...
├── docs/                  # per-stage study log & experiment results (see docs/README.md)
└── tests/
```

```bash
uv run python scripts/stage1_tabular.py
uv run omok-arena ppo heuristic --env omok9 --games 200   # colors alternate every game
```

## 5. Evaluation

- **Win rate vs fixed opponents**: Random, Heuristic, and (later) the best agent from previous stages
- **Alternate colors**: black has the advantage in Omok, so always play an equal number of games as each color
- **Elo rating**: round-robin among checkpoints to track growth across generations
- **Final benchmark**: `omok.OmokAgent(model_index=0/1)`
- Log experiments with TensorBoard (or CSV), and record results in `docs/` following the [documentation guidelines](docs/README.md)

## 6. Progress Checklist

- [x] Stage 0 — Environment, Random/Heuristic agents, Arena ([docs](docs/stage0-foundations/README.md))
- [x] Stage 1 — Tabular MC / TD / Sarsa / Q-learning (tic-tac-toe) ([docs](docs/stage1-tabular/README.md))
- [ ] Stage 2 — DQN family (6x6 → 9x9)
- [ ] Stage 3 — REINFORCE / A2C / PPO
- [ ] Stage 4 — Minimax, MCTS
- [ ] Stage 5 — AlphaZero (9x9 → 15x15)
- [ ] Stage 6 — Advanced

## 7. References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Stages 1–3
- David Silver, UCL RL Course — general concepts
- OpenAI, *Spinning Up in Deep RL* — Stages 2–3
- Mnih et al., 2015 (DQN) / van Hasselt et al., 2016 (Double DQN) / Wang et al., 2016 (Dueling)
- Schulman et al., 2017 (PPO)
- Silver et al., 2017/2018 (AlphaGo Zero, AlphaZero)
- Wu, 2019 (KataGo) / Danihelka et al., 2022 (Gumbel AlphaZero)
