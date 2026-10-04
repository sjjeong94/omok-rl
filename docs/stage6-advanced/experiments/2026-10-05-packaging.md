# Shipping the Stage 6 networks in the `omok` package

> **TL;DR** — The Renju and freestyle networks of Stage 6 are now in the `omok` package as `omok.AlphaZeroAgent`, with the same interface as
> `omok.OmokAgent` (`agent(state, player)`, `agent.get_probs(state, player)`), so they also play in the web UI (`python -m omok --agent alphazero`).
> The networks don't need the "last move" input plane (no loss of strength), which is what makes the old interface enough.
> Through the package, they beat the package's previous best model, `b.onnx`: 0.70 / 0.83 with the network alone and 0.84 / 0.92 with
> a 200-simulation search (Renju / freestyle, 100 games each), in line with the results in omok-rl. A search of 200 simulations takes 1 s per move on the CPU.

## Why

Stage 6's networks beat the pretrained models shipped with `omok` (Experiment 3 of the [Stage 6 document](../README.md)), but they only ran inside
this repository: they need omok-rl's observation planes, its search, and either PyTorch or `python -m omok_rl.play`. The goal is to make them available
to anyone who installs `omok`, without torch, and without changing what existing users of `OmokAgent` see.

## The interface problem

`OmokAgent` and the web UI call `agent(state, player)`: the board (15x15, 0 / 1 black / 2 white) and whose turn it is. Our networks were trained on
`env.get_observation()`, five planes: own stones, opponent stones, "Black to play", **the last move**, and Renju forbidden points. Every plane but the
last move can be computed from `(state, player)`; the last move needs the move history. Rather than change the package's interface, we measured
whether the network needs it.

## Checks before shipping

- **Command**: `uv run python scripts/package_checks.py` (results in `runs/stage6/eval/package-*.json`). Code version `0ea6875-dirty` (the uncommitted change
  was the check script itself). 100 games per match from 50 random 4-move openings, each played once per color, unless noted; ±0.04–0.05 per score.

| Check | Match | Score |
|---|---|---:|
| Last move | Renju network, raw policy, vs b.onnx (Renju) | 0.74 |
| | the same, last-move plane set to 0 | 0.76 |
| | Renju network, PUCT 200 simulations, vs b.onnx | 0.88 |
| | the same, last-move plane set to 0 | 0.91 |
| | last-move plane at 0 vs the normal network, both PUCT 200 | 0.51 |
| One network for both rules? (freestyle games) | Renju network vs freestyle network, raw policy | 0.40 |
| | Renju network vs freestyle network, PUCT 200 | 0.52 |
| | Renju network vs b.onnx, raw / PUCT 200 | 0.84 / 0.86 |
| Empty board (as in the web UI) | Renju network vs b.onnx, raw policy | 0.67 |
| | the same, last-move plane set to 0 | 0.62 |
| | Renju network vs b.onnx, PUCT 200 | 0.89 |

- **Observation**:
  - **The last-move plane doesn't matter**: setting it to zero changes nothing beyond the noise, in either direction, and the two versions are even head to head (0.51).
    The board alone is enough, so the networks can keep `OmokAgent`'s interface. The exported networks set the plane to zero inside the graph, so they behave the same
    however they are called.
  - **Ship one network per rule**: the Renju network also plays freestyle well with search (0.52 against the freestyle network) but is behind without it (0.40), so
    freestyle gets its own network.
  - **From the empty board** the networks still beat b.onnx (0.62–0.67 raw, 0.89 with search). The games are longer (50–57 moves vs 35–40 from random openings).

## What was added

In this repository:

| File | What it does |
|---|---|
| [`src/omok_rl/play.py`](../../../src/omok_rl/play.py) | `export_onnx(..., ignore_last_move=True, metadata={...})`: input plane 3 multiplied by 0 inside the graph; source, commit and results stored as ONNX metadata |
| [`scripts/package_checks.py`](../../../scripts/package_checks.py) | The checks above |
| [`scripts/export_for_omok.py`](../../../scripts/export_for_omok.py) | Writes `alphazero-renju.onnx` and `alphazero-freestyle.onnx` (1.8 MB each) into the package's `models/` |
| [`scripts/package_parity.py`](../../../scripts/package_parity.py) | Plays the *package's* agent against b.onnx (below) |

In the `omok` package (`../omok`, not yet released):

| File | What it does |
|---|---|
| `omok/alphazero.py` | `AlphaZeroAgent(rule, simulations=0, ...)`: builds the planes from `(state, player)` (`env_from_state`), evaluates with onnxruntime through a random board symmetry, and optionally runs a PUCT search (a numpy-only copy of `omok_rl.puct.PUCT`, 8 leaves per network call with virtual loss). `get_probs` averages the network over the 8 symmetries, or returns the search's visit shares; the search is cached per position, so the move and the overlay share one search |
| `omok/agent.py` | `check_model(name)`: download any model file by name (`check_models(index)` now uses it) |
| `omok/__main__.py` | `python -m omok --agent alphazero [--simulations N]`; the default agent is unchanged |
| `test/test_alphazero.py` | Rebuilt env matches the real one (players, legal moves, forbidden points), the network ignores the last move, probabilities (legal, sum to 1, symmetric), search wins and blocks, legal games under both rules, a move through the web UI |
| `README.md` | Usage and results |

`env_from_state` builds an `Omok` env from a bare board: the env decides whose turn follows each move from the *length* of the move history, so the history is
filled with the board's stones (any order) to get the length right. Its contents only feed the last-move plane, which the network ignores.

## Parity: the package's agent against b.onnx

- **Question**: does the package's re-implementation (planes from the board, numpy search, onnxruntime) play as well as omok-rl's?
- **Command**: `PYTHONPATH=../omok uv run python scripts/package_parity.py --rule renju` (and `--rule freestyle`); 12 CPU processes with one ONNX thread each.
  Same protocol as above.

| Rule | Package, raw policy | omok-rl, raw policy | Package, 200 simulations | omok-rl, 200 simulations |
|---|---:|---:|---:|---:|
| Renju | 0.70 (72% / 68% wins as Black / White) | 0.74–0.76 | 0.84 (88% / 80%) | 0.88–0.91 (PUCT), 0.95 (Gumbel) |
| freestyle | 0.83 (88% / 78%) | 0.82 | 0.92 (100% / 84%) | 0.88 (PUCT) |

- **Observation**: the package agent plays as well as the original within the noise of two 100-game samples (about ±0.06 for a difference).
  **Speed**, one agent alone with onnxruntime's default threads: 0.01 s per move for the network, 0.26 s with 50 simulations, 1.0 s with 200, 2.0 s with 400.
  One network call costs 5.9 ms for one position and 34 ms for 8, so batching 8 leaves per call saves about 4x over evaluating them one by one.

## Not done yet

- **Release** (step 4 of the plan): upload the two model files to `models/` in the `omok` repository (the agent downloads them from there; until then,
  pass `model_path=`), bump the version, publish, and pin the new version here.
- The web UI's default agent is still `OmokAgent` (b.onnx); making `AlphaZeroAgent` the default is a one-line change in `omok/__main__.py`.
