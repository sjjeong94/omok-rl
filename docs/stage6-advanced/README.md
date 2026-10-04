# Stage 6. Advanced — Cheaper Search Targets, 15x15, and Playing in the Browser

> **TL;DR** — *(in progress)*

## Goal

Stage 5 left one success criterion unmet: on the 15x15 board, self-play collapsed into 9-move games where Black attacks
and White never learns to defend, so no 15x15 network beat the heuristic. The cause was the search's **breadth**: 200 PUCT
simulations spread over ~220 legal moves rarely find White's single blocking move, so the training targets never teach it.
This stage tries the fixes that Stage 5's [Next Steps](../stage5-alphazero/README.md#next-steps) listed, measures each on
`omok9` first (where we have a reference run), and then applies the best to 15x15.

1. **Gumbel AlphaZero** (Danihelka et al., 2022): a policy-improvement step that is still an improvement with very few
   simulations (2–50), because it plans which root moves to try instead of letting visit counts decide.
2. **Playout cap randomization** (Wu, 2019, KataGo): a full search on a random quarter of the moves (the training samples),
   a cheap one on the rest (which only serve to finish the game).
3. **Tree reuse**: keep the subtree of the move played as the next search's root.
4. **Scale to 15x15** freestyle and Renju with whichever works, and compare with the `omok` package's pretrained models.
5. **Black/White asymmetry** under Renju vs freestyle, measured in self-play and evaluation games.
6. **Export** the network to ONNX and hook it up to the `omok` web UI (`python -m omok`) to play against it.

**Success criteria**

- On `omok9`, Gumbel search with ≤ 50 simulations per move learns where PUCT with 50 collapsed (Stage 5: 0.02 against the
  heuristic), and reaches the score of the 200-simulation main run (0.77) at a lower cost.
- On 15x15 freestyle and Renju, a network that beats the heuristic (score > 0.5, 100 games) and the pretrained models.
- A playable browser game against the trained network.

## Background

### Why visit counts need many simulations

PUCT's policy target is the visit distribution $\pi(a) = N(a) / \sum_b N(b)$. It is a good target only when there are many
more simulations than plausible moves: with $n$ simulations, at most $n$ moves get any visit, and a move's share of visits
is a noisy, integer-valued estimate of how good it is. Dirichlet noise makes it worse at small $n$: it spends simulations on
random moves. In Stage 5, 50 simulations on `omok9` (~70 legal moves) and 200 on 15x15 (~220 legal moves) both collapsed.
AlphaZero also has no guarantee that the visit distribution is better than the prior it started from when $n$ is small.

### Gumbel AlphaZero

Danihelka et al. (2022) replace both the root search and the target, keeping everything below the root close to MCTS.

**Sampling root moves without replacement.** Draw independent Gumbel noise $g(a) \sim \text{Gumbel}(0, 1)$ for every legal
move. The move $\arg\max_a [g(a) + \text{logits}(a)]$ is an exact sample from the prior $p = \text{softmax}(\text{logits})$
(the *Gumbel-max trick*), and the top-$m$ moves by $g + \text{logits}$ are a sample of $m$ distinct moves
without replacement. Only these $m$ moves ($m = 16$ here) are searched at the root.

**Sequential Halving.** The simulations are split into $\lceil \log_2 m \rceil$ phases. In each phase every remaining move
gets the same number of visits; at the end of the phase the worse half is dropped, ranked by

$$
g(a) + \text{logits}(a) + \sigma(\hat q(a)),
$$

where $\hat q(a)$ is the move's mean value so far and $\sigma$ a monotone transformation (below). The move left at the end
is played. With the Gumbel noise this is a *sample* from the improved policy, so no temperature or Dirichlet noise is
needed. In evaluation games $g = 0$ and the choice is deterministic.

**Improved policy target.** Every move, visited or not, gets a target:

$$
\pi'(a) = \text{softmax}\left(\text{logits}(a) + \sigma(\text{completed}\ q(a))\right), \qquad
\sigma(q) = (c_\text{visit} + \max_b N(b)) \, c_\text{scale} \, q,
$$

where *completed q* is $\hat q(a)$ for visited moves and $v_\text{mix}$ for the others,

$$
v_\text{mix} = \frac{1}{1 + \sum_b N(b)} \left( \hat v + \sum_b N(b) \cdot
\frac{\sum_{b:\,N(b) > 0} p(b)\, \hat q(b)}{\sum_{b:\,N(b) > 0} p(b)} \right),
$$

a mix of the network's value $\hat v$ of the position and the prior-weighted mean value of the visited moves. The
completed q values are rescaled to $[0, 1]$ over the legal moves before $\sigma$; $c_\text{visit} = 50$, $c_\text{scale} = 0.1$
(the defaults of DeepMind's `mctx` library). The paper proves that, in expectation, $\pi'$ is not worse than the prior
for any number of simulations. With zero simulations, $\pi' = p$: the target says "keep what you have".

**Below the root** the move is chosen deterministically: the one whose share of the node's visits is furthest below
$\pi'$, $\arg\max_a \left[\pi'(a) - N(a) / (1 + \sum_b N(b))\right]$.

### Playout cap randomization

KataGo (Wu, 2019) noticed that the *value* target $z$ needs many games, while the *policy* target needs deep searches,
and that most positions of a game only matter for $z$. So each move independently gets either a full search
(probability $p_\text{full} = 0.25$, $n_\text{full}$ simulations, with noise) or a fast one ($n_\text{fast}$ simulations, no noise).
Only full-search positions are recorded for training. The average cost per move drops from $n_\text{full}$ to
$p_\text{full} n_\text{full} + (1 - p_\text{full}) n_\text{fast}$, while every training target still comes from a full search.

### Tree reuse

After a move is played, the subtree under it already has statistics from the previous search (often a third or more of its
simulations). Keeping it as the next root and only adding the missing simulations saves that work. The kept root gets
fresh Dirichlet noise. Tree reuse doesn't combine with Gumbel search, whose Sequential Halving schedule starts from an
empty root.

## Implementation

| File | What it does |
|---|---|
| [`src/omok_rl/puct.py`](../../src/omok_rl/puct.py) | `PUCT` now takes per-position simulation counts and noise flags and optional kept roots; `Gumbel` (subclass), `halving_schedule`, `make_search` |
| [`src/omok_rl/selfplay.py`](../../src/omok_rl/selfplay.py) | `self_play` with `fast_simulations` / `full_prob` (playout cap randomization) and `reuse_tree` |
| [`src/omok_rl/alphazero.py`](../../src/omok_rl/alphazero.py) | New `AZConfig` fields: `search`, `considered`, `c_visit`, `c_scale`, `fast_simulations`, `full_prob`, `reuse_tree` |
| [`src/omok_rl/agents/alphazero.py`](../../src/omok_rl/agents/alphazero.py) | `AlphaZeroAgent(search='gumbel')` |
| [`scripts/stage6_alphazero.py`](../../scripts/stage6_alphazero.py) | All training runs of this document |
| [`tests/test_stage6.py`](../../tests/test_stage6.py) | Halving schedule, Gumbel tactics with an uninformed network, targets, tree reuse, playout cap randomization |

### Gumbel search in the batched PUCT code

`Gumbel` is a subclass of Stage 5's `PUCT` and changes four methods, so the batched search loop, the worker processes and
the GPU server are unchanged:

- `make_root` draws the Gumbel noise and precomputes the Sequential Halving **schedule**: entry $k$ is the visit count a root
  move must have to receive simulation $k$ (the trick from `mctx`). For $m = 16$ and 32 simulations it is
  `0 x16, 1 x8, 2 x4, 3 x4`: every considered move gets one visit, then the best 8 get a second, the best 4 a third, and so on.
  The root then needs no other state: at simulation $k$ it picks the best-scoring move among those with exactly
  `schedule[k]` visits.
- `select` uses the schedule at the root and $\arg\max [\pi' - N / (1 + \sum N)]$ below it.
- `target` returns $\pi'$ instead of the visit distribution, and `choose` the move left by Sequential Halving.

```python
def sigma_q(self, node):
    q = w / n for visited moves, v_mix for the others   # "completed" q, for the player to move
    q = (q - q.min()) / max(q.max() - q.min(), 1e-8)    # rescaled to [0, 1] over the legal moves
    return (self.c_visit + node.n.max()) * self.c_scale * q

def improved_policy(self, node):
    return softmax(log(node.prior) + self.sigma_q(node))
```

### Playout cap randomization and tree reuse

`PUCT.search_steps` now accepts one simulation count and one noise flag *per position*, so a batch of games can mix full
and fast searches. `self_play_steps` draws "full or fast" for every move of every game, and only records the full ones.
For tree reuse it keeps `root.children[move]` per game and passes the kept roots back into the next search, which skips their
network evaluation and only adds the missing simulations.

## Experiments

*(in progress)*

## References

- Danihelka, Guez, Schrittwieser & Silver, *Policy improvement by planning with Gumbel*, ICLR 2022 (Gumbel AlphaZero / MuZero)
- DeepMind `mctx` library, `gumbel_muzero_policy` and `qtransform_completed_by_mix_value` (the defaults used here)
- Wu, *Accelerating Self-Play Learning in Go*, 2019 (KataGo: playout cap randomization)
- Karnin, Koren & Somekh, *Almost optimal exploration in multi-armed bandits*, ICML 2013 (Sequential Halving)
- Kool, van Hoof & Welling, *Stochastic beams and where to find them: the Gumbel-top-k trick*, ICML 2019
