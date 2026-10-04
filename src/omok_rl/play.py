"""Play against a trained AlphaZero network in the browser, through the `omok` package's web UI (Stage 6).

The network is exported to ONNX once and then run with onnxruntime (already installed with `omok`), so playing needs
neither torch nor a GPU: the search (`omok_rl.puct`) and the evaluator below are numpy only.

    uv run python -m omok_rl.play runs/stage6/freestyle15/<run>.pt --rule freestyle --simulations 400

The web UI is `omok.OmokGame` (15x15, freestyle or Renju). It calls `agent(state, player)` for the computer's move and
`agent.get_probs(state, player)` for the move-probability overlay; this agent answers both from one search of the game's
own environment (bound with `bind`), so the overlay shows the search's visit distribution.
"""

import argparse
from pathlib import Path

import numpy as np
from omok.env import BoardGame

from omok_rl.puct import make_search
from omok_rl.symmetry import permutations


def export_onnx(checkpoint: str | Path, out: str | Path, in_channels: int | None = None) -> Path:
    """Export an AlphaZero checkpoint to ONNX: input `obs` (batch, planes, size, size), outputs `logits` and `value`.

    Batch and board size are dynamic (the network is fully convolutional). `in_channels` loads the network for a board
    with more input planes (e.g. an omok9 network on 15x15 Renju; the extra planes get zero weights)."""
    import torch

    from omok_rl.nets import AlphaZeroNet

    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    c = ckpt['config']
    net = AlphaZeroNet(in_channels or ckpt['in_channels'], c['channels'], c['blocks'])
    net.load_transfer(ckpt['state_dict'])
    net.eval()
    x = torch.zeros(1, net.in_channels, ckpt['size'], ckpt['size'])
    out = Path(out)
    torch.onnx.export(net, (x,), out, input_names=['obs'], output_names=['logits', 'value'], dynamo=False,
                      dynamic_axes={'obs': {0: 'batch', 2: 'size', 3: 'size'}, 'logits': {0: 'batch', 1: 'cells'},
                                    'value': {0: 'batch'}})
    return out


class OnnxEvaluator:
    """Like `nets.NetEvaluator` (same call signature), with onnxruntime on the CPU and numpy for the symmetries."""

    def __init__(self, path: str | Path, size: int, symmetries: bool = True, seed: int | None = None):
        import onnxruntime

        self.session = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        self.planes = self.session.get_inputs()[0].shape[1]
        self.perms, self.inverse = permutations(size)
        self.symmetries, self.rng = symmetries, np.random.default_rng(seed)
        self.calls = self.positions = 0

    def __call__(self, obs: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, c = obs.shape[:2]
        self.calls += 1
        self.positions += n
        x = obs.reshape(n, c, -1).astype(np.float32)
        if c < self.planes:  # e.g. a Renju network (5 planes) asked about a freestyle position (4)
            x = np.concatenate([x, np.zeros((n, self.planes - c, x.shape[2]), np.float32)], 1)
        codes = self.rng.integers(8, size=n) if self.symmetries else np.zeros(n, np.int64)
        x = np.take_along_axis(x, self.perms[codes][:, None], 2).reshape(n, self.planes, *obs.shape[2:])
        logits, values = self.session.run(None, {'obs': x})
        logits = np.take_along_axis(logits, self.inverse[codes], 1)
        logits = np.where(masks, logits, -np.inf)
        probs = np.exp(logits - logits.max(1, keepdims=True))
        return probs / probs.sum(1, keepdims=True), values.astype(np.float32)


class WebAgent:
    """An AlphaZero player for `omok.OmokGame`: one search per position, shared by the overlay and the move."""

    def __init__(self, evaluator, simulations: int = 400, search: str = 'puct', leaves: int = 8, seed: int = 0):
        self.search = make_search(search, evaluator, seed=seed)
        self.simulations, self.leaves = simulations, leaves
        self.rng = np.random.default_rng(seed)
        self.env: BoardGame | None = None
        self.cache = (None, None)  # (move history, root)

    def bind(self, env: BoardGame) -> 'WebAgent':
        self.env = env
        return self

    def root(self):
        key = tuple(self.env.get_move_history())
        if self.cache[0] != key:
            leaves = self.leaves if self.search.__class__.__name__ == 'PUCT' else 1  # Gumbel: one leaf at a time
            self.cache = key, self.search.search([self.env], self.simulations, leaves=leaves)[0]
        return self.cache[1]

    def get_probs(self, state, player) -> np.ndarray:
        root = self.root()
        probs = np.zeros(self.env.size ** 2, np.float32)
        probs[root.moves] = self.search.target(root, self.env.size)[root.moves] if root.n.sum() else root.prior
        return probs

    def __call__(self, state, player) -> int:
        root = self.root()
        if root.n.sum() == 0:
            return int(root.moves[np.argmax(root.prior)])
        return self.search.choose(root, self.rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', type=Path, help='an AlphaZero checkpoint (.pt, exported to .onnx next to it) or .onnx')
    parser.add_argument('--rule', choices=('freestyle', 'renju'), default='freestyle')
    parser.add_argument('--simulations', type=int, default=400, help='0: play the raw network')
    parser.add_argument('--search', choices=('puct', 'gumbel'), default='puct')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    import omok

    path = args.model
    if path.suffix == '.pt':
        onnx_path = path.with_name(f'{path.stem}-{args.rule}.onnx')
        if not onnx_path.exists():
            export_onnx(path, onnx_path, in_channels=omok.Omok(rule=args.rule).get_observation().shape[0])
            print(f'exported {onnx_path}')
        path = onnx_path
    agent = WebAgent(OnnxEvaluator(path, 15), args.simulations, args.search)
    game = omok.OmokGame(agent=agent, rule=args.rule)
    agent.bind(game.env)
    game.run(args.host, args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()
