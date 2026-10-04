"""Networks that map board planes (C, size, size) to one output per cell.

- Q-networks (Stage 2): Q(s, a) for every move a.
- Policy-value network (Stage 3): logits of pi(a | s) for every move a, and one state value V(s).

Values are for the player to move and squashed to [-1, 1] with tanh, the range of the game's returns.
"""

import numpy as np
import torch
from torch import nn

from omok_rl.symmetry import permutations


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return torch.relu(x + self.conv2(torch.relu(self.conv1(x))))


class ConvQNet(nn.Module):
    """Fully convolutional: the same weights look at every part of the board, and it works for any board size.

    With `dueling`, Q(s, a) = V(s) + A(s, a) - mean_a A(s, a), where V comes from a global-pooled head.
    """

    def __init__(self, in_channels: int, channels: int = 64, blocks: int = 3, dueling: bool = False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, channels, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.advantage = nn.Conv2d(channels, 1, 1)
        self.dueling = dueling
        if dueling:
            self.value = nn.Sequential(nn.Linear(channels, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.blocks(self.stem(x))
        q = self.advantage(h).flatten(1)
        if self.dueling:
            v = self.value(h.mean(dim=(2, 3)))
            q = v + q - q.mean(dim=1, keepdim=True)
        return torch.tanh(q)


class MLPQNet(nn.Module):
    """Fully connected baseline: every cell has its own weights, so nothing learned in one place transfers to another."""

    def __init__(self, in_channels: int, size: int, hidden: int = 512, layers: int = 3):
        super().__init__()
        dims = [in_channels * size * size] + [hidden] * layers
        mods = []
        for a, b in zip(dims, dims[1:]):
            mods += [nn.Linear(a, b), nn.ReLU()]
        self.net = nn.Sequential(nn.Flatten(), *mods, nn.Linear(hidden, size * size))

    def forward(self, x):
        return torch.tanh(self.net(x))


class PolicyValueNet(nn.Module):
    """Shared convolutional body (as in `ConvQNet`) with two heads.

    - policy: a 1x1 convolution gives one logit per cell (illegal moves are masked by the caller).
    - value: global average pooling, a small MLP, and tanh give V(s) in [-1, 1].
    """

    def __init__(self, in_channels: int, channels: int = 64, blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, channels, 3, padding=1), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy = nn.Conv2d(channels, 1, 1)
        self.value = nn.Sequential(nn.Linear(channels, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.policy(h).flatten(1), torch.tanh(self.value(h.mean(dim=(2, 3))))[:, 0]


def make_qnet(arch: str, in_channels: int, size: int, dueling: bool = False, channels: int = 64,
              blocks: int = 3) -> nn.Module:
    if arch == 'cnn':
        return ConvQNet(in_channels, channels, blocks, dueling)
    if arch == 'mlp':
        if dueling:
            raise ValueError('dueling is only implemented for the cnn')
        return MLPQNet(in_channels, size)
    raise ValueError(f'unknown arch {arch!r}')


class BNResidualBlock(nn.Module):
    """conv-BN-ReLU-conv-BN plus the input, then ReLU (the AlphaGo Zero block)."""

    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels),
                                  nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels))

    def forward(self, x):
        return torch.relu(x + self.body(x))


class AlphaZeroNet(nn.Module):
    """ResNet policy-value network (Stage 5), fully convolutional so that one network plays on any board size.

    Differences from `PolicyValueNet`: batch normalization, more blocks, larger heads, and a constant plane of ones
    added to the input. With zero padding, the ones plane is what tells the network where the board ends: an empty
    cell and a cell off the board otherwise look the same (all planes 0) when White is to move.

    - policy: two 1x1 convolutions give one logit per cell (illegal moves are masked by the caller).
    - value: a 1x1 convolution, global average pooling, a small MLP, and tanh give V(s) in [-1, 1].
    """

    def __init__(self, in_channels: int, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.in_channels = in_channels
        self.stem = nn.Sequential(nn.Conv2d(in_channels + 1, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*[BNResidualBlock(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                    nn.Conv2d(32, 1, 1))
        self.value_conv = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.value_fc = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        ones = torch.ones_like(x[:, :1])
        h = self.blocks(self.stem(torch.cat([x, ones], 1)))
        value = torch.tanh(self.value_fc(self.value_conv(h).mean(dim=(2, 3))))[:, 0]
        return self.policy(h).flatten(1), value

    def load_transfer(self, state_dict: dict):
        """Load weights trained with fewer input planes (e.g. omok9's 4 into renju15's 5): the extra planes get zero weights.

        The ones plane is always the last input channel, so the stem's input weights are copied plane by plane."""
        state_dict = dict(state_dict)
        w = state_dict['stem.0.weight']
        old = w.shape[1] - 1
        if old != self.in_channels:
            new = torch.zeros(w.shape[0], self.in_channels + 1, *w.shape[2:], dtype=w.dtype)
            new[:, :old] = w[:, :old]
            new[:, -1] = w[:, -1]
            state_dict['stem.0.weight'] = new
        self.load_state_dict(state_dict)


def permute_planes(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Apply one symmetry per position: x (B, C, size, size), p (B, size*size) rows of `symmetry.permutations`."""
    b, c = x.shape[:2]
    return x.reshape(b, c, -1).gather(2, p[:, None].expand(b, c, -1)).reshape(x.shape)


class NetEvaluator:
    """Evaluates batches of positions with a policy-value network.

    With `symmetries`, each position is shown to the network through one of the 8 rotations/reflections, drawn at
    random (as in AlphaGo Zero), and the policy is mapped back. Returns legal-move probabilities and values.

    On a GPU, small batches cost about as much as large ones: the time goes into launching the ~40 kernels of one
    forward pass, not into computing. So the whole evaluation is recorded once per batch size as a **CUDA graph**
    and replayed with one launch; batches are padded up to the next power of two to reuse a few graphs.
    The graphs read the network's weights in place, so they stay valid while the network trains.
    """

    def __init__(self, net: torch.nn.Module, device: str | torch.device, size: int, symmetries: bool = True,
                 seed: int | None = None, graphs: bool | None = None, max_graph_batch: int = 2048):
        self.net, self.device, self.symmetries = net, torch.device(device), symmetries
        self.rng = np.random.default_rng(seed)
        perms, inverse = permutations(size)
        self.perms = torch.as_tensor(perms, device=self.device)
        self.inverse = torch.as_tensor(inverse, device=self.device)
        if self.device.type == 'cuda':  # NHWC is faster for these convolutions (and changes nothing else)
            net.to(memory_format=torch.channels_last)
        self.graphs = self.device.type == 'cuda' if graphs is None else graphs
        self.max_graph_batch = max_graph_batch
        self.captured = {}  # (batch, planes) -> (graph, static inputs, static outputs)
        self.calls = self.positions = 0

    def compute(self, x: torch.Tensor, codes: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.symmetries:
            x = permute_planes(x, self.perms[codes])
        if self.device.type == 'cuda':
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda',
                            cache_enabled=False):
            logits, values = self.net(x)
        logits = logits.float()
        if self.symmetries:
            logits = logits.gather(1, self.inverse[codes])
        return torch.softmax(logits.masked_fill(~mask, -torch.inf), 1), values.float()

    def capture(self, batch: int, shape: tuple):
        x = torch.zeros(batch, *shape, device=self.device)
        codes = torch.zeros(batch, dtype=torch.long, device=self.device)
        mask = torch.ones(batch, shape[1] * shape[2], dtype=torch.bool, device=self.device)
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):  # warm up (cuDNN picks its algorithms) outside the graph
            for _ in range(3):
                self.compute(x, codes, mask)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.compute(x, codes, mask)
        return graph, (x, codes, mask), out

    @torch.inference_mode()
    def __call__(self, obs: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """obs (B, C, size, size), masks (B, size*size) -> probabilities (B, size*size) and values (B,)."""
        n = len(obs)
        self.calls += 1
        self.positions += n
        codes = self.rng.integers(8, size=n) if self.symmetries else np.zeros(n, np.int64)
        if not self.graphs or n > self.max_graph_batch:
            probs, values = self.compute(torch.as_tensor(obs, device=self.device),
                                         torch.as_tensor(codes, device=self.device),
                                         torch.as_tensor(masks, device=self.device))
            return probs.cpu().numpy(), values.cpu().numpy()
        batch = 1 << max(4, (n - 1).bit_length())
        key = (batch, obs.shape[1:])
        if key not in self.captured:
            self.captured[key] = self.capture(batch, obs.shape[1:])
        graph, (x, c, m), (probs, values) = self.captured[key]
        x[:n].copy_(torch.from_numpy(obs))
        c[:n].copy_(torch.from_numpy(codes))
        m[:n].copy_(torch.from_numpy(masks))
        graph.replay()
        return probs[:n].cpu().numpy(), values[:n].cpu().numpy()
