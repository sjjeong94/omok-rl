"""Export the Stage 6 networks for the `omok` package: models/alphazero-renju.onnx and models/alphazero-freestyle.onnx.

The networks ignore the last-move plane (checked in scripts/package_checks.py: no loss of strength), so they play from
`(state, player)` alone, like `omok.OmokAgent`. Each file stores where it came from in its ONNX metadata.

    uv run python scripts/export_for_omok.py --out ../omok/models
"""

import argparse
import hashlib
import json
from pathlib import Path

from stage5_alphazero import git_commit

from omok_rl.play import export_onnx

MODELS = {  # rule -> (checkpoint, results to record)
    'renju': ('runs/stage6/renju15/gumbel50-transfer-seed0.pt',
              'Renju, vs b.onnx (pretrained1), 100 games: 0.76 raw policy, 0.91 with PUCT 200 simulations'),
    'freestyle': ('runs/stage6/freestyle15/gumbel50-transfer-seed0.pt',
                  'freestyle, vs b.onnx (pretrained1), 100 games: 0.82 raw policy, 0.88 with PUCT 200 simulations'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    for rule, (ckpt, results) in MODELS.items():
        path = export_onnx(ckpt, args.out / f'alphazero-{rule}.onnx', in_channels=5, ignore_last_move=True, metadata={
            'source': 'https://github.com/sjjeong94/omok-rl (Stage 6, Gumbel AlphaZero)', 'checkpoint': ckpt,
            'rule': rule, 'commit': commit, 'results': results,
            'inputs': 'obs (batch, 5, 15, 15) float32: own stones, opponent stones, 1 if black to play, '
                      'last move (ignored), forbidden points (Renju, black to play)',
            'outputs': 'logits (batch, 225) over all cells (mask illegal moves), value (batch,) in [-1, 1] '
                       'for the player to move'})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(json.dumps({'file': str(path), 'bytes': path.stat().st_size, 'sha256': digest}))


if __name__ == '__main__':
    main()
