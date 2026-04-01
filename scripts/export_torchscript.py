#!/usr/bin/env python3

import argparse
import pathlib
import sys


def _normalize_state_dict(state):
    if not isinstance(state, dict):
        return state

    # Training used DataParallel, so checkpoints may store weights as
    # "module.xxx" while export loads a plain BVParsingNet instance.
    if all(key.startswith("module.") for key in state.keys()):
        return {key[len("module."):]: value for key, value in state.items()}

    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Export livox lane detector checkpoint to TorchScript.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to the source .pth checkpoint. Defaults to model/livox_lane_det.pth",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output TorchScript file. Defaults to model/livox_lane_det.ts",
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    import torch  # noqa: WPS433
    from network.bv_parsing_net import BVParsingNet  # noqa: WPS433

    checkpoint = pathlib.Path(args.checkpoint or (repo_root / "model" / "livox_lane_det.pth"))
    output = pathlib.Path(args.output or (repo_root / "model" / "livox_lane_det.ts"))

    model = BVParsingNet()
    state = torch.load(checkpoint, map_location="cpu")
    state_dict = _normalize_state_dict(state["state_dict"])
    model.load_state_dict(state_dict)
    model.eval()

    # Default config yields 840 x 1200 BEV tensors for 6-lidar inference.
    example = torch.zeros((1, 2, 840, 1200), dtype=torch.float32)
    traced = torch.jit.trace(model, example)
    traced.save(str(output))

    print(f"Saved TorchScript model to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
