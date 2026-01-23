"""Project training entrypoint.

This is the thin CLI wrapper around :class:`pose.engine.trainer.Trainer`.

High-level flow:
    1) Load YAML config via :class:`pose.config.Config`
    2) Seed PyTorch RNG (for basic reproducibility)
    3) Create Trainer (datasets/model/loss/optim/scheduler are all built from config)
    4) Run training loop, optionally resuming from a checkpoint

The training behavior is entirely driven by the YAML (see `configs/`).

Notes:
    - `--device cuda` will fall back to CPU if CUDA is unavailable.
    - `--resume` expects a checkpoint produced by Trainer (e.g. best.pth / epoch_XX.pth).
"""

import argparse
import torch
from pose.config import Config
from pose.engine.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pth to resume from")
    args = parser.parse_args()

    # Config is loaded as a plain dict (cfg.raw) so the Trainer can stay framework-agnostic.
    cfg = Config.from_yaml(args.config).raw
    cfg["_config_path"] = args.config  # keep if you use config-copying

    # Seed only torch here; if you need full determinism also seed numpy/random
    # and enable deterministic algorithms (often slower).
    torch.manual_seed(cfg.get("seed", 42))

    # Trainer handles:
    # - dataset(s) construction
    # - model construction via registries
    # - loss wiring
    # - checkpointing + reporting
    trainer = Trainer(cfg, device=args.device)
    trainer.run(resume_path=args.resume)

if __name__ == "__main__":
    main()
