# scripts/train.py

import argparse
import torch
from pose.config import Config
from pose.engine.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config).raw

    torch.manual_seed(cfg.get("seed", 42))

    trainer = Trainer(cfg, device=args.device)
    trainer.run()


if __name__ == "__main__":
    main()
