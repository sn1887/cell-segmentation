from __future__ import annotations

import hydra
from omegaconf import DictConfig

from cellseg_challenge.runner import train_from_config


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    best_path = train_from_config(cfg)
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()

