from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from primed_ai.utils.reproducibility import set_seed
from primed_ai.utils.run_manifest import create_run_dir, save_run_metadata


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Smoke test for config loading and reproducibility logging."""
    set_seed(cfg.seed)

    run_dir = create_run_dir(cfg.run.output_dir)

    config_path = Path(run_dir) / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    save_run_metadata(
        run_dir,
        {
            "script": "scripts/run_config_smoke_test.py",
            "seed": cfg.seed,
            "device": cfg.device,
            "batch_size": cfg.batch_size,
        },
    )

    print(f"Smoke test complete. Run saved to: {run_dir}")


if __name__ == "__main__":
    main()