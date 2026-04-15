import hydra
from omegaconf import DictConfig

import sv2m


@sv2m.main()
def main(config: DictConfig) -> None:
    trainer_config = config.train.trainer
    trainer = hydra.utils.instantiate(trainer_config, config)
    trainer.run()


if __name__ == "__main__":
    main()
