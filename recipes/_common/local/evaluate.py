import hydra
from omegaconf import DictConfig

import sv2m


@sv2m.main()
def main(config: DictConfig) -> None:
    evaluator_config = config.evaluate.evaluator
    evaluator = hydra.utils.instantiate(evaluator_config, config)
    evaluator.run()


if __name__ == "__main__":
    main()
