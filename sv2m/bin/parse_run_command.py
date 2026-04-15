import torch
from omegaconf import DictConfig

import sv2m


@sv2m.main(config_name="parse_run_command")
def main(config: DictConfig) -> None:
    """Determine command to run script.

    If multiple GPSs are available, ``torchrun --nnodes=...`` is returned to stdout.
    Otherwise, ``python`` is returned.

    """
    if torch.cuda.is_available():
        nproc_per_node = torch.cuda.device_count()

        if nproc_per_node == 0:
            raise RuntimeError("Despite availability of CUDA, no devices are found.")
        elif nproc_per_node > 1:
            is_distributed = True
        else:
            is_distributed = False
    else:
        nproc_per_node = 0
        is_distributed = False

    if is_distributed:
        nnodes = 1

        cmd = "torchrun"
        cmd += " --standalone"
        cmd += f" --nnodes={nnodes} --nproc-per-node={nproc_per_node}"
    else:
        cmd = "python"

    print(cmd)


if __name__ == "__main__":
    main()
