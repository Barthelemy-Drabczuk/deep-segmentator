"""Distributed training setup (stub — requires multi-GPU environment)."""
from typing import Any, Optional


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """
    Initialise PyTorch DistributedDataParallel process group.

    Args:
        rank: Rank of the current process.
        world_size: Total number of processes.
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU).

    Raises:
        NotImplementedError: Always (stub — requires multi-GPU environment).
    """
    # TODO: Call torch.distributed.init_process_group(backend, rank=rank,
    #         world_size=world_size) and wrap model with DDP.
    raise NotImplementedError(
        "setup_ddp() requires a multi-GPU environment. "
        "Implement DDP setup here."
    )


def cleanup_ddp() -> None:
    """Destroy the DistributedDataParallel process group."""
    # TODO: torch.distributed.destroy_process_group()
    raise NotImplementedError("cleanup_ddp() requires a multi-GPU environment.")


def wrap_model_ddp(model: Any, device_ids: Optional[list] = None) -> Any:
    """
    Wrap a model with DistributedDataParallel.

    Args:
        model: PyTorch model to wrap.
        device_ids: GPU IDs for this process. Defaults to [local_rank].

    Raises:
        NotImplementedError: Always (stub).
    """
    raise NotImplementedError("wrap_model_ddp() requires a multi-GPU environment.")
