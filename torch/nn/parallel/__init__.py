from .parallel_apply import parallel_apply
from .replicate import replicate
from .data_parallel import DataParallel, data_parallel
from .scatter_gather import scatter, gather
from .distributed import DistributedDataParallel
from .distributed_cpu import DistributedDataParallelCPU
import torch.nn.parallel.deprecated  # noqa: F401

__all__ = ['replicate', 'scatter', 'parallel_apply', 'gather', 'data_parallel',
           'DataParallel', 'DistributedDataParallel', 'DistributedDataParallelCPU']
