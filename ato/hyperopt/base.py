import math
import uuid
from copy import deepcopy as dcp
from itertools import product

import numpy as np

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    dist = None

from ato.adict import ADict


class HyperOpt:
    def __init__(self, scope, search_spaces, tracker=None, mode='max'):
        if mode not in ('min', 'max'):
            raise ValueError('mode must be either "min" or "max".')
        self.scope = scope
        self.search_spaces = search_spaces
        self.config = scope.config.clone()
        self.tracker = tracker
        self.mode = mode
        self.config.__hyperopt_id__ = self.get_hyperopt_id()

    @classmethod
    def get_hyperopt_id(cls):
        return str(uuid.uuid4())

    def main(self, func):
        raise NotImplementedError()


class DistributedMixIn:
    def __init__(self, rank=0, world_size=1, backend='pytorch'):
        if not TORCH_AVAILABLE:
            raise RuntimeError('DistributedMixin requires PyTorch to be installed.')
        self.rank = rank
        self.world_size = world_size
        self.backend = backend

    @property
    def is_root(self):
        return self.rank == 0

    def broadcast_object_from_root(self, obj):
        if self.backend == 'pytorch':
            obj = [obj]
            dist.broadcast_object_list(obj)
            obj = obj[0]
        else:
            raise ValueError(f'Unsupported backend: {self.backend}')
        return obj

    def all_gather_object(self, obj):
        if self.backend == 'pytorch':
            gathered_objects = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_objects, obj)
        else:
            raise ValueError(f'Unsupported backend: {self.backend}')
        return gathered_objects

    def destroy(self):
        if self.backend == 'pytorch':
            if dist.is_initialized():
                dist.destroy_process_group()
        else:
            raise ValueError(f'Unsupported backend: {self.backend}')

    def get_hyperopt_id(self):
        return self.broadcast_object_from_root(str(uuid.uuid4()))


class DistributedHyperOpt(DistributedMixIn, HyperOpt):
    def __init__(
        self,
        scope,
        search_spaces,
        tracker=None,
        mode='max',
        rank=0,
        world_size=1,
        backend='pytorch'
    ):
        HyperOpt.__init__(self, scope, search_spaces, tracker, mode)
        DistributedMixIn.__init__(self, rank, world_size, backend)


class GridSpaceMixIn:
    @classmethod
    def prepare_distributions(cls, base_config, search_spaces):
        sampling_spaces = ADict()
        for param_name, search_space in search_spaces.items():
            if 'param_type' not in search_space:
                raise KeyError(f'param_type for parameter {param_name} is not defined at search_spaces.')
            param_type = search_space['param_type'].upper()
            if param_type == 'INTEGER':
                start, stop = search_space.param_range
                space_type = search_space.get('space_type', 'LINEAR')
                if space_type == 'LINEAR':
                    optim_space = np.linspace(
                        start=start,
                        stop=stop,
                        num=search_space.num_samples,
                        dtype=np.int64
                    ).tolist()
                elif space_type == 'LOG':
                    base = search_space.get('base', 2)
                    optim_space = np.logspace(
                        start=math.log(start, base),
                        stop=math.log(stop, base),
                        num=search_space.num_samples,
                        dtype=np.int64,
                        base=base
                    ).tolist()
                else:
                    raise ValueError(f'Invalid space_type: {space_type}')
            elif param_type == 'FLOAT':
                start, stop = search_space.param_range
                space_type = search_space.get('space_type', 'LINEAR')
                if space_type == 'LINEAR':
                    optim_space = np.linspace(
                        start=start,
                        stop=stop,
                        num=search_space.num_samples,
                        dtype=np.float32
                    ).tolist()
                elif space_type == 'LOG':
                    base = search_space.get('base', 10)
                    optim_space = np.logspace(
                        start=math.log(start, base),
                        stop=math.log(stop, base),
                        num=search_space.num_samples,
                        dtype=np.float32,
                        base=base
                    ).tolist()
                else:
                    raise ValueError(f'Invalid space_type: {space_type}')
            elif param_type == 'CATEGORY':
                optim_space = search_space.categories
            else:
                raise ValueError(f'Unknown param_type for parameter {param_name}; {param_type}')
            sampling_spaces[param_name] = optim_space
        grid_space = [ADict(zip(sampling_spaces.keys(), values)) for values in product(*sampling_spaces.values())]
        distributions = [
            dcp(base_config).update(**partial_config)
            for index, partial_config in enumerate(grid_space)
        ]
        return distributions
