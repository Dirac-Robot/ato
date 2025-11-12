import os
import random
import unittest

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    dist = None

from ato.adict import ADict
from ato.hyperopt.hyperband import HyperBand, DistributedHyperBand
from ato.scope import Scope


class TestHyperBand(unittest.TestCase):
    def setUp(self):
        Scope.initialize_registry()
        """Set up test fixtures for HyperBand tests."""
        self.scope = Scope(name='unit_test_config')
        self.search_spaces = ADict(
            lr=ADict(param_type='FLOAT', param_range=(0.0001, 0.1), num_samples=8, space_type='LOG'),
            batch_size=ADict(param_type='INTEGER', param_range=(1, 64), num_samples=10, space_type='LOG'),
            model_type=ADict(param_type='CATEGORY', categories=('resnet50', 'resnet101', 'swin_s', 'vit-s'))
        )
        self.hyperband = HyperBand(self.scope, self.search_spaces, 0.3, 4, None)

    def test_hyperband_basic_execution(self):
        @self.hyperband.main
        def main(unit_test_config):
            metric = random.random()
            return metric

        results = main()
        expected_best = max(results.logs[-1], key=lambda item: item.__metric__).__metric__
        self.assertEqual(results.metric, expected_best)

    def test_hyperband_num_generations(self):
        @self.hyperband.main
        def main(unit_test_config):
            metric = random.random()
            return metric

        main()
        num_generations = self.hyperband.num_generations()
        self.assertIsInstance(num_generations, int)
        self.assertGreater(num_generations, 0)

    def test_hyperband_optimized_steps(self):
        optimized_steps = self.hyperband.compute_optimized_initial_training_steps(24)
        self.assertTrue(all(map(lambda step: isinstance(step, (int, float)) and step > 0, optimized_steps)))


@unittest.skipIf(not TORCH_AVAILABLE, 'PyTorch is not available')
class TestDistributedHyperBand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up distributed environment for all tests."""
        cls.rank = int(os.environ.get('RANK', '0'))
        cls.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        cls.world_size = int(os.environ.get('WORLD_SIZE', '1'))

        # Only initialize distributed if CUDA is available and world_size > 1
        cls.distributed_available = torch.cuda.is_available() and cls.world_size > 1

        if cls.distributed_available:
            torch.cuda.set_device(cls.local_rank)
            dist.init_process_group(
                backend='nccl',
                init_method='env:',
                rank=cls.rank,
                world_size=cls.world_size
            )
            cls.rank = dist.get_rank()
            cls.world_size = dist.get_world_size()

    @classmethod
    def tearDownClass(cls):
        if cls.distributed_available and dist.is_initialized():
            dist.destroy_process_group()

    def setUp(self):
        Scope.initialize_registry()
        self.scope = Scope(name='unit_test_config')
        self.search_spaces = ADict(
            lr=ADict(param_type='FLOAT', param_range=(0.0001, 0.1), num_samples=10, space_type='LOG'),
            batch_size=ADict(param_type='INTEGER', param_range=(1, 64), num_samples=7, space_type='LOG'),
            model_type=ADict(param_type='CATEGORY', categories=('resnet50', 'resnet101', 'swin_s'))
        )
        rank = self.rank if self.distributed_available else 0
        world_size = self.world_size if self.distributed_available else 1

        self.hyperband = DistributedHyperBand(
            self.scope,
            self.search_spaces,
            0.3,
            4,
            None,
            mode='max',
            rank=rank,
            world_size=world_size
        )

    def test_distributed_hyperband_basic_execution(self):
        if not self.distributed_available:
            self.skipTest('Distributed environment not available')

        @self.hyperband.main
        def main(unit_test_config):
            metric = random.random()
            return metric

        results = main()
        expected_best = max(results.logs[-1], key=lambda item: item.__metric__).__metric__
        self.assertEqual(results.metric, expected_best)

    def test_distributed_hyperband_non_distributed_mode(self):
        @self.hyperband.main
        def main(unit_test_config):
            metric = random.random()
            return metric

        results = main()
        self.assertIsNotNone(results)
        self.assertTrue(hasattr(results, 'metric'))
        self.assertTrue(hasattr(results, 'logs'))


if __name__ == '__main__':
    unittest.main()