"""
Extreme Edge Case Tests for Scope Features
Tests priority ordering, lazy evaluation, chain_with, freeze/defrost, etc.
Aligned with ato's actual behavior.
"""
import unittest
import sys
from ato.scope import Scope, parse_args_pythonic
from ato.adict import ADict


class PriorityEdgeCaseTest(unittest.TestCase):
    """Priority ordering edge cases"""

    def setUp(self):
        Scope.initialize_registry()
        Scope.parsed = False
        Scope.stored_arguments = None
        self.config = ADict()
        self.scope = Scope(config=self.config, name='config')
        sys.argv = ['test.py']

    def test_negative_priority(self):
        @self.scope.observe(priority=-10)
        def neg_priority(config):
            config.val = 'negative'

        @self.scope.observe(priority=0)
        def zero_priority(config):
            config.val = 'zero'

        self.scope.assign('neg_priority')
        self.scope.assign('zero_priority')
        self.scope.apply()
        self.assertEqual(self.config.val, 'zero')

    def test_same_priority_order_by_assign(self):
        @self.scope.observe(priority=5)
        def view_a(config):
            config.val = 'A'

        @self.scope.observe(priority=5)
        def view_b(config):
            config.val = 'B'

        self.scope.assign('view_a')
        self.scope.assign('view_b')
        self.scope.apply()
        self.assertEqual(self.config.val, 'B')

    def test_very_large_priority(self):
        @self.scope.observe(priority=1000000)
        def high_priority(config):
            config.val = 'high'

        @self.scope.observe(priority=1)
        def low_priority(config):
            config.val = 'low'

        self.scope.assign('low_priority')
        self.scope.assign('high_priority')
        self.scope.apply()
        self.assertEqual(self.config.val, 'high')

    def test_priority_vs_cli(self):
        @self.scope.observe(priority=1000)
        def very_high(config):
            config.val = 'view_val'

        sys.argv = ['test.py', 'val:=cli_val', 'very_high']
        parse_args_pythonic()
        self.scope.apply()
        self.assertEqual(self.config.val, 'cli_val')

    def test_priority_order_result(self):
        @self.scope.observe(priority=3)
        def p3(config):
            config.order_3 = True

        @self.scope.observe(priority=1)
        def p1(config):
            config.order_1 = True

        @self.scope.observe(priority=2)
        def p2(config):
            config.order_2 = True

        self.scope.assign('p3')
        self.scope.assign('p1')
        self.scope.assign('p2')
        self.scope.apply()
        self.assertTrue(self.config.order_1)
        self.assertTrue(self.config.order_2)
        self.assertTrue(self.config.order_3)


class LazyEdgeCaseTest(unittest.TestCase):
    """Lazy evaluation edge cases"""

    def setUp(self):
        Scope.initialize_registry()
        Scope.parsed = False
        Scope.stored_arguments = None
        self.config = ADict()
        self.scope = Scope(config=self.config, name='config')

    def test_lazy_sees_cli_args(self):
        @self.scope.observe(default=True)
        def defaults(config):
            config.base = 10

        @self.scope.observe(lazy=True, default=True)
        def computed(config):
            config.result = config.base*2

        sys.argv = ['test.py', 'base=100']
        parse_args_pythonic()
        self.scope.apply()
        self.assertEqual(self.config.base, 100)
        self.assertEqual(self.config.result, 200)

    def test_lazy_priority_order(self):
        @self.scope.observe(lazy=True, priority=2)
        def lazy_high(config):
            config.lazy_high_done = True

        @self.scope.observe(lazy=True, priority=1)
        def lazy_low(config):
            config.lazy_low_done = True

        sys.argv = ['test.py', 'lazy_high', 'lazy_low']
        parse_args_pythonic()
        self.scope.apply()
        self.assertTrue(self.config.lazy_low_done)
        self.assertTrue(self.config.lazy_high_done)

    def test_lazy_after_non_lazy(self):
        @self.scope.observe(priority=100)
        def non_lazy_high(config):
            config.non_lazy_done = True

        @self.scope.observe(lazy=True, priority=1)
        def lazy_low(config):
            config.lazy_done = config.non_lazy_done

        self.scope.assign('non_lazy_high')
        self.scope.assign('lazy_low')
        sys.argv = ['test.py']
        parse_args_pythonic()
        self.scope.apply()
        self.assertTrue(self.config.non_lazy_done)
        self.assertTrue(self.config.lazy_done)

    def test_lazy_with_string_cli(self):
        @self.scope.observe(default=True)
        def defaults(config):
            config.model = 'default'

        @self.scope.observe(lazy=True, default=True)
        def computed(config):
            if config.model == 'resnet':
                config.layers = 50
            else:
                config.layers = 10

        sys.argv = ['test.py', 'model:=resnet']
        parse_args_pythonic()
        self.scope.apply()
        self.assertEqual(self.config.model, 'resnet')
        self.assertEqual(self.config.layers, 50)


class ChainWithEdgeCaseTest(unittest.TestCase):
    """chain_with edge cases"""

    def setUp(self):
        Scope.initialize_registry()
        Scope.parsed = False
        Scope.stored_arguments = None
        self.config = ADict()
        self.scope = Scope(config=self.config, name='config')
        sys.argv = ['test.py']

    def test_single_chain_result(self):
        @self.scope.observe()
        def base(config):
            config.base_done = True

        @self.scope.observe(chain_with='base')
        def derived(config):
            config.derived_done = config.base_done

        self.scope.assign('derived')
        self.scope.apply()
        self.assertTrue(self.config.base_done)
        self.assertTrue(self.config.derived_done)

    def test_multiple_chain(self):
        @self.scope.observe()
        def a(config):
            config.a_done = True

        @self.scope.observe()
        def b(config):
            config.b_done = True

        @self.scope.observe(chain_with=['a', 'b'])
        def c(config):
            config.c_done = config.a_done and config.b_done

        self.scope.assign('c')
        self.scope.apply()
        self.assertTrue(self.config.a_done)
        self.assertTrue(self.config.b_done)
        self.assertTrue(self.config.c_done)

    def test_deep_chain_result(self):
        @self.scope.observe()
        def level1(config):
            config.level1 = 1

        @self.scope.observe(chain_with='level1')
        def level2(config):
            config.level2 = config.level1 + 1

        @self.scope.observe(chain_with='level2')
        def level3(config):
            config.level3 = config.level2 + 1

        @self.scope.observe(chain_with='level3')
        def level4(config):
            config.level4 = config.level3 + 1

        self.scope.assign('level4')
        self.scope.apply()
        self.assertEqual(self.config.level1, 1)
        self.assertEqual(self.config.level2, 2)
        self.assertEqual(self.config.level3, 3)
        self.assertEqual(self.config.level4, 4)

    def test_chain_with_priority(self):
        @self.scope.observe(priority=10)
        def base_high(config):
            config.val = 'base'

        @self.scope.observe(priority=1, chain_with='base_high')
        def derived_low(config):
            config.val = 'derived'

        self.scope.assign('derived_low')
        self.scope.apply()
        self.assertEqual(self.config.val, 'base')

    def test_lazy_chain_result(self):
        @self.scope.observe()
        def base(config):
            config.base_done = True

        @self.scope.observe(lazy=True, chain_with='base')
        def lazy_derived(config):
            config.lazy_done = config.base_done

        sys.argv = ['test.py', 'lazy_derived']
        parse_args_pythonic()
        self.scope.apply()
        self.assertTrue(self.config.base_done)
        self.assertTrue(self.config.lazy_done)


class DefaultViewEdgeCaseTest(unittest.TestCase):
    """default view edge cases"""

    def setUp(self):
        Scope.initialize_registry()
        Scope.parsed = False
        Scope.stored_arguments = None
        self.config = ADict()
        self.scope = Scope(config=self.config, name='config')
        sys.argv = ['test.py']

    def test_multiple_defaults(self):
        @self.scope.observe(default=True, priority=1)
        def default1(config):
            config.a = 1

        @self.scope.observe(default=True, priority=2)
        def default2(config):
            config.b = 2

        self.scope.apply()
        self.assertEqual(self.config.a, 1)
        self.assertEqual(self.config.b, 2)

    def test_default_overridden_by_cli(self):
        @self.scope.observe(default=True)
        def defaults(config):
            config.val = 'default'

        sys.argv = ['test.py', 'val:=override']
        parse_args_pythonic()
        self.scope.apply()
        self.assertEqual(self.config.val, 'override')

    def test_default_and_explicit(self):
        @self.scope.observe(default=True)
        def defaults(config):
            config.a = 1

        @self.scope.observe()
        def explicit(config):
            config.a = 100

        self.scope.assign('explicit')
        self.scope.apply()
        self.assertEqual(self.config.a, 100)


class FreezDefrostEdgeCaseTest(unittest.TestCase):
    """freeze/defrost edge cases - ADict silently ignores writes when frozen"""

    def setUp(self):
        Scope.initialize_registry()
        self.config = ADict(val=10)
        self.scope = Scope(config=self.config, name='config')

    def test_freeze_ignores_write(self):
        self.config.freeze()
        self.config.val = 20
        self.assertEqual(self.config.val, 10)
        self.config.defrost()
        self.config.val = 30
        self.assertEqual(self.config.val, 30)

    def test_nested_freeze(self):
        self.config.nested = ADict(inner=5)
        self.config.freeze()
        self.config.nested.inner = 10
        self.assertEqual(self.config.nested.inner, 5)
        self.config.defrost()
        self.config.nested.inner = 15
        self.assertEqual(self.config.nested.inner, 15)


class PauseActivateEdgeCaseTest(unittest.TestCase):
    """activate/deactivate/pause edge cases"""

    def setUp(self):
        Scope.initialize_registry()
        Scope.parsed = False
        self.config = ADict()
        self.scope = Scope(config=self.config, name='config')
        sys.argv = ['test.py']

    def test_pause_skips_config(self):
        @self.scope
        def main(config=None):
            return config

        result_normal = main()
        self.assertIsNotNone(result_normal)

        with self.scope.pause():
            result_paused = main()
        self.assertIsNone(result_paused)

        result_resumed = main()
        self.assertIsNotNone(result_resumed)

    def test_deactivate_activate(self):
        @self.scope
        def main(config=None):
            return config

        result = main()
        self.assertIsNotNone(result)

        self.scope.deactivate()
        result_off = main()
        self.assertIsNone(result_off)

        self.scope.activate()
        result_on = main()
        self.assertIsNotNone(result_on)


class ConfigViewEdgeCaseTest(unittest.TestCase):
    """Config-type view (non-function) edge cases"""

    def setUp(self):
        Scope.initialize_registry()
        Scope.parsed = False
        self.config = ADict()
        self.scope = Scope(config=self.config, name='config')
        sys.argv = ['test.py']

    def test_config_view(self):
        preset = ADict(a=1, b=2)
        self.scope.observe('preset_config', preset, priority=1)
        self.scope.assign('preset_config')
        self.scope.apply()
        self.assertEqual(self.config.a, 1)
        self.assertEqual(self.config.b, 2)

    def test_config_view_with_nested(self):
        preset = ADict(model=ADict(name='vgg', layers=16))
        self.scope.observe('nested_preset', preset, priority=1)
        self.scope.assign('nested_preset')
        self.scope.apply()
        self.assertEqual(self.config.model.name, 'vgg')
        self.assertEqual(self.config.model.layers, 16)


if __name__ == '__main__':
    unittest.main()
