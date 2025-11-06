import sys
import unittest

from ato.scope import Scope, MultiScope


class MultiScopeTest(unittest.TestCase):
    def setUp(self):
        Scope.initialize_registry()
        self.scope_1 = Scope(name='config_1')
        self.scope_2 = Scope(name='config_2')
        self.multi_scope = MultiScope(self.scope_1, self.scope_2)

        @self.scope_1.observe(default=True, priority=0)
        def config_1_base(config_1):
            config_1.text = 'A'
            config_1.lr = 0.1

        @self.scope_2.observe(default=True, priority=0)
        def config_2_base(config_2):
            config_2.text = 'B'
            config_2.lr = 1.0

    def test_cli_isolation(self):
        sys.argv = 't.py config_1.text=%X% config_2.lr=3.14'.split()

        @self.multi_scope
        def main(config_1, config_2):
            return config_1.text, config_1.lr, config_2.text, config_2.lr

        out = main()
        self.assertEqual(out, ('X', 0.1, 'B', 3.14))

    def test_priority_chain_independence(self):
        sys.argv = 't.py config_1.config_1_chain config_2.config_2_chain'.split()

        @self.scope_1.observe(priority=1)
        def config_1_high(config_1):
            config_1.lr = 0.2

        @self.scope_2.observe(priority=1)
        def config_2_high(config_2):
            config_2.lr = 2.0

        @self.scope_1.observe(chain_with=['config_1_high'], priority=2)
        def config_1_chain(config_1):
            config_1.weight_decay = 1

        @self.scope_2.observe(chain_with=['config_2_high'], priority=2)
        def config_2_chain(config_2):
            config_2.weight_decay = 9

        @self.multi_scope
        def main(config_1, config_2):
            return config_1.lr, config_1.weight_decay, config_2.lr, config_2.weight_decay

        self.assertEqual(main(), (0.2, 1, 2.0, 9))


if __name__ == "__main__":
    unittest.main()
