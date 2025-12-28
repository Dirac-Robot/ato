"""
Extreme Edge Case Tests for ato CLI Parsing
Tests various corner cases for :=, nested keys, special characters, etc.
"""
import unittest
import sys
from ato.parser import parse_command
from ato.scope import Scope, parse_args_pythonic
from ato.adict import ADict


class ExtremeCaseParserTest(unittest.TestCase):
    """Parser-level edge case tests"""

    def test_nested_key_with_colon_equals(self):
        command = 'x.y:=abc'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['x.y:=abc'])

    def test_deep_nested_key(self):
        command = 'a.b.c.d.e:=value'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['a.b.c.d.e:=value'])

    def test_nested_key_with_quoted_space(self):
        command = 'x.y:="hello world"'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['x.y:=hello world'])

    def test_empty_value_colon_equals(self):
        command = 'key:='
        tokens = parse_command(command)
        self.assertEqual(tokens, ['key:='])

    def test_value_contains_equals(self):
        command = 'x.y:=a=b=c'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['x.y:=a=b=c'])

    def test_value_contains_colon(self):
        command = 'path:=/usr/local:bin'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['path:=/usr/local:bin'])

    def test_path_like_value(self):
        command = 'checkpoint:=./models/checkpoint_100.pth'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['checkpoint:=./models/checkpoint_100.pth'])

    def test_url_like_value(self):
        command = 'url:="https://example.com/path?q=1"'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['url:=https://example.com/path?q=1'])

    def test_single_quote_in_double_quoted(self):
        command = "prompt:=\"it's a test\""
        tokens = parse_command(command)
        self.assertEqual(tokens, ["prompt:=it's a test"])

    def test_double_quote_in_single_quoted(self):
        command = "prompt:='say \"hello\"'"
        tokens = parse_command(command)
        self.assertEqual(tokens, ['prompt:=say "hello"'])

    def test_escaped_quotes(self):
        command = r'text:="escaped \"quote\""'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['text:=escaped "quote"'])

    def test_backslash_in_value(self):
        command = r'path:=C:\Users\test'
        tokens = parse_command(command)
        self.assertEqual(tokens, [r'path:=C:\Users\test'])

    def test_unicode_value(self):
        command = 'name:="한글 테스트"'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['name:=한글 테스트'])

    def test_mixed_types_multiple_args(self):
        command = 'view1 x.y:=hello z=123 a.b.c:="with space" num=[1,2,3]'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['view1', 'x.y:=hello', 'z=123', 'a.b.c:=with space', 'num=[1,2,3]'])

    def test_dict_value(self):
        command = 'config={a: 1, b: 2}'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['config={a: 1, b: 2}'])

    def test_nested_list_value(self):
        command = 'layers=[[1,2],[3,4]]'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['layers=[[1,2],[3,4]]'])

    def test_very_long_string(self):
        long_val = 'x'*1000
        command = f'text:="{long_val}"'
        tokens = parse_command(command)
        self.assertEqual(tokens, [f'text:={long_val}'])

    def test_special_chars_in_value(self):
        command = 'chars:="@#$%^&*()[]{}|"'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['chars:=@#$%^&*()[]{}|'])


class ExtremeCaseScopeTest(unittest.TestCase):
    """Scope-level edge case tests (full integration)"""

    def setUp(self):
        Scope.initialize_registry()
        Scope.parsed = False
        Scope.stored_arguments = None
        self.config = ADict()
        self.scope = Scope(config=self.config, name='config')

        @self.scope.observe(default=True)
        def defaults(config):
            config.model = ADict(name='resnet', path='/default')
            config.data = ADict(prompt='original', path='/data')
            config.deep = ADict(a=ADict(b=ADict(c='deep_val')))

    def _run_with_args(self, args):
        Scope.parsed = False
        sys.argv = ['test.py'] + args
        parse_args_pythonic()
        self.scope.apply()

    def test_nested_string_basic(self):
        self._run_with_args(['model.name:=vgg'])
        self.assertEqual(self.config.model.name, 'vgg')

    def test_nested_string_with_space(self):
        self._run_with_args(['data.prompt:="hello world"'])
        self.assertEqual(self.config.data.prompt, 'hello world')

    def test_deep_nested_string(self):
        self._run_with_args(['deep.a.b.c:=new_value'])
        self.assertEqual(self.config.deep.a.b.c, 'new_value')

    def test_path_value(self):
        self._run_with_args(['model.path:=./checkpoints/model.pth'])
        self.assertEqual(self.config.model.path, './checkpoints/model.pth')

    def test_single_quote_in_value(self):
        self._run_with_args(["data.prompt:=\"it's working\""])
        self.assertEqual(self.config.data.prompt, "it's working")

    def test_multiple_nested_string_args(self):
        self._run_with_args([
            'model.name:=transformer',
            'model.path:=/new/path',
            'data.prompt:="test prompt"'
        ])
        self.assertEqual(self.config.model.name, 'transformer')
        self.assertEqual(self.config.model.path, '/new/path')
        self.assertEqual(self.config.data.prompt, 'test prompt')

    def test_mixed_string_and_value(self):
        self._run_with_args([
            'model.name:=vgg',
            'data.path:=/other/path'
        ])
        self.assertEqual(self.config.model.name, 'vgg')
        self.assertEqual(self.config.data.path, '/other/path')

    def test_unicode_value(self):
        self._run_with_args(['data.prompt:="한글 테스트"'])
        self.assertEqual(self.config.data.prompt, '한글 테스트')

    def test_empty_string_value(self):
        self._run_with_args(["data.prompt:=''"])
        self.assertEqual(self.config.data.prompt, '')


class ExtremeCaseMultiScopeTest(unittest.TestCase):
    """MultiScope edge case tests"""

    def setUp(self):
        Scope.initialize_registry()
        Scope.parsed = False
        Scope.stored_arguments = None
        from ato.scope import MultiScope
        self.scope_1 = Scope(name='model')
        self.scope_2 = Scope(name='data')
        self.multi = MultiScope(self.scope_1, self.scope_2)

        @self.scope_1.observe(default=True)
        def model_defaults(model):
            model.backbone = ADict(name='resnet', layers=50)

        @self.scope_2.observe(default=True)
        def data_defaults(data):
            data.dataset = ADict(name='cifar', path='/data')

    def test_multi_scope_nested_string(self):
        sys.argv = ['test.py', 'model.backbone.name:=vgg', 'data.dataset.name:=imagenet']

        @self.multi
        def main(model, data):
            return model.backbone.name, data.dataset.name

        result = main()
        self.assertEqual(result, ('vgg', 'imagenet'))

    def test_multi_scope_with_space(self):
        sys.argv = ['test.py', 'data.dataset.path:="/path/with space"']

        @self.multi
        def main(model, data):
            return data.dataset.path

        result = main()
        self.assertEqual(result, '/path/with space')


if __name__ == '__main__':
    unittest.main()
