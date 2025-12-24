import unittest
from ato.parser import (
    parse_command,
    parse_value,
    parse_colon_equals_string,
    parse_bracketed_value,
    parse_quoted_string,
)


class ParserUnitTest(unittest.TestCase):
    def test_parse_simple_command(self):
        command = 'batch_size=16 learning_rate=0.1'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['batch_size=16', 'learning_rate=0.1'])

    def test_parse_command_with_view(self):
        command = 'default_view batch_size=16'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['default_view', 'batch_size=16'])

    def test_parse_command_with_list(self):
        command = 'layers=[1, 2, 3] name=model'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['layers=[1, 2, 3]', 'name=model'])

    def test_parse_command_with_nested_list(self):
        command = 'config=[[1, 2], [3, 4]]'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['config=[[1, 2], [3, 4]]'])

    def test_parse_command_with_colon_equals_string(self):
        command = 'prompt:=Hello'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['prompt:=Hello'])

    def test_parse_command_with_colon_equals_quoted_string(self):
        command = 'prompt:="Hello World"'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['prompt:=Hello World'])

    def test_parse_command_with_colon_equals_single_quoted_string(self):
        command = "prompt:='Hello World'"
        tokens = parse_command(command)
        self.assertEqual(tokens, ['prompt:=Hello World'])

    def test_parse_command_mixed(self):
        command = 'view1 lr=0.1 prompt:=Hello layers=[1,2,3]'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['view1', 'lr=0.1', 'prompt:=Hello', 'layers=[1,2,3]'])

    def test_parse_command_mixed_with_quoted(self):
        command = 'view1 lr=0.1 prompt:="Hello World" layers=[1,2,3]'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['view1', 'lr=0.1', 'prompt:=Hello World', 'layers=[1,2,3]'])

    def test_parse_value_simple(self):
        command = '0.1'
        value, end_pos = parse_value(command, 0)
        self.assertEqual(value, '0.1')
        self.assertEqual(end_pos, 3)

    def test_parse_value_list(self):
        command = '[1, 2, 3]'
        value, end_pos = parse_value(command, 0)
        self.assertEqual(value, '[1, 2, 3]')
        self.assertEqual(end_pos, 9)

    def test_parse_value_dict(self):
        command = '{a: 1, b: 2}'
        value, end_pos = parse_value(command, 0)
        self.assertEqual(value, '{a: 1, b: 2}')
        self.assertEqual(end_pos, 12)

    def test_parse_colon_equals_string_simple(self):
        command = 'Hello'
        value, end_pos = parse_colon_equals_string(command, 0)
        self.assertEqual(value, 'Hello')
        self.assertEqual(end_pos, 5)

    def test_parse_colon_equals_string_quoted(self):
        command = '"Hello World" rest'
        value, end_pos = parse_colon_equals_string(command, 0)
        self.assertEqual(value, 'Hello World')
        self.assertEqual(end_pos, 13)

    def test_parse_colon_equals_string_single_quoted(self):
        command = "'Hello World' rest"
        value, end_pos = parse_colon_equals_string(command, 0)
        self.assertEqual(value, 'Hello World')
        self.assertEqual(end_pos, 13)

    def test_parse_quoted_string_with_escape(self):
        command = r'"Hello \"World\""'
        value, end_pos = parse_quoted_string(command, 0)
        self.assertEqual(value, 'Hello "World"')
        self.assertEqual(end_pos, 17)

    def test_parse_bracketed_value_list(self):
        command = '[1, 2, 3]'
        value, end_pos = parse_bracketed_value(command, 0)
        self.assertEqual(value, '[1, 2, 3]')
        self.assertEqual(end_pos, 9)

    def test_parse_bracketed_value_nested_list(self):
        command = '[[1, 2], [3, 4]]'
        value, end_pos = parse_bracketed_value(command, 0)
        self.assertEqual(value, '[[1, 2], [3, 4]]')
        self.assertEqual(end_pos, 16)

    def test_parse_bracketed_value_with_quoted_string(self):
        command = '["text", 1]'
        value, end_pos = parse_bracketed_value(command, 0)
        self.assertEqual(value, '["text", 1]')
        self.assertEqual(end_pos, 11)

    def test_parse_bracketed_value_dict(self):
        command = '{a: 1, b: 2}'
        value, end_pos = parse_bracketed_value(command, 0)
        self.assertEqual(value, '{a: 1, b: 2}')
        self.assertEqual(end_pos, 12)

    def test_parse_bracketed_value_with_escape(self):
        command = r'["\[escaped\]", 1]'
        value, end_pos = parse_bracketed_value(command, 0)
        self.assertEqual(value, r'["\[escaped\]", 1]')
        self.assertEqual(end_pos, 18)

    def test_parse_command_empty(self):
        command = ''
        tokens = parse_command(command)
        self.assertEqual(tokens, [])

    def test_parse_command_whitespace_only(self):
        command = '   '
        tokens = parse_command(command)
        self.assertEqual(tokens, [])

    def test_parse_command_with_multiple_spaces(self):
        command = 'view1    lr=0.1    batch_size=32 data:="v1 v2"'
        tokens = parse_command(command)
        self.assertEqual(tokens, ['view1', 'lr=0.1', 'batch_size=32', 'data:=v1 v2'])


if __name__ == '__main__':
    unittest.main()
