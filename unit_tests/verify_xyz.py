import unittest
import tempfile
import os
from beacon import xyz


class XYZUnitTest(unittest.TestCase):
    def setUp(self):
        self.simple_dict = {
            "name": "John",
            "age": 30,
            "city": "New York"
        }
        self.nested_dict = {
            "user": {
                "name": "John",
                "age": 30,
                "address": {
                    "city": "New York",
                    "country": "USA"
                }
            },
            "status": "active"
        }
        self.simple_list = [1, 2, 3, 4, 5]
        self.nested_list = [[1, 2], [3, 4], [5, 6]]
        self.mixed_structure = {
            "numbers": [1, 2, 3],
            "config": {
                "enabled": True,
                "values": [10, 20]
            }
        }

    def test_dumps_simple_dict(self):
        result = xyz.dumps(self.simple_dict)
        self.assertIsInstance(result, str)
        self.assertIn('name:', result)
        self.assertIn('age:', result)
        self.assertIn('city:', result)

    def test_dumps_nested_dict(self):
        result = xyz.dumps(self.nested_dict)
        self.assertIsInstance(result, str)
        self.assertIn('user:', result)
        self.assertIn('address:', result)

    def test_dumps_simple_list(self):
        result = xyz.dumps(self.simple_list)
        self.assertIsInstance(result, str)

    def test_dumps_nested_list(self):
        result = xyz.dumps(self.nested_list)
        self.assertIsInstance(result, str)

    def test_dumps_mixed_structure(self):
        result = xyz.dumps(self.mixed_structure)
        self.assertIsInstance(result, str)
        self.assertIn('numbers:', result)
        self.assertIn('config:', result)

    def test_loads_simple_dict(self):
        xyz_str = xyz.dumps(self.simple_dict)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored['name'], 'John')
        self.assertEqual(restored['age'], 30)
        self.assertEqual(restored['city'], 'New York')

    def test_loads_nested_dict(self):
        xyz_str = xyz.dumps(self.nested_dict)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored['user']['name'], 'John')
        self.assertEqual(restored['user']['address']['city'], 'New York')

    def test_loads_simple_list(self):
        xyz_str = xyz.dumps(self.simple_list)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored, self.simple_list)

    def test_loads_nested_list(self):
        xyz_str = xyz.dumps(self.nested_list)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored, self.nested_list)

    def test_loads_mixed_structure(self):
        xyz_str = xyz.dumps(self.mixed_structure)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored['numbers'], [1, 2, 3])
        self.assertEqual(restored['config']['enabled'], True)
        self.assertEqual(restored['config']['values'], [10, 20])

    def test_roundtrip_simple_dict(self):
        xyz_str = xyz.dumps(self.simple_dict)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored, self.simple_dict)

    def test_roundtrip_nested_dict(self):
        xyz_str = xyz.dumps(self.nested_dict)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored, self.nested_dict)

    def test_roundtrip_simple_list(self):
        xyz_str = xyz.dumps(self.simple_list)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored, self.simple_list)

    def test_roundtrip_mixed_structure(self):
        xyz_str = xyz.dumps(self.mixed_structure)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored, self.mixed_structure)

    def test_dump_and_load_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.xyz') as f:
            temp_path = f.name

        try:
            xyz.dump(self.nested_dict, temp_path)
            restored = xyz.load(temp_path)
            self.assertEqual(restored, self.nested_dict)
        finally:
            os.unlink(temp_path)

    def test_dump_and_load_file_object(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.xyz') as f:
            xyz.dump(self.nested_dict, f)
            f.seek(0)
            restored = xyz.load(f)
            temp_path = f.name

        try:
            self.assertEqual(restored, self.nested_dict)
        finally:
            os.unlink(temp_path)

    def test_convert_structure_to_tree(self):
        tree = xyz.convert_structure_to_tree(self.simple_dict)
        self.assertIsInstance(tree, xyz.GlobalParser)
        self.assertEqual(tree.node_type, 'dict')

    def test_convert_tree_to_structure(self):
        tree = xyz.convert_structure_to_tree(self.simple_dict)
        restored = xyz.convert_tree_to_structure(tree)
        self.assertEqual(restored, self.simple_dict)

    def test_custom_format_dict(self):
        format_dict = {
            'key_prefix': '- ',
            'key_postfix': ' =>',
            'index_prefix': '# ',
            'index_postfix': ']'
        }
        xyz_str = xyz.dumps(self.simple_dict, format_dict=format_dict)
        self.assertIn('- ', xyz_str)
        self.assertIn(' =>', xyz_str)

    def test_empty_dict(self):
        empty_dict = {}
        xyz_str = xyz.dumps(empty_dict)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored, empty_dict)

    def test_empty_list(self):
        empty_list = []
        xyz_str = xyz.dumps(empty_list)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored, empty_list)

    def test_single_value(self):
        single_value = 42
        xyz_str = xyz.dumps(single_value)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored, single_value)

    def test_string_value(self):
        string_value = "Hello World"
        xyz_str = xyz.dumps(string_value)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored, string_value)

    def test_boolean_values(self):
        bool_dict = {"enabled": True, "disabled": False}
        xyz_str = xyz.dumps(bool_dict)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored['enabled'], True)
        self.assertEqual(restored['disabled'], False)

    def test_none_value(self):
        none_dict = {"value": None}
        xyz_str = xyz.dumps(none_dict)
        restored = xyz.loads(xyz_str)
        self.assertEqual(restored['value'], None)

    def test_float_values(self):
        float_dict = {"pi": 3.14159, "e": 2.71828}
        xyz_str = xyz.dumps(float_dict)
        restored = xyz.loads(xyz_str)
        self.assertAlmostEqual(restored['pi'], 3.14159)
        self.assertAlmostEqual(restored['e'], 2.71828)


if __name__ == "__main__":
    unittest.main()