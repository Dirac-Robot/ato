import io
import json
import pickle
import unittest

from ato.adict import ADict

from copy import deepcopy as dcp


class ADictUnitTest(unittest.TestCase):
    def setUp(self):
        self.simple_dict = {"name": "John Doe", "age": 30, "city": "New York"}
        self.nested_dict = {
            "user": {
                "name": "John Doe",
                "age": 30,
                "address": {"city": "New York", "country": "USA"},
            },
            "posts": [{"title": "Post 1", "content": "Hello, world!"}],
            "family": ["mother", "father", "sister", "brother", "wife", "son"]
        }
        self.long_dict = {
            "user-0": {"name": "Michael", "score": 12},
            "user-1": {"name": "William", "score": 61},
            "user-2": {"name": "Wilson", "score": 52},
            "user-3": {"name": "Andrew", "score": 93},
            "user-4": {"name": "Eugene", "score": 28},
            "user-5": {"name": "Richard", "score": 42},
            "user-6": {"name": "Lucy", "score": 66},
            "user-7": {"name": "Tracy", "score": 77},
            "user-8": {"name": "John", "score": 78},
            "user-9": {"name": "Elly", "score": 100}
        }
        self.adict_simple = ADict(self.simple_dict)
        self.adict_nested = ADict(self.nested_dict)
        self.adict_long = ADict(self.long_dict)

    def test_initialize(self):
        ADict(self.simple_dict)
        ADict(self.nested_dict)

    def test_default(self):
        def auto_nested_config():
            return ADict(default=auto_nested_config)
        config = auto_nested_config()
        config.plan.alpha.beta.gamma = 0
        self.assertEqual(config.plan.alpha.beta.gamma, 0)
        self.assertIsInstance(config.plan, ADict)
        self.assertIsInstance(config.plan.alpha, ADict)
        self.assertIsInstance(config.plan.alpha.beta, ADict)

    def test_initialize_from_kwargs(self):
        ADict(**self.simple_dict)
        ADict(**self.nested_dict)

    def test_implicit_convert(self):
        config = ADict(self.nested_dict)
        self.assertIsInstance(config['user'], ADict)
        self.assertIsInstance(config['user']['address'], ADict)

    def test_get_item_by_attribute(self):
        self.assertEqual(self.adict_simple.age, 30)

    def test_get_item_by_key(self):
        self.assertEqual(self.adict_simple['city'], "New York")

    def test_set_item_by_attribute(self):
        self.adict_simple.age = 10
        self.adict_nested.user.age = 12
        self.assertEqual(self.adict_simple.age, 10)
        self.assertEqual(self.adict_nested.user.age, 12)

    def test_set_item_by_key(self):
        self.adict_simple["city"] = "London"
        self.adict_nested.user.address["city"] = "London"
        self.assertEqual(self.adict_simple.city, "London")
        self.assertEqual(self.adict_nested.user.address.city, "London")

    def test_get_item_by_iterable(self):
        self.assertEqual(
            self.adict_nested.user["name", "age"],
            ["John Doe", 30]
        )

    def test_compute_with_value(self):
        self.adict_simple.age *= 10
        self.assertEqual(
            self.adict_simple.age, 300
        )

    def test_set_item_by_iterable(self):
        self.adict_nested.user["name", "age", "address"] = ["Richard Kim", 29, {"city": "Seoul", "country": "Korea"}]
        self.assertEqual(
            list(self.adict_nested.user.values()),
            ["Richard Kim", 29, ADict({"city": "Seoul", "country": "Korea"})]
        )

    def test_delete(self):
        with self.assertRaises(KeyError):
            del self.adict_simple["gender"]
            del self.adict_nested["careers"]
        del self.adict_simple["age"]
        self.assertTrue("age" not in self.adict_simple)

    def test_construct_with_various_inputs(self):
        with self.assertRaises(TypeError):
            ADict(10)
            ADict(self.nested_dict["family"])
            ADict(None)

    def test_clear(self):
        self.adict_simple.clear()
        self.assertEqual(len(self.adict_simple), 0)

    def test_deepcopy(self):
        self.assertEqual(self.adict_nested, dcp(self.adict_nested))

    def test_update(self):
        adict_nested = dcp(self.adict_nested)
        adict_nested.user.update(self.adict_simple)
        self.assertEqual(
            dict(adict_nested),
            {
                "user": {
                    "name": "John Doe",
                    "age": 30,
                    "city": "New York",
                    "address": {"city": "New York", "country": "USA"},
                },
                "posts": [{"title": "Post 1", "content": "Hello, world!"}],
                "family": ["mother", "father", "sister", "brother", "wife", "son"]
            }
        )

    def test_convert_between_json(self):
        adict_json = self.adict_nested.json()
        restored_adict = ADict(json.loads(adict_json))
        self.assertEqual(self.adict_nested, restored_adict)

    def test_convert_to_structural_repr(self):
        structural_repr = self.adict_nested.get_structural_repr()
        self.adict_nested.user.age = 31  # type is not changed
        edited_structural_repr = self.adict_nested.get_structural_repr()
        self.adict_nested.user.age = '35'  # type is changed
        type_edited_structural_repr = self.adict_nested.get_structural_repr()
        self.assertEqual(structural_repr, edited_structural_repr)
        self.assertNotEqual(structural_repr, type_edited_structural_repr)

    def test_convert_to_structural_hash(self):
        structural_hash = self.adict_nested.get_structural_hash()
        self.adict_nested.user.age = 31  # type is not changed
        edited_structural_hash = self.adict_nested.get_structural_hash()
        self.adict_nested.user.age = '35'  # type is changed
        type_edited_structural_hash = self.adict_nested.get_structural_hash()
        self.assertEqual(structural_hash, edited_structural_hash)
        self.assertNotEqual(structural_hash, type_edited_structural_hash)

    def test_pickle(self):
        pickle_io = io.BytesIO()
        pickle.dump(dcp(self.adict_nested), pickle_io)
        pickle_io.seek(0)
        restored_adict = pickle.load(pickle_io)
        self.assertEqual(self.adict_nested, restored_adict)

    def test_raw(self):
        self.assertEqual(self.adict_simple.raw('name'), ADict(key='name', value='John Doe'))

    def test_convert_to_immutable(self):
        self.adict_simple.convert_to_immutable()
        self.adict_nested.convert_to_immutable()
        with self.assertRaises(TypeError):
            del self.adict_simple['name']
        with self.assertRaises(TypeError):
            self.adict_simple['name'] = 'poo'
        with self.assertRaises(TypeError):
            del self.adict_nested['user']
        with self.assertRaises(TypeError):
            self.adict_nested['user'] = 'poo'
        with self.assertRaises(TypeError):
            del self.adict_simple.name
        with self.assertRaises(TypeError):
            self.adict_simple.name = 'poo'
        with self.assertRaises(TypeError):
            del self.adict_nested.user
        with self.assertRaises(TypeError):
            self.adict_nested.user = 'poo'

    def test_replace_keys(self):
        pass

    def test_recurrent_update(self):
        self.adict_nested.update(
            {
                "user": {
                    "name": "John Christopher",
                    "address": {"city": "Texas"}
                },
                "posts": [{"title": "Post 3", "content": "Hello, world!"}]
            },
            user={'age': 20},
            recurrent=True
        )
        self.assertIn('country', self.adict_nested.user.address)
        self.assertEqual(self.adict_nested.user.address.city, 'Texas')

    def test_convert_from_iterables(self):
        adict_converted = ADict([('Andrew', 'Jackson'), ('John', 'Christopher')])
        self.assertEqual(adict_converted.Andrew, 'Jackson')
        adict_converted['aa'] = ADict(bb=ADict(ee='ll'))


class RecursiveADictConversionTest(unittest.TestCase):
    '''Strict unit tests for recursive dict to ADict conversion in __setitem__.'''

    def test_single_level_dict_conversion(self):
        '''Single level dict should be converted to ADict.'''
        config = ADict()
        config.data = {'key': 'value'}
        self.assertIsInstance(config.data, ADict)
        self.assertEqual(config.data.key, 'value')

    def test_nested_dict_conversion(self):
        '''Nested dicts should be recursively converted to ADict.'''
        config = ADict()
        config.data = {'level1': {'level2': {'level3': 'value'}}}
        self.assertIsInstance(config.data, ADict)
        self.assertIsInstance(config.data.level1, ADict)
        self.assertIsInstance(config.data.level1.level2, ADict)
        self.assertEqual(config.data.level1.level2.level3, 'value')

    def test_deeply_nested_conversion(self):
        '''Very deeply nested dicts should all be converted.'''
        config = ADict()
        config.deep = {'a': {'b': {'c': {'d': {'e': {'f': 'bottom'}}}}}}
        self.assertIsInstance(config.deep, ADict)
        self.assertIsInstance(config.deep.a, ADict)
        self.assertIsInstance(config.deep.a.b, ADict)
        self.assertIsInstance(config.deep.a.b.c, ADict)
        self.assertIsInstance(config.deep.a.b.c.d, ADict)
        self.assertIsInstance(config.deep.a.b.c.d.e, ADict)
        self.assertEqual(config.deep.a.b.c.d.e.f, 'bottom')

    def test_list_of_dicts_conversion(self):
        '''Dicts inside lists should be converted to ADict.'''
        config = ADict()
        config.entries = [{'name': 'item1'}, {'name': 'item2'}]
        self.assertIsInstance(config.entries, list)
        self.assertIsInstance(config.entries[0], ADict)
        self.assertIsInstance(config.entries[1], ADict)
        self.assertEqual(config.entries[0].name, 'item1')
        self.assertEqual(config.entries[1].name, 'item2')

    def test_tuple_of_dicts_conversion(self):
        '''Dicts inside tuples should be converted to ADict and remain tuple.'''
        config = ADict()
        config.entries = ({'name': 'item1'}, {'name': 'item2'})
        self.assertIsInstance(config.entries, tuple)
        self.assertIsInstance(config.entries[0], ADict)
        self.assertIsInstance(config.entries[1], ADict)

    def test_nested_list_of_dicts_conversion(self):
        '''Nested dicts inside lists should also be converted.'''
        config = ADict()
        config.data = [{'nested': {'value': 42}}]
        self.assertIsInstance(config.data[0], ADict)
        self.assertIsInstance(config.data[0].nested, ADict)
        self.assertEqual(config.data[0].nested.value, 42)

    def test_mixed_types_in_list(self):
        '''Mixed types in list: only dicts should be converted.'''
        config = ADict()
        config.mixed = [{'a': 1}, 'string', 123, None, [{'b': 2}]]
        self.assertIsInstance(config.mixed[0], ADict)
        self.assertEqual(config.mixed[1], 'string')
        self.assertEqual(config.mixed[2], 123)
        self.assertIsNone(config.mixed[3])
        self.assertIsInstance(config.mixed[4], list)
        self.assertIsInstance(config.mixed[4][0], ADict)
        self.assertEqual(config.mixed[4][0].b, 2)

    def test_setitem_with_bracket_notation(self):
        '''Bracket notation setitem should also recursively convert.'''
        config = ADict()
        config['data'] = {'nested': {'deep': 'value'}}
        self.assertIsInstance(config['data'], ADict)
        self.assertIsInstance(config['data']['nested'], ADict)
        self.assertEqual(config['data']['nested']['deep'], 'value')

    def test_setitem_with_iterable_keys(self):
        '''Setitem with iterable keys and dict values should convert each.'''
        config = ADict()
        config['a', 'b'] = [{'x': 1}, {'y': 2}]
        self.assertIsInstance(config.a, ADict)
        self.assertIsInstance(config.b, ADict)
        self.assertEqual(config.a.x, 1)
        self.assertEqual(config.b.y, 2)

    def test_existing_adict_not_rewrapped(self):
        '''Existing ADict should not be re-wrapped.'''
        config = ADict()
        inner = ADict(name='inner')
        config.inner = inner
        self.assertIs(config.inner, inner)

    def test_empty_dict_conversion(self):
        '''Empty dict should be converted to empty ADict.'''
        config = ADict()
        config.empty = {}
        self.assertIsInstance(config.empty, ADict)
        self.assertEqual(len(config.empty), 0)

    def test_complex_structure(self):
        '''Complex structure with mixed nesting.'''
        config = ADict()
        config.complex = {
            'users': [
                {'name': 'Alice', 'meta': {'score': 100}},
                {'name': 'Bob', 'meta': {'score': 95}}
            ],
            'settings': {
                'debug': True,
                'options': {'a': 1, 'b': 2}
            }
        }
        self.assertIsInstance(config.complex, ADict)
        self.assertIsInstance(config.complex.users, list)
        self.assertIsInstance(config.complex.users[0], ADict)
        self.assertIsInstance(config.complex.users[0].meta, ADict)
        self.assertEqual(config.complex.users[0].meta.score, 100)
        self.assertIsInstance(config.complex.settings, ADict)
        self.assertIsInstance(config.complex.settings.options, ADict)
        self.assertTrue(config.complex.settings.debug)

    def test_overwrite_with_dict_converts(self):
        '''Overwriting an existing key with a dict should convert.'''
        config = ADict(value=10)
        config.value = {'nested': 'data'}
        self.assertIsInstance(config.value, ADict)
        self.assertEqual(config.value.nested, 'data')

    def test_primitives_not_affected(self):
        '''Primitive values should remain unchanged.'''
        config = ADict()
        config.string = 'hello'
        config.number = 42
        config.floating = 3.14
        config.boolean = True
        config.none = None
        self.assertEqual(config.string, 'hello')
        self.assertEqual(config.number, 42)
        self.assertEqual(config.floating, 3.14)
        self.assertTrue(config.boolean)
        self.assertIsNone(config.none)

    def test_frozen_does_not_convert_or_set(self):
        '''Frozen ADict should not allow setitem.'''
        config = ADict()
        config.freeze()
        config.data = {'should': 'not_set'}
        self.assertNotIn('data', config)


if __name__ == '__main__':
    unittest.main()
