import pandas as pd
import pytest

from sdv.metadata import Table


class TestTable:

    def test__make_ids(self):
        """Test whether regex is correctly generating expressions."""
        metadata = {'subtype': 'string', 'regex': '[a-d]'}
        keys = Table._make_ids(metadata, 3)
        assert (keys == pd.Series(['a', 'b', 'c'])).all()

    def test__make_ids_fail(self):
        """Test if regex fails with more requested ids than available unique values."""
        metadata = {'subtype': 'string', 'regex': '[a-d]'}
        with pytest.raises(ValueError):
            Table._make_ids(metadata, 20)

    def test_make_ids_unique_field_not_unique(self):
        """Test that id column is replaced with all unique values if not already unique."""
        metadata_dict = {
            'fields': {
                'item 0': {'type': 'id', 'subtype': 'integer'},
                'item 1': {'type': 'boolean'}
            },
            'primary_key': 'item 0'
        }
        metadata = Table.from_dict(metadata_dict)
        data = pd.DataFrame({
            'item 0': [0, 1, 1, 2, 3, 5, 5, 6],
            'item 1': [True, True, False, False, True, False, False, True]
        })

        new_data = metadata.make_ids_unique(data)

        assert new_data['item 1'].equals(data['item 1'])
        assert new_data['item 0'].is_unique

    def test_make_ids_unique_field_already_unique(self):
        """Test that id column is kept if already unique."""
        metadata_dict = {
            'fields': {
                'item 0': {'type': 'id', 'subtype': 'integer'},
                'item 1': {'type': 'boolean'}
            },
            'primary_key': 'item 0'
        }
        metadata = Table.from_dict(metadata_dict)
        data = pd.DataFrame({
            'item 0': [9, 1, 8, 2, 3, 7, 5, 6],
            'item 1': [True, True, False, False, True, False, False, True]
        })

        new_data = metadata.make_ids_unique(data)

        assert new_data['item 1'].equals(data['item 1'])
        assert new_data['item 0'].equals(data['item 0'])

    def test_make_ids_unique_field_index_out_of_order(self):
        """Test that updated id column is unique even if index is out of order."""
        metadata_dict = {
            'fields': {
                'item 0': {'type': 'id', 'subtype': 'integer'},
                'item 1': {'type': 'boolean'}
            },
            'primary_key': 'item 0'
        }
        metadata = Table.from_dict(metadata_dict)
        data = pd.DataFrame({
            'item 0': [0, 1, 1, 2, 3, 5, 5, 6],
            'item 1': [True, True, False, False, True, False, False, True]
        }, index=[0, 1, 1, 2, 3, 5, 5, 6])

        new_data = metadata.make_ids_unique(data)

        assert new_data['item 1'].equals(data['item 1'])
        assert new_data['item 0'].is_unique
