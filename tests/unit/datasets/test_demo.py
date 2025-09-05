import io
import json
import re
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from sdv.datasets.demo import (
    _collect_json_keys,
    _download,
    _find_data_zip_key,
    _get_data_from_bucket,
    _iter_metainfo_yaml_entries,
    _list_objects,
    _select_metadata_v1_bytes,
    download_demo,
    get_available_demos,
)
from sdv.errors import DemoResourceNotFoundError


def test_download_demo_invalid_modality():
    """Test it crashes when an invalid modality is passed."""
    # Run and Assert
    err_msg = re.escape("'modality' must be in ['single_table', 'multi_table', 'sequential'].")
    with pytest.raises(ValueError, match=err_msg):
        download_demo('invalid_modality', 'dataset_name')


def test_download_demo_folder_already_exists(tmpdir):
    """Test it crashes when folder ``output_folder_name`` already exist."""
    # Run and Assert
    err_msg = re.escape(
        f"Folder '{tmpdir}' already exists. Please specify a different name "
        "or use 'load_csvs' to load from an existing folder."
    )
    with pytest.raises(ValueError, match=err_msg):
        download_demo('single_table', 'dataset_name', tmpdir)


@patch('sdv.datasets.demo._list_objects', return_value=[])
def test_download_demo_dataset_doesnt_exist(_mock_list):
    """Test it raises DemoResourceNotFoundError when dataset doesn't exist."""
    with pytest.raises(DemoResourceNotFoundError, match="Dataset 'invalid_dataset'.*was not found"):
        download_demo('single_table', 'invalid_dataset')


@patch('sdv.datasets.demo._get_data_from_bucket')
@patch('sdv.datasets.demo._list_objects')
def test_download_demo_single_table(mock_list, mock_get, tmpdir):
    """Test it can download a single table dataset using the new structure."""
    mock_list.return_value = [
        {'Key': 'single_table/ring/data.zip'},
        {'Key': 'single_table/ring/metadata.json'},
    ]
    df = pd.DataFrame({'0': [0, 0], '1': [0, 0]})
    zip_bytes = _make_zip_with_csv('ring.csv', df)
    meta_bytes = json.dumps({
        'METADATA_SPEC_VERSION': 'V1',
        'tables': {
            'ring': {
                'columns': {
                    '0': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                    '1': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                }
            }
        },
        'relationships': [],
    }).encode()

    def side_effect(key):
        if key.endswith('data.zip'):
            return zip_bytes
        if key.endswith('metadata.json'):
            return meta_bytes
        raise KeyError(key)

    mock_get.side_effect = side_effect

    # Run
    table, metadata = download_demo('single_table', 'ring', tmpdir / 'test_folder')

    # Assert
    expected_table = pd.DataFrame({'0': [0, 0], '1': [0, 0]})
    pd.testing.assert_frame_equal(table.head(2), expected_table)
    expected_metadata_dict = {
        'tables': {
            'ring': {
                'columns': {
                    '0': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                    '1': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                },
            }
        },
        'METADATA_SPEC_VERSION': 'V1',
        'relationships': [],
    }
    assert metadata.to_dict() == expected_metadata_dict


@patch('boto3.Session')
@patch('sdv.datasets.demo.BUCKET', 'bucket')
def test__get_data_from_bucket(session_mock):
    """Test the ``_get_data_from_bucket`` method."""
    # Setup
    mock_s3_client = Mock()
    session_mock.return_value.client.return_value = mock_s3_client
    mock_s3_client.get_object.return_value = {'Body': Mock(read=lambda: b'data')}

    # Run
    result = _get_data_from_bucket('object_key')

    # Assert
    assert result == b'data'
    session_mock.assert_called_once()
    mock_s3_client.get_object.assert_called_once_with(Bucket='bucket', Key='object_key')


@patch('sdv.datasets.demo._get_data_from_bucket')
@patch('sdv.datasets.demo._list_objects')
def test__download(mock_list, mock_get_data_from_bucket):
    """Test the ``_download`` method with new structure."""
    mock_list.return_value = [
        {'Key': 'single_table/ring/data.zip'},
        {'Key': 'single_table/ring/metadata.json'},
    ]
    # Return bytes for any key
    mock_get_data_from_bucket.return_value = json.dumps({'METADATA_SPEC_VERSION': 'V1'}).encode()

    # Run
    data_io, metadata_bytes = _download('single_table', 'ring')

    # Assert
    assert isinstance(data_io, io.BytesIO)
    assert isinstance(metadata_bytes, (bytes, bytearray))


@patch('sdv.datasets.demo._get_data_from_bucket')
@patch('sdv.datasets.demo._list_objects')
def test_download_demo_single_table_no_output_folder(mock_list, mock_get):
    """Test it can download a single table dataset when no output folder is passed."""
    mock_list.return_value = [
        {'Key': 'single_table/ring/data.zip'},
        {'Key': 'single_table/ring/metadata.json'},
    ]
    df = pd.DataFrame({'0': [0, 0], '1': [0, 0]})
    zip_bytes = _make_zip_with_csv('ring.csv', df)
    meta_bytes = json.dumps({
        'METADATA_SPEC_VERSION': 'V1',
        'tables': {
            'ring': {
                'columns': {
                    '0': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                    '1': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                }
            }
        },
        'relationships': [],
    }).encode()
    mock_get.side_effect = lambda key: zip_bytes if key.endswith('data.zip') else meta_bytes

    table, metadata = download_demo('single_table', 'ring')

    expected_table = pd.DataFrame({'0': [0, 0], '1': [0, 0]})
    pd.testing.assert_frame_equal(table.head(2), expected_table)
    expected_metadata_dict = {
        'tables': {
            'ring': {
                'columns': {
                    '0': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                    '1': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                },
            }
        },
        'METADATA_SPEC_VERSION': 'V1',
        'relationships': [],
    }
    assert metadata.to_dict() == expected_metadata_dict


@patch('sdv.datasets.demo._get_data_from_bucket')
@patch('sdv.datasets.demo._list_objects')
def test_download_demo_timeseries(mock_list, mock_get, tmpdir):
    """Test it can download a timeseries dataset using new structure."""
    mock_list.return_value = [
        {'Key': 'sequential/Libras/data.zip'},
        {'Key': 'sequential/Libras/metadata.json'},
    ]
    df = pd.DataFrame({
        'ml_class': [1, 1],
        'e_id': [0, 0],
        's_index': [0, 1],
        'tt_split': [1, 1],
        'dim_0': [0.67892, 0.68085],
        'dim_1': [0.27315, 0.27315],
    })
    zip_bytes = _make_zip_with_csv('Libras.csv', df)
    meta_bytes = json.dumps({
        'METADATA_SPEC_VERSION': 'V1',
        'relationships': [],
        'tables': {
            'Libras': {
                'columns': {
                    'e_id': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                    'dim_0': {'sdtype': 'numerical', 'computer_representation': 'Float'},
                    'dim_1': {'sdtype': 'numerical', 'computer_representation': 'Float'},
                    'ml_class': {'sdtype': 'categorical'},
                }
            }
        },
    }).encode()
    mock_get.side_effect = lambda key: zip_bytes if key.endswith('data.zip') else meta_bytes

    table, metadata = download_demo('sequential', 'Libras', tmpdir / 'test_folder')
    expected_table = pd.DataFrame({
        'ml_class': [1, 1],
        'e_id': [0, 0],
        's_index': [0, 1],
        'tt_split': [1, 1],
        'dim_0': [0.67892, 0.68085],
        'dim_1': [0.27315, 0.27315],
    })
    pd.testing.assert_frame_equal(table.head(2), expected_table)
    expected_metadata_dict = {
        'METADATA_SPEC_VERSION': 'V1',
        'relationships': [],
        'tables': {
            'Libras': {
                'columns': {
                    'e_id': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                    'dim_0': {'sdtype': 'numerical', 'computer_representation': 'Float'},
                    'dim_1': {'sdtype': 'numerical', 'computer_representation': 'Float'},
                    'ml_class': {'sdtype': 'categorical'},
                }
            }
        },
    }
    assert metadata.to_dict() == expected_metadata_dict


@patch('sdv.datasets.demo._get_data_from_bucket')
@patch('sdv.datasets.demo._list_objects')
def test_download_demo_multi_table(mock_list, mock_get, tmpdir):
    """Test it can download a multi table dataset using new structure."""
    mock_list.return_value = [
        {'Key': 'multi_table/got_families/data.zip'},
        {'Key': 'multi_table/got_families/metadata.json'},
    ]
    families = pd.DataFrame({'family_id': [1, 2], 'name': ['Stark', 'Tully']})
    character_families = pd.DataFrame({
        'character_id': [1, 1],
        'family_id': [1, 4],
        'generation': [8, 5],
        'type': ['father', 'mother'],
    })
    characters = pd.DataFrame({'age': [20, 16], 'character_id': [1, 2], 'name': ['Jon', 'Arya']})
    import zipfile

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('families.csv', families.to_csv(index=False))
        zf.writestr('character_families.csv', character_families.to_csv(index=False))
        zf.writestr('characters.csv', characters.to_csv(index=False))
    zip_bytes = zip_buf.getvalue()
    meta_bytes = json.dumps({
        'tables': {
            'characters': {
                'columns': {
                    'character_id': {'sdtype': 'id', 'regex_format': '^[1-9]{1,2}$'},
                    'name': {'sdtype': 'categorical'},
                    'age': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                },
                'primary_key': 'character_id',
            },
            'families': {
                'columns': {
                    'family_id': {'sdtype': 'id', 'regex_format': '^[1-9]$'},
                    'name': {'sdtype': 'categorical'},
                },
                'primary_key': 'family_id',
            },
            'character_families': {
                'columns': {
                    'character_id': {'sdtype': 'id', 'regex_format': '[A-Za-z]{5}'},
                    'family_id': {'sdtype': 'id', 'regex_format': '[A-Za-z]{5}'},
                    'type': {'sdtype': 'categorical'},
                    'generation': {'sdtype': 'numerical', 'computer_representation': 'Int64'},
                },
            },
        },
        'relationships': [
            {
                'parent_table_name': 'families',
                'parent_primary_key': 'family_id',
                'child_table_name': 'character_families',
                'child_foreign_key': 'family_id',
            },
            {
                'parent_table_name': 'characters',
                'parent_primary_key': 'character_id',
                'child_table_name': 'character_families',
                'child_foreign_key': 'character_id',
            },
        ],
        'METADATA_SPEC_VERSION': 'V1',
    }).encode()
    mock_get.side_effect = lambda key: zip_bytes if key.endswith('data.zip') else meta_bytes

    tables, metadata = download_demo('multi_table', 'got_families', tmpdir / 'test_folder')
    pd.testing.assert_frame_equal(tables['families'].head(2), families.head(2))
    pd.testing.assert_frame_equal(tables['character_families'].head(2), character_families.head(2))
    pd.testing.assert_frame_equal(tables['characters'].head(2), characters.head(2))
    expected_metadata_dict = json.loads(meta_bytes.decode())
    assert metadata.to_dict() == expected_metadata_dict


def test_get_available_demos_invalid_modality():
    """Test it crashes when an invalid modality is passed."""
    # Run and Assert
    err_msg = re.escape("'modality' must be in ['single_table', 'multi_table', 'sequential'].")
    with pytest.raises(ValueError, match=err_msg):
        get_available_demos('invalid_modality')


def _make_zip_with_csv(csv_name: str, df: pd.DataFrame) -> bytes:
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, df.to_csv(index=False))
    return buf.getvalue()


def test__list_objects_pagination():
    class FakeClient:
        def __init__(self):
            self.calls = []
            self.page = 0

        def list_objects_v2(self, **kwargs):
            self.calls.append(kwargs)
            if self.page == 0:
                self.page += 1
                return {
                    'IsTruncated': True,
                    'NextContinuationToken': 'token-1',
                    'Contents': [{'Key': 'single_table/d1/metainfo.yaml'}],
                }
            return {
                'IsTruncated': False,
                'Contents': [{'Key': 'single_table/d2/metainfo.yaml'}],
            }

    with patch('sdv.datasets.demo._create_s3_client', return_value=FakeClient()):
        contents = _list_objects('single_table/')
        keys = [c['Key'] for c in contents]
        assert keys == ['single_table/d1/metainfo.yaml', 'single_table/d2/metainfo.yaml']


def test__find_data_zip_key_and__collect_json_keys():
    contents = [
        {'Key': 'single_table/dataset/data.ZIP'},
        {'Key': 'single_table/dataset/metadata.json'},
        {'Key': 'single_table/dataset/aaa_wrong.json'},
        {'Key': 'single_table/dataset/README.txt'},
    ]
    dataset_prefix = 'single_table/dataset/'
    zip_key = _find_data_zip_key(contents, dataset_prefix)
    assert zip_key == 'single_table/dataset/data.ZIP'
    json_keys = _collect_json_keys(contents, dataset_prefix)
    assert set(json_keys) == {
        'single_table/dataset/metadata.json',
        'single_table/dataset/aaa_wrong.json',
    }


@patch('sdv.datasets.demo._get_data_from_bucket')
def test__select_metadata_v1_bytes(mock_get):
    v2 = json.dumps({'METADATA_SPEC_VERSION': 'V2'}).encode()
    bad = b'not-json'
    v1 = json.dumps({'METADATA_SPEC_VERSION': 'V1'}).encode()

    def side_effect(key):
        return {'k1.json': v2, 'k2.json': bad, 'k3.json': v1}[key]

    mock_get.side_effect = side_effect
    got = _select_metadata_v1_bytes(['k1.json', 'k2.json', 'k3.json'])
    assert got == v1


def test__iter_metainfo_yaml_entries_filters():
    contents = [
        {'Key': 'single_table/d1/metainfo.yaml'},
        {'Key': 'single_table/d1/METAINFO.YAML'},
        {'Key': 'single_table/d2/not.yaml'},
        {'Key': 'multi_table/d3/metainfo.yaml'},
        {'Key': 'single_table/metainfo.yaml'},
    ]
    got = list(_iter_metainfo_yaml_entries(contents, 'single_table'))
    assert ('d1', 'single_table/d1/metainfo.yaml') in got
    assert ('d1', 'single_table/d1/METAINFO.YAML') in got
    assert all(name != 'd3' for name, _ in got)
    assert all(key != 'single_table/metainfo.yaml' for _, key in got)


@patch('sdv.datasets.demo._get_data_from_bucket')
@patch('sdv.datasets.demo._list_objects')
def test_get_available_demos_robust_parsing(mock_list, mock_get):
    mock_list.return_value = [
        {'Key': 'single_table/d1/metainfo.yaml'},
        {'Key': 'single_table/d2/metainfo.yaml'},
        {'Key': 'single_table/bad/metainfo.yaml'},
        {'Key': 'single_table/ignore.txt'},
    ]

    def side_effect(key):
        if key.endswith('d1/metainfo.yaml'):
            return b'dataset-name: d1\nnum-tables: 2\ndataset-size-mb: 10.5\nsource: EXTERNAL\n'
        if key.endswith('d2/metainfo.yaml'):
            return b'dataset-name: d2\nnum-tables: not_a_number\ndataset-size-mb: NaN\n'
        raise ValueError('invalid yaml')

    mock_get.side_effect = side_effect
    df = get_available_demos('single_table')
    assert set(df['dataset_name']) == {'d1', 'd2'}
    # d1 parsed correctly
    row1 = df[df['dataset_name'] == 'd1'].iloc[0]
    assert row1['num_tables'] == 2
    assert row1['size_MB'] == 10.5
    # d2 falls back to NaN
    row2 = df[df['dataset_name'] == 'd2'].iloc[0]
    assert np.isnan(row2['num_tables']) or row2['num_tables'] is None
    assert np.isnan(row2['size_MB']) or row2['size_MB'] is None


@patch('sdv.datasets.demo._get_data_from_bucket')
@patch('sdv.datasets.demo._list_objects')
def test_download_demo_success_single_table(mock_list, mock_get):
    mock_list.return_value = [
        {'Key': 'single_table/word/data.ZIP'},
        {'Key': 'single_table/word/metadata.json'},
    ]
    df = pd.DataFrame({'id': [1, 2], 'name': ['a', 'b']})
    zip_bytes = _make_zip_with_csv('word.csv', df)
    meta_bytes = json.dumps({
        'METADATA_SPEC_VERSION': 'V1',
        'tables': {
            'word': {
                'columns': {
                    'id': {'sdtype': 'id'},
                    'name': {'sdtype': 'categorical'},
                },
                'primary_key': 'id',
            }
        },
        'relationships': [],
    }).encode()

    def side_effect(key):
        if key.endswith('data.ZIP'):
            return zip_bytes
        if key.endswith('metadata.json'):
            return meta_bytes
        raise KeyError(key)

    mock_get.side_effect = side_effect

    data, metadata = download_demo('single_table', 'word')
    assert isinstance(data, pd.DataFrame)
    assert set(data.columns) == {'id', 'name'}
    assert metadata.to_dict()['tables']['word']['primary_key'] == 'id'


@patch('sdv.datasets.demo._get_data_from_bucket', return_value=b'{}')
@patch('sdv.datasets.demo._list_objects')
def test_download_demo_missing_zip_raises(mock_list, _mock_get):
    mock_list.return_value = [
        {'Key': 'single_table/word/metadata.json'},
    ]
    with pytest.raises(DemoResourceNotFoundError, match="Could not find 'data.zip'"):
        download_demo('single_table', 'word')


@patch('sdv.datasets.demo._get_data_from_bucket')
@patch('sdv.datasets.demo._list_objects')
def test_download_demo_no_v1_metadata_raises(mock_list, mock_get):
    mock_list.return_value = [
        {'Key': 'single_table/word/data.zip'},
        {'Key': 'single_table/word/metadata.json'},
    ]
    mock_get.side_effect = lambda key: json.dumps({'METADATA_SPEC_VERSION': 'V2'}).encode()
    with pytest.raises(DemoResourceNotFoundError, match='METADATA_SPEC_VERSION'):
        download_demo('single_table', 'word')
