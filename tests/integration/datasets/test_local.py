import pandas as pd
import pytest

from sdv.datasets.local import save_csvs


@pytest.fixture
def data():
    parent = pd.DataFrame(
        data={
            'id': [0, 1, 2, 3, 4],
            'A': [True, True, False, True, False],
            'B': [0.434, 0.312, 0.212, 0.339, 0.491],
        }
    )

    child = pd.DataFrame(
        data={'parent_id': [0, 1, 2, 2, 5], 'C': ['Yes', 'No', 'Maye', 'No', 'No']}
    )

    grandchild = pd.DataFrame(
        data={'child_id': [0, 1, 2, 3, 4], 'D': [0.434, 0.312, 0.212, 0.339, 0.491]}
    )

    grandchild2 = pd.DataFrame(
        data={'child_id': [0, 1, 2, 3, 4], 'E': [0.434, 0.312, 0.212, 0.339, 0.491]}
    )

    return {'parent': parent, 'child': child, 'grandchild': grandchild, 'grandchild2': grandchild2}


def test_save_csvs(data, tmpdir):
    """Test that ``save_csvs`` end to end."""
    # Setup
    folder = tmpdir.mkdir('data')

    # Run
    save_csvs(data, folder, suffix='-synthetic', to_csv_parameters={'index': False})

    # Assert
    assert len(folder.listdir()) == 4
    for table in data:
        assert (folder / f'{table}-synthetic.csv').check()
