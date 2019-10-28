import os
from unittest.mock import patch

import numpy as np
import pandas as pd

from sdv import SDV


def get_metadata():
    return {
        'path': '',
        'tables': [
            {
                'headers': True,
                'name': 'types',
                'path': 'types.csv',
                'use': True,
                'fields': [
                    {
                        'name': 'int',
                        'type': 'number',
                        'subtype': 'integer'
                    },
                    {
                        'name': 'float',
                        'type': 'number',
                        'subtype': 'float'
                    }, {
                        'name': 'str',
                        'type': 'categorical',
                        'subtype': 'categorical'
                    }, {
                        'name': 'bool',
                        'type': 'categorical',
                        'subtype': 'boolean'
                    }, {
                        'name': 'datetime',
                        'type': 'datetime',
                        'format': '%Y-%m-%d'
                    }
                ]
            }
        ]
    }


@patch('sdv.modeler.Modeler.model_database')
def test_fit_metadata_as_dict_default(mock_model_db):
    """Test SDV instance fit metadata as type dict with default arguments.

    ``sdv.modeler.Modeler.model_database`` is patched in order to avoid data load from csv.
    """
    # Run
    sdv = SDV()
    metadata = get_metadata()
    sdv.fit(metadata)

    # Asserts
    mock_model_db.assert_called_once_with(None)
