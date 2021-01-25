import pandas as pd
import rdt
from copulas.univariate import BetaUnivariate

from sdv.demo import load_demo
from sdv.relational import HMA1
from sdv.tabular import GaussianCopula


def test_sdv_model_kwargs():
    metadata, tables = load_demo(metadata=True)
    tables = {'users': tables['users']}
    metadata = metadata.to_dict()
    del metadata['tables']['sessions']
    del metadata['tables']['transactions']

    hma = HMA1(metadata, model=GaussianCopula, model_kwargs={
        'default_distribution': 'beta',
        'categorical_transformer': 'label_encoding',
    })
    hma.fit(tables)

    model = hma._models['users']
    assert model._default_distribution == BetaUnivariate
    assert model._DTYPE_TRANSFORMERS['O'] == 'label_encoding'
    assert isinstance(
        model._metadata._hyper_transformer._transformers['gender'],
        rdt.transformers.categorical.LabelEncodingTransformer
    )


def test_ids_only_child():
    """Ensure tables with nothing else than ids can be modeled and sampled."""
    parent = pd.DataFrame({
        'parent_id': range(10),
    })
    pk_child = pd.DataFrame({
        'child_id': range(10),
        'parent_id': range(10),
    })
    no_pk_child = pd.DataFrame({
        'parent_id': range(10),
    })

    metadata = {
        'tables': {
            'parent': {
                'fields': {
                    'parent_id': {
                        'type': 'id',
                    },
                },
                'primary_key': 'parent_id',
            },
            'pk_child': {
                'fields': {
                    'child_id': {
                        'type': 'id',
                    },
                    'parent_id': {
                        'type': 'id',
                        'ref': {
                            'table': 'parent',
                            'field': 'field_id'
                        }
                    },
                },
                'primary_key': 'child_id',
            },
            'no_pk_child': {
                'fields': {
                    'parent_id': {
                        'type': 'id',
                        'ref': {
                            'table': 'parent',
                            'field': 'field_id'
                        }
                    },
                },
            },
        }
    }
    tables = {
        'parent': parent,
        'pk_child': pk_child,
        'no_pk_child': no_pk_child,
    }

    hma1 = HMA1(metadata=metadata)
    hma1.fit(tables)
    sampled = hma1.sample()

    assert set(sampled.keys()) == {'parent', 'pk_child', 'no_pk_child'}

    for name, table in tables.items():
        assert table.shape == sampled[name].shape
        assert table.columns.tolist() == sampled[name].columns.tolist()
