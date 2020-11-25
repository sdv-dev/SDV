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
