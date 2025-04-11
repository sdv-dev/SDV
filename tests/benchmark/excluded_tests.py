"""Excluded test combinations."""

from tests.benchmark.numpy_dtypes import NUMPY_DATETIME_DTYPES, NUMPY_DTYPES
from tests.benchmark.pandas_dtypes import PANDAS_DATETIME_DTYPES, PANDAS_DTYPES

EXCLUDED_DATA_TYPES = {
    ('pd.boolean', 'numerical'),
    ('pd.string', 'numerical'),
    ('pd.category', 'numerical'),
    ('pd.Period', 'numerical'),
    ('np.bool', 'numerical'),
    ('np.object', 'numerical'),
    ('np.string', 'numerical'),
    ('np.unicode', 'numerical'),
    ('pd.datetime64', 'numerical'),
    ('np.datetime64', 'numerical'),
}

for dtypes in (NUMPY_DTYPES, PANDAS_DTYPES):
    for dtype in dtypes:
        EXCLUDED_DATA_TYPES.add((dtype, 'datetime'))

for dtypes in (NUMPY_DATETIME_DTYPES, PANDAS_DATETIME_DTYPES):
    for dtype in dtypes:
        for sdtype in ('numerical', 'id', 'categorical'):
            EXCLUDED_DATA_TYPES.add((dtype, sdtype))
