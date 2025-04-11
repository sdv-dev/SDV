"""Excluded test combinations."""
from tests.benchmark.numpy_dtypes import NUMPY_DTYPES, NUMPY_DATETIME_DTYPES
from tests.benchmark.pandas_dtypes import PANDAS_DTYPES, PANDAS_DATETIME_DTYPES

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
    ('pd.boolean', 'datetime'),
    ('pd.timedelta64', 'datetime'),
    ('pd.Period', 'datetime'),
    ('pd.Complex', 'datetime'),
    ('np.complex64', 'datetime'),
    ('np.complex128', 'datetime'),
    ('np.bool', 'datetime'),
    ('np.unicode', 'datetime'),
    ('np.timedelta64', 'datetime'),
}

for dtypes in (NUMPY_DTYPES, PANDAS_DTYPES):
    for dtype in dtypes:
        EXCLUDED_DATA_TYPES.add((dtype, 'datetime'))

for dtypes in (NUMPY_DATETIME_DTYPES, PANDAS_DATETIME_DTYPES):
    for dtype in dtypes:
        for sdtype in ('numerical', 'id', 'categorical'):
            EXCLUDED_DATA_TYPES.add((dtype, sdtype))
