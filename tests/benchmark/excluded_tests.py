"""Excluded tests from constraints due to hard crashing from NumPy or Pandas."""

EXCLUDED_CONSTRAINT_TESTS = [
    ('numerical', 'pd.boolean', 'FixedIncrements'),
    ('numerical', 'pd.object', 'Positive'),
    ('numerical', 'pd.object', 'Negative'),
    ('numerical', 'pd.object', 'ScalarInequality'),
    ('numerical', 'pd.object', 'ScalarRange'),
    ('numerical', 'pd.string', 'Positive'),
    ('numerical', 'pd.string', 'Negative'),
    ('numerical', 'pd.string', 'ScalarInequality'),
    ('numerical', 'pd.category', 'Positive'),
    ('numerical', 'pd.category', 'Negative'),
    ('numerical', 'pd.category', 'ScalarInequality'),
    ('numerical', 'pd.category', 'ScalarRange'),
    ('numerical', 'pd.datetime64', 'Positive'),
    ('numerical', 'pd.datetime64', 'Negative'),
    ('numerical', 'pd.datetime64', 'ScalarInequality'),
    ('numerical', 'pd.timedelta64', 'Positive'),
    ('numerical', 'pd.timedelta64', 'Negative'),
    ('numerical', 'pd.timedelta64', 'ScalarInequality'),
    ('numerical', 'pd.Period', 'Positive'),
    ('numerical', 'pd.Period', 'Negative'),
    ('numerical', 'pd.Period', 'ScalarInequality'),
    ('numerical', 'pd.Period', 'FixedIncrements'),
    ('numerical', 'np.object', 'Positive'),
    ('numerical', 'np.object', 'Negative'),
    ('numerical', 'np.object', 'ScalarInequality'),
    ('numerical', 'np.string', 'Positive'),
    ('numerical', 'np.string', 'Negative'),
    ('numerical', 'np.string', 'ScalarInequality'),
    ('numerical', 'np.bytes', 'Positive'),
    ('numerical', 'np.bytes', 'Negative'),
    ('numerical', 'np.bytes', 'ScalarInequality'),
    ('numerical', 'np.unicode', 'Positive'),
    ('numerical', 'np.unicode', 'Negative'),
    ('numerical', 'np.unicode', 'ScalarInequality'),
    ('numerical', 'np.datetime64', 'Positive'),
    ('numerical', 'np.datetime64', 'Negative'),
    ('numerical', 'np.datetime64', 'ScalarInequality'),
    ('numerical', 'np.timedelta64', 'Positive'),
    ('numerical', 'np.timedelta64', 'Negative'),
    ('numerical', 'np.timedelta64', 'ScalarInequality'),
    ('numerical', 'pa.string', 'Positive'),
    ('numerical', 'pa.string', 'Negative'),
    ('numerical', 'pa.string', 'ScalarInequality'),
    ('numerical', 'pa.utf8', 'Positive'),
    ('numerical', 'pa.utf8', 'Negative'),
    ('numerical', 'pa.utf8', 'ScalarInequality'),
    ('numerical', 'pa.binary', 'Positive'),
    ('numerical', 'pa.binary', 'Negative'),
    ('numerical', 'pa.binary', 'ScalarInequality'),
    ('numerical', 'pa.binary', 'FixedIncrements'),
    ('numerical', 'pa.large_binary', 'Positive'),
    ('numerical', 'pa.large_binary', 'Negative'),
    ('numerical', 'pa.large_binary', 'ScalarInequality'),
    ('numerical', 'pa.large_binary', 'FixedIncrements'),
    ('numerical', 'pa.large_string', 'Positive'),
    ('numerical', 'pa.large_string', 'Negative'),
    ('numerical', 'pa.large_string', 'ScalarInequality'),
    ('numerical', 'pa.date32', 'Positive'),
    ('numerical', 'pa.date32', 'Negative'),
    ('numerical', 'pa.date32', 'ScalarInequality'),
    ('numerical', 'pa.date64', 'Positive'),
    ('numerical', 'pa.date64', 'Negative'),
    ('numerical', 'pa.date64', 'ScalarInequality'),
    ('numerical', 'pa.timestamp', 'Positive'),
    ('numerical', 'pa.timestamp', 'Negative'),
    ('numerical', 'pa.timestamp', 'ScalarInequality'),
    ('numerical', 'pa.duration', 'Positive'),
    ('numerical', 'pa.duration', 'Negative'),
    ('numerical', 'pa.duration', 'ScalarInequality'),
    ('numerical', 'pa.time32', 'Positive'),
    ('numerical', 'pa.time32', 'Negative'),
    ('numerical', 'pa.time32', 'ScalarInequality'),
    ('numerical', 'pa.time64', 'Positive'),
    ('numerical', 'pa.time64', 'Negative'),
    ('numerical', 'pa.time64', 'ScalarInequality'),
    ('numerical', 'pa.binary_view', 'Positive'),
    ('numerical', 'pa.binary_view', 'Negative'),
    ('numerical', 'pa.binary_view', 'ScalarInequality'),
    ('numerical', 'pa.binary_view', 'FixedIncrements'),
    ('numerical', 'pa.string_view', 'Positive'),
    ('numerical', 'pa.string_view', 'Negative'),
    ('numerical', 'pa.string_view', 'ScalarInequality'),
    ('datetime', 'pd.object', 'ScalarRange'),
    ('datetime', 'pd.category', 'ScalarRange'),
    ('numerical', 'pd.category', 'Inequality'),
    ('numerical', 'pd.category', 'Range'),
    ('numerical', 'pd.datetime64', 'Inequality'),
    ('numerical', 'pd.datetime64', 'Range'),
    ('numerical', 'pd.Period', 'Inequality'),
    ('numerical', 'pd.Period', 'Range'),
    ('numerical', 'np.datetime64', 'Inequality'),
    ('numerical', 'np.datetime64', 'Range'),
    ('numerical', 'pa.bool', 'Inequality'),
    ('numerical', 'pa.bool', 'Range'),
    ('numerical', 'pa.large_binary', 'Inequality'),
    ('numerical', 'pa.large_binary', 'Range'),
    ('numerical', 'pa.date32', 'Inequality'),
    ('numerical', 'pa.date32', 'Range'),
    ('numerical', 'pa.date64', 'Inequality'),
    ('numerical', 'pa.date64', 'Range'),
    ('numerical', 'pa.timestamp', 'Inequality'),
    ('numerical', 'pa.timestamp', 'Range'),
    ('numerical', 'pa.time32', 'Inequality'),
    ('numerical', 'pa.time32', 'Range'),
    ('numerical', 'pa.time64', 'Inequality'),
    ('numerical', 'pa.time64', 'Range'),
    ('numerical', 'pa.string', 'FixedIncrements'),
    ('numerical', 'pa.utf8', 'FixedIncrements'),
    ('numerical', 'pa.large_string', 'FixedIncrements'),
    ('numerical', 'pa.string_view', 'FixedIncrements'),
    ('numerical', 'pa.string', 'Inequality'),
    ('numerical', 'pa.string', 'Range'),
    ('numerical', 'pa.utf8', 'Inequality'),
    ('numerical', 'pa.utf8', 'Range'),
    ('numerical', 'pa.binary', 'Inequality'),
    ('numerical', 'pa.binary', 'Range'),
    ('numerical', 'pa.large_string', 'Inequality'),
    ('numerical', 'pa.large_string', 'Range'),
    ('numerical', 'pa.binary_view', 'Inequality'),
    ('numerical', 'pa.binary_view', 'Range'),
    ('numerical', 'pa.string_view', 'Inequality'),
    ('numerical', 'pa.string_view', 'Range'),
]

EXCLUDED_DATA_TYPES = {
    ('pd.boolean', 'numerical'),
    ('pd.string', 'numerical'),
    ('pd.category', 'numerical'),
    ('pd.Period', 'numerical'),
    ('np.bool', 'numerical'),
    ('np.object', 'numerical'),
    ('np.string', 'numerical'),
    ('np.unicode', 'numerical'),
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
