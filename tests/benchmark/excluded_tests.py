"""Excluded test combinations."""

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
