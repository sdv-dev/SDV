from operator import attrgetter

from sdv import SDV, load_demo
from tests.integration import datasets


def test_sdv():
    metadata, tables = load_demo(metadata=True)

    sdv = SDV()
    sdv.fit(metadata, tables)

    # Sample all
    sampled = sdv.sample()

    assert set(sampled.keys()) == {'users', 'sessions', 'transactions'}
    assert len(sampled['users']) == 10

    # Sample with children
    sampled = sdv.sample('users', reset_primary_keys=True)

    assert set(sampled.keys()) == {'users', 'sessions', 'transactions'}
    assert len(sampled['users']) == 10

    # Sample without children
    users = sdv.sample('users', sample_children=False)

    assert users.shape == tables['users'].shape
    assert set(users.columns) == set(tables['users'].columns)

    sessions = sdv.sample('sessions', sample_children=False)

    assert sessions.shape == tables['sessions'].shape
    assert set(sessions.columns) == set(tables['sessions'].columns)

    transactions = sdv.sample('transactions', sample_children=False)

    assert transactions.shape == tables['transactions'].shape
    assert set(transactions.columns) == set(tables['transactions'].columns)


def test_sdv_multiparent():
    metadata, tables = load_demo('got_families', metadata=True)

    sdv = SDV()
    sdv.fit(metadata, tables)

    # Sample all
    sampled = sdv.sample()

    assert set(sampled.keys()) == {'characters', 'families', 'character_families'}
    assert len(sampled['characters']) == 7

    # Sample with children
    sampled = sdv.sample('characters', reset_primary_keys=True)

    assert set(sampled.keys()) == {'characters', 'character_families'}
    assert len(sampled['characters']) == 7
    assert 'family_id' in sampled['character_families']

    # Sample without children
    characters = sdv.sample('characters', sample_children=False)

    assert characters.shape == tables['characters'].shape
    assert set(characters.columns) == set(tables['characters'].columns)

    families = sdv.sample('families', sample_children=False)

    assert families.shape == tables['families'].shape
    assert set(families.columns) == set(tables['families'].columns)

    character_families = sdv.sample('character_families', sample_children=False)

    assert character_families.shape == tables['character_families'].shape
    assert set(character_families.columns) == set(tables['character_families'].columns)


def test_sdv_multi_foreign_key():
    """Ensure multi-foreign-key datasets are properly covered.

    Multi-foreign-key datasets are those that have one table with
    2 foreign keys to the same parent.
    """
    metadata, tables = datasets.load_multi_foreign_key()

    sdv = SDV()
    sdv.fit(metadata, tables)

    # Sample all
    sampled = sdv.sample()

    assert set(sampled.keys()) == {'parent', 'child'}
    assert len(sampled['parent']) == 10


def test_integer_categoricals():
    """Ensure integer categoricals are still sampled as integers.

    The origin of this tests can be found in the github issue #194:
    https://github.com/sdv-dev/SDV/issues/194
    """
    metadata, tables = load_demo(metadata=True)
    metadata_dict = metadata.to_dict()
    metadata_dict['tables']['users']['fields']['age'] = {
        'type': 'categorical'
    }

    sdv = SDV()
    sdv.fit(metadata, tables)
    sampled = sdv.sample()

    kind = attrgetter('kind')
    for name, table in tables.items():
        assert (sampled[name].dtypes.apply(kind) == table.dtypes.apply(kind)).all()
