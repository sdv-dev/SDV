from sdv import Metadata, load_demo
from sdv.demo import DEMO_METADATA


def test_build_demo_metadata_from_tables():
    """Build metadata from the demo tables.

    Then compare the built metadata with the demo one
    to make sure that they are the same.
    """
    tables = load_demo(metadata=False)

    new_meta = Metadata()
    new_meta.add_table('users', data=tables['users'], primary_key='user_id')
    new_meta.add_table('sessions', data=tables['sessions'], primary_key='session_id',
                       parent='users', foreign_key='user_id')
    transactions_fields = {
        'timestamp': {
            'type': 'datetime',
            'format': '%Y-%m-%d'
        }
    }
    new_meta.add_table('transactions', tables['transactions'],
                       fields_metadata=transactions_fields,
                       primary_key='transaction_id', parent='sessions')

    assert DEMO_METADATA == new_meta.to_dict()


def test_build_demo_metadata_without_tables():
    metadata = Metadata()

    metadata.add_table('users')
    metadata.add_field('users', 'user_id', 'id', 'integer')
    metadata.add_field('users', 'country', 'categorical')
    metadata.add_field('users', 'gender', 'categorical')
    metadata.add_field('users', 'age', 'numerical', 'integer')
    metadata.set_primary_key('users', 'user_id')

    metadata.add_table('sessions')
    metadata.add_field('sessions', 'session_id', 'id', 'integer')
    metadata.add_field('sessions', 'user_id', 'id', 'integer')
    metadata.add_field('sessions', 'device', 'categorical')
    metadata.add_field('sessions', 'os', 'categorical')
    metadata.set_primary_key('sessions', 'session_id')
    metadata.add_relationship('users', 'sessions')

    metadata.add_table('transactions')
    metadata.add_field('transactions', 'transaction_id', 'id', 'integer')
    metadata.add_field('transactions', 'session_id', 'id', 'integer')
    metadata.add_field('transactions', 'timestamp', 'datetime', properties={'format': '%Y-%m-%d'})
    metadata.add_field('transactions', 'amount', 'numerical', 'float')
    metadata.add_field('transactions', 'approved', 'boolean')
    metadata.set_primary_key('transactions', 'transaction_id')
    metadata.add_relationship('sessions', 'transactions')

    assert DEMO_METADATA == metadata.to_dict()
