from sdv import SDV, Metadata, load_demo
from sdv.evaluation import evaluate


def test_evaluate_tables_from_demo():
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

    sdv = SDV()
    sdv.fit(new_meta, tables=tables)

    sampled = sdv.sample_all()

    table_scores = dict()
    for table in new_meta.get_tables():
        table_scores[table] = evaluate(
            sampled[table], tables[table], metadata=new_meta, table_name=table)

    score = evaluate(sampled, tables, metadata=new_meta)

    assert isinstance(score, float)
    assert 0 <= score <= 1
