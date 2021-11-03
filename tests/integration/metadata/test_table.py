import pandas as pd

from sdv import SDV


def test_pii_with_single_localization():
    """Test SDV with pii data that specifies one localization."""
    table_name = 'table1'
    num_values = 100

    metadata = {
        'tables': {
            table_name: {
                'fields': {
                    'company': {
                        'type': 'categorical',
                        'pii': True,
                        'pii_category': 'company',
                        'pii_locales': ['en_US'],
                    },
                },
            },
        }
    }
    data = {table_name: pd.DataFrame({'company': [str(i) for i in range(num_values)]})}

    sdv = SDV()
    sdv.fit(metadata, data)

    sampled = sdv.sample(table_name, num_values)
    sampled_company = sampled[table_name]['company']

    # Check that all companies are sampled from the `en_US` locale
    assert ((sampled_company > u'\u0000') & (sampled_company < u'\u007F')).all()


def test_pii_with_multiple_localizations():
    """Test SDV with pii data that specifies multiple localizations."""
    table_name = 'table1'
    num_values = 100

    metadata = {
        'tables': {
            table_name: {
                'fields': {
                    'company': {
                        'type': 'categorical',
                        'pii': True,
                        'pii_category': 'company',
                        'pii_locales': ['en_US', 'zh_CN'],
                    },
                },
            },
        }
    }
    data = {table_name: pd.DataFrame({'company': [str(i) for i in range(num_values)]})}

    sdv = SDV()
    sdv.fit(metadata, data)

    sampled = sdv.sample(table_name, num_values)
    sampled_company = sampled[table_name]['company']

    # Check that we have sampled companies from the `en_US` locale
    assert ((sampled_company > u'\u0000') & (sampled_company < u'\u007F')).any()
    # Check that we have sampled companies from the `zh_CH` locale
    assert ((sampled_company > u'\u4e00') & (sampled_company < u'\u9fff')).any()
