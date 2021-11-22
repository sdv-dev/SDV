import re

import numpy as np
import pandas as pd

from sdv.metadata import Table


def test_table():
    """Test Table with pii and non-pii columns."""
    num_values = 100

    metadata_dict = {
        'fields': {
            'years_employed': {
                'type': 'numerical',
                'subtype': 'integer',
            },
            'ssn': {
                'type': 'categorical',
                'pii': True,
                'pii_category': 'ssn',
            },
            'company_US': {
                'type': 'categorical',
                'pii': True,
                'pii_category': 'company',
                'pii_locales': ['en_US'],
            },
            'company_US_CN': {
                'type': 'categorical',
                'pii': True,
                'pii_category': 'company',
                'pii_locales': ['en_US', 'zh_CN'],
            },
        },
    }
    data = pd.DataFrame({
        'years_employed': np.random.choice(20, num_values),
        'ssn': [str(i) for i in range(num_values)],
        'company_US': [str(i) for i in range(num_values)],
        'company_US_CN': [str(i) for i in range(num_values)],
    })
    metadata = Table.from_dict(metadata_dict)

    metadata.fit(data)

    transformed = metadata.transform(data)
    assert transformed.dtypes.isin([np.dtype('int32'), np.dtype('int64')]).all()

    reverse_transformed = metadata.reverse_transform(transformed)

    sampled_years_employed = reverse_transformed['years_employed']
    assert sampled_years_employed.dtype == 'int'
    assert ((sampled_years_employed >= 0) & (sampled_years_employed < 20)).all()

    sampled_ssn = reverse_transformed['ssn']
    ssn_regex = re.compile(r'^\d\d\d-\d\d-\d\d\d\d$')
    assert sampled_ssn.dtype == 'object'
    assert sampled_ssn.str.match(ssn_regex).all()

    sampled_company_US = reverse_transformed['company_US']
    assert sampled_company_US.dtype == 'object'
    # Check that all companies are sampled from the `en_US` locale
    assert ((sampled_company_US > u'\u0000') & (sampled_company_US < u'\u007F')).all()

    sampled_company_US_CN = reverse_transformed['company_US_CN']
    assert sampled_company_US_CN.dtype == 'object'
    # Check that we have sampled companies from the `en_US` locale
    assert ((sampled_company_US_CN > u'\u0000') & (sampled_company_US_CN < u'\u007F')).any()
    # Check that we have sampled companies from the `zh_CH` locale
    assert ((sampled_company_US_CN > u'\u4e00') & (sampled_company_US_CN < u'\u9fff')).any()
