"""Integration tests for the ``DataProcessor``."""
import itertools
import re

import numpy as np
import pandas as pd
import pytest
from rdt.transformers import (
    AnonymizedFaker,
    BinaryEncoder,
    FloatFormatter,
    IDGenerator,
    UniformEncoder,
    UnixTimestampEncoder,
)

from sdv._utils import _get_datetime_format
from sdv.data_processing import DataProcessor
from sdv.data_processing.datetime_formatter import DatetimeFormatter
from sdv.data_processing.numerical_formatter import NumericalFormatter
from sdv.datasets.demo import download_demo
from sdv.errors import SynthesizerInputError
from sdv.metadata import SingleTableMetadata


class TestDataProcessor:
    def test_with_anonymized_columns(self):
        """Test the ``DataProcessor``.

        Test that when we update a ``pii`` column this uses ``AnonymizedFaker`` to create
        a new set of data for that column, while it drops it on the ``transform``.

        Setup:
            - Load metadata and data.
            - Parse the old metadata to a new metadata.
            - Anonymize the field ``occupation``.

        Run:
            - Create a ``DataProcessor`` with the new metadata.
            - Fit the ``DataProcessor`` instance.
            - Transform the data.
            - Reverse transform the data.

        Assert:
            - The column ``occupation`` has been dropped in the ``transform`` process.
            - The column ``occupation`` has been re-created with new values from the
            ``AnonymizedFaker`` instance.
        """
        # Load metadata and data
        data, metadata = download_demo('single_table', 'adult')

        # Add anonymized field
        metadata.update_column('occupation', sdtype='job', pii=True)

        # Instance ``DataProcessor``
        dp = DataProcessor(metadata)

        # Fit
        dp.fit(data)

        # Transform
        transformed = dp.transform(data)

        # Reverse Transform
        reverse_transformed = dp.reverse_transform(transformed)

        # Assert
        assert reverse_transformed.occupation.isin(data.occupation).sum() == 0
        assert 'occupation' not in transformed.columns

    def test_with_anonymized_columns_and_primary_key(self):
        """Test the ``DataProcessor``.

        Test that when we update a ``pii`` column this uses ``AnonymizedFaker`` to create
        a new set of data for that column, while it drops it on the ``transform`` and this values
        can be repeated. Meanwhile, when we set a primary key this has to have a unique length
        of the input data.

        Setup:
            - Load metadata and data.
            - Parse the old metadata to a new metadata.
            - Anonymize the field ``occupation``.
            - Create ``id``.

        Run:
            - Create a ``DataProcessor`` with the new metadata.
            - Fit the ``DataProcessor`` instance.
            - Transform the data.
            - Reverse transform the data.

        Assert:
            - The column ``occupation`` has been dropped in the ``transform`` process.
            - The column ``occupation`` has been re-created with new values from the
            ``AnonymizedFaker`` instance.
            - The column ``id`` has been created in ``transform`` with a unique length of the data.
        """
        # Load metadata and data
        data, metadata = download_demo('single_table', 'adult')

        # Add anonymized field
        metadata.update_column('occupation', sdtype='job', pii=True)

        # Add primary key field
        metadata.add_column('id', sdtype='id', regex_format='ID_\\d{4}[0-9]')
        metadata.set_primary_key('id')

        # Add id
        size = len(data)
        data['id'] = np.arange(0, size).astype('O')

        # Instance ``DataProcessor``
        dp = DataProcessor(metadata)

        # Fit
        dp.fit(data)

        # Transform
        transformed = dp.transform(data)

        # Reverse Transform
        reverse_transformed = dp.reverse_transform(transformed)

        # Assert
        assert transformed.index.name == 'id'
        assert reverse_transformed.occupation.isin(data.occupation).sum() == 0
        assert 'occupation' not in transformed.columns
        assert 'id' not in transformed.columns
        assert reverse_transformed.id.isin(data.id).sum() == 0
        assert len(reverse_transformed.id.unique()) == size

    def test_with_primary_key_numerical(self):
        """End to end test for the ``DataProcessor``.

        Test that when running the ``DataProcessor`` with a numerical primary key, this is able
        to generate such a key in the ``transform`` method. The key should be the same length
        as the input data size.

        Setup:
            - Load metadata and data.
            - Detect the metadata from the ``dataframe``.
            - Create ``id`` column.

        Run:
            - Create a ``DataProcessor`` with the new metadata.
            - Fit the ``DataProcessor`` instance.
            - Transform the data.
            - Reverse transform the data.

        Assert:
            - The column ``id`` has been created during ``transform`` with a unique length
            of the data.
            matching the original numerical ``id`` column.
        """
        # Load metadata and data
        data, _ = download_demo('single_table', 'adult')
        adult_metadata = SingleTableMetadata()
        adult_metadata.detect_from_dataframe(data=data)

        # Add primary key field
        adult_metadata.add_column('id', sdtype='id')
        adult_metadata.set_primary_key('id')

        # Add id
        size = len(data)
        id_generator = itertools.count()
        ids = [next(id_generator) for _ in range(size)]
        data['id'] = ids

        # Instance ``DataProcessor``
        dp = DataProcessor(adult_metadata)

        # Fit
        dp.fit(data)

        # Transform
        transformed = dp.transform(data)

        # Reverse Transform
        reverse_transformed = dp.reverse_transform(transformed)

        # Assert
        assert transformed.index.name == 'id'
        assert 'id' not in transformed.columns
        assert reverse_transformed.index.isin(data.id).sum() == size
        assert len(reverse_transformed.id.unique()) == size

    def test_with_alternate_keys(self):
        """Test that alternate keys are being generated in a unique way.

        Test that the alternate keys are being generated and dropped the same way
        as with the ``primary_key``.
        """
        # Load metadata and data
        data, _ = download_demo('single_table', 'adult')
        data['fnlwgt'] = data['fnlwgt'].astype(str)
        adult_metadata = SingleTableMetadata()
        adult_metadata.detect_from_dataframe(data=data)

        # Add primary key field
        adult_metadata.add_column('id', sdtype='id')
        adult_metadata.set_primary_key('id')

        adult_metadata.add_column('secondary_id', sdtype='id')
        adult_metadata.update_column('fnlwgt', sdtype='id', regex_format='ID_\\d{4}[0-9]')

        adult_metadata.add_alternate_keys(['secondary_id', 'fnlwgt'])

        # Add id
        size = len(data)
        id_generator = itertools.count()
        ids = [next(id_generator) for _ in range(size)]
        data['id'] = ids
        data['secondary_id'] = ids

        # Instance ``DataProcessor``
        dp = DataProcessor(adult_metadata)

        # Fit
        dp.fit(data)

        # Transform
        transformed = dp.transform(data)

        # Reverse Transform
        reverse_transformed = dp.reverse_transform(transformed)

        # Assert
        assert 'id' not in transformed.columns
        assert 'secondary_id' not in transformed.columns
        assert 'fnlwgt' not in transformed.columns
        assert len(reverse_transformed.id.unique()) == size
        assert len(reverse_transformed.secondary_id.unique()) == size
        assert len(reverse_transformed.fnlwgt.unique()) == size

    def test_prepare_for_fitting(self):
        """Test the ``prepare_for_fitting`` method.

        Test that the method sets an expected list of transformers for the given
        data types of the ``metadata`` and also respects the extra parameters
        that those have. In this case the columns ``start_date`` and ``end_date`` have
        a ``datetime_format`` which has to be set to the ``UnixTimestampEncoder`` and
        the column ``salary`` has a ``computer_representation`` set to ``Int64``.
        """
        # Setup
        data, metadata = download_demo(
            modality='single_table',
            dataset_name='student_placements_pii'
        )
        dp = DataProcessor(metadata)

        # Run
        dp.prepare_for_fitting(data)

        # Assert
        field_transformers = dp._hyper_transformer.field_transformers
        expected_transformers = {
            'mba_spec': UniformEncoder,
            'employability_perc': FloatFormatter,
            'placed': UniformEncoder,
            'student_id': AnonymizedFaker,
            'experience_years': FloatFormatter,
            'duration': UniformEncoder,
            'salary': FloatFormatter,
            'second_perc': FloatFormatter,
            'start_date': UnixTimestampEncoder,
            'address': AnonymizedFaker,
            'gender': UniformEncoder,
            'mba_perc': FloatFormatter,
            'degree_type': UniformEncoder,
            'end_date': UnixTimestampEncoder,
            'high_spec': UniformEncoder,
            'high_perc': FloatFormatter,
            'work_experience': UniformEncoder,
            'degree_perc': FloatFormatter
        }
        for column_name, transformer_class in expected_transformers.items():
            if transformer_class is not None:
                assert isinstance(field_transformers[column_name], transformer_class)
            else:
                assert field_transformers[column_name] is None

        assert field_transformers['start_date'].datetime_format == '%Y-%m-%d'
        assert field_transformers['end_date'].datetime_format == '%Y-%m-%d'
        assert field_transformers['salary'].computer_representation == 'Int64'

    def test_reverse_transform_with_formatters(self):
        """End to end test using formatters."""
        # Setup
        data, metadata = download_demo(
            modality='single_table',
            dataset_name='student_placements'
        )
        dp = DataProcessor(metadata)

        # Run
        dp.fit(data)

        transformed = dp.transform(data)
        reverse_transformed = dp.reverse_transform(transformed)
        reverse_transformed = reverse_transformed.drop('student_id', axis=1)
        reverse_transformed = reverse_transformed.reset_index()

        # Assert
        assert isinstance(dp.formatters['degree_perc'], NumericalFormatter)
        assert isinstance(dp.formatters['employability_perc'], NumericalFormatter)
        assert isinstance(dp.formatters['experience_years'], NumericalFormatter)
        assert isinstance(dp.formatters['high_perc'], NumericalFormatter)
        assert isinstance(dp.formatters['mba_perc'], NumericalFormatter)
        assert isinstance(dp.formatters['salary'], NumericalFormatter)
        assert isinstance(dp.formatters['second_perc'], NumericalFormatter)
        assert isinstance(dp.formatters['start_date'], DatetimeFormatter)
        assert isinstance(dp.formatters['end_date'], DatetimeFormatter)

        start_date_data_format = _get_datetime_format(
            data['start_date'][~data['start_date'].isna()][0]
        )
        reversed_start_date = reverse_transformed['start_date'][
            ~reverse_transformed['start_date'].isna()
        ]
        reversed_start_date_format = _get_datetime_format(reversed_start_date.iloc[0])
        assert start_date_data_format == reversed_start_date_format

        end_date_data_format = _get_datetime_format(data['end_date'][~data['end_date'].isna()][0])
        reversed_end_date = reverse_transformed['end_date'][
            ~reverse_transformed['end_date'].isna()
        ]
        reversed_end_date_format = _get_datetime_format(reversed_end_date.iloc[0])
        assert end_date_data_format == reversed_end_date_format

    def test_refit_hypertransformer(self):
        """Test data processor re-fits _hyper_transformer."""
        # Setup
        data, metadata = download_demo(
            modality='single_table',
            dataset_name='student_placements'
        )
        dp = DataProcessor(metadata)

        # Run
        dp.fit(data)
        dp.update_transformers({'placed': BinaryEncoder()})

        # Assert
        assert dp._hyper_transformer._fitted
        assert dp._hyper_transformer._modified_config

        dp.fit(data)

        transformed = dp.transform(data)
        assert all(transformed.dtypes == float)

    def test_localized_anonymized_columns(self):
        """Test data processor uses the default locale for anonymized columns."""
        # Setup
        data, metadata = download_demo('single_table', 'adult')
        metadata.update_column('occupation', sdtype='job', pii=True)

        dp = DataProcessor(metadata, locales=['en_CA', 'fr_CA'])

        # Run
        dp.fit(data)

        # Assert
        assert dp._hyper_transformer.field_transformers['occupation'].locales == ['en_CA', 'fr_CA']

    def test_categorical_column_with_numbers(self):
        """Test that UniformEncoder is assigned for categorical columns defined with numbers."""
        # Setup
        data = pd.DataFrame({
            'category_col': [
                1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2,
                1, 1, 2, 1, 2, 2
            ],
            'numerical_col': np.random.rand(20),
        })

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)

        dp = DataProcessor(metadata)

        # Run
        dp.fit(data)

        # Assert
        assert isinstance(dp._hyper_transformer.field_transformers['category_col'], UniformEncoder)

    def test_update_transformers_id_generator(self):
        """ Test that updating to transformer to id generator is valid"""
        # Setup
        data = pd.DataFrame({
            'user_id': list(range(4)),
            'user_cat': ['a', 'b', 'c', 'd']
        })
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        metadata.update_column('user_id', sdtype='id')
        metadata.set_primary_key('user_id')
        dp = DataProcessor(metadata)
        id_gen = IDGenerator(starting_value=5)
        dp.fit(data)

        # Run
        dp.update_transformers({'user_id': id_gen})
        transformers = dp._hyper_transformer.get_config()['transformers']

        # Assert
        assert transformers['user_id'] == id_gen
        assert type(transformers['user_cat']).__name__ == 'UniformEncoder'

        error_msg = re.escape(
            "Invalid transformer 'UniformEncoder' for a primary or alternate key 'user_id'. "
            'Please use a generator transformer instead.'
        )
        with pytest.raises(SynthesizerInputError, match=error_msg):
            dp.update_transformers({'user_id': UniformEncoder()})
