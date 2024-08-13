import re

import numpy as np
import pandas as pd
import pytest
from rdt.transformers import FloatFormatter, LabelEncoder

from sdv.datasets.demo import download_demo
from sdv.errors import InvalidDataTypeError
from sdv.evaluation.single_table import evaluate_quality, get_column_pair_plot, get_column_plot
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer


def test__estimate_num_columns():
    """Test the number of columns is estimated correctly."""
    # Setup
    metadata = SingleTableMetadata()
    metadata.add_column('numerical', sdtype='numerical')
    metadata.add_column('categorical', sdtype='categorical')
    metadata.add_column('categorical2', sdtype='categorical')
    metadata.add_column('categorical3', sdtype='categorical')
    metadata.add_column('datetime', sdtype='datetime')
    metadata.add_column('boolean', sdtype='boolean')
    data = pd.DataFrame({
        'numerical': [0.1, 0.2, 0.3],
        'datetime': ['2020-01-01', '2020-01-02', '2020-01-03'],
        'categorical': ['a', 'b', 'b'],
        'categorical2': ['a', 'b', 'b'],
        'categorical3': [float('nan'), np.nan, None],
        'boolean': [True, False, True],
    })
    instance = CTGANSynthesizer(metadata)

    # Run
    instance.auto_assign_transformers(data)
    instance.update_transformers({'categorical2': LabelEncoder()})
    result = instance._estimate_num_columns(data)

    # Assert
    assert result == {
        'numerical': 11,
        'datetime': 11,
        'categorical': 2,
        'categorical2': 11,
        'categorical3': 1,
        'boolean': 2,
    }


def test_synthesize_table_ctgan(tmp_path):
    """End to end test for the CTGAN synthesizer.

    Tests quality reports, anonymization, and customizing the synthesizer.
    """
    # Setup
    real_data, metadata = download_demo(modality='single_table', dataset_name='fake_hotel_guests')
    synthesizer = CTGANSynthesizer(metadata)
    custom_synthesizer = CTGANSynthesizer(metadata, epochs=100)
    sensitive_columns = ['guest_email', 'billing_address', 'credit_card_number']
    model_path = tmp_path / 'synthesizer.pkl'

    # Run - fit
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(num_rows=500)

    # Run - evaluate
    quality_report = evaluate_quality(real_data, synthetic_data, metadata)

    column_plot = get_column_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        column_name='room_type',
        metadata=metadata,
    )

    pair_plot = get_column_pair_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        column_names=['room_rate', 'room_type'],
        metadata=metadata,
    )

    # Run - save model
    synthesizer.save(model_path)

    # Run - custom synthesizer
    custom_synthesizer.fit(real_data)
    synthetic_data_customized = custom_synthesizer.sample(num_rows=500)
    custom_quality_report = evaluate_quality(real_data, synthetic_data_customized, metadata)

    # Assert - fit
    assert set(real_data.columns) == set(synthetic_data.columns)
    assert real_data.shape[1] == synthetic_data.shape[1]
    assert len(synthetic_data) == 500
    for column in sensitive_columns:
        assert synthetic_data[column].isin(real_data[column]).sum() == 0
    loss_values = synthesizer.get_loss_values()
    assert list(loss_values.columns) == ['Epoch', 'Generator Loss', 'Discriminator Loss']
    assert len(loss_values) == 300
    custom_loss_values = custom_synthesizer.get_loss_values()
    assert list(custom_loss_values.columns) == ['Epoch', 'Generator Loss', 'Discriminator Loss']
    assert len(custom_loss_values) == 100

    # Assert - evaluate
    assert quality_report.get_score() > 0
    assert column_plot
    assert pair_plot

    # Assert - save/load model
    assert model_path.exists()
    assert model_path.is_file()
    loaded_synthesizer = CTGANSynthesizer.load(model_path)
    assert isinstance(synthesizer, CTGANSynthesizer)
    assert loaded_synthesizer.get_info() == synthesizer.get_info()
    assert loaded_synthesizer.metadata.to_dict() == metadata._convert_to_single_table().to_dict()
    loaded_synthesizer.sample(20)

    # Assert - custom synthesizer
    assert custom_quality_report.get_score() > 0


def test_categoricals_are_not_preprocessed():
    """Test that ensures categorical data is not preprocessed by the CTGANSynthesizer.

    It verifies that the transformer assignments and data transformations are handled correctly
    for different data types.
    """
    # Setup
    data = pd.DataFrame(
        data={
            'age': [56, 61, 36, 52, 42],
            'therapy': [True, False, True, False, True],
            'alcohol': ['medium', 'medium', 'low', 'high', 'low'],
        }
    )
    metadata = SingleTableMetadata.load_from_dict({
        'columns': {
            'age': {'sdtype': 'numerical'},
            'therapy': {'sdtype': 'boolean'},
            'alcohol': {'sdtype': 'categorical'},
        }
    })

    # Run auto_assign_transformers
    synth1 = CTGANSynthesizer(metadata)
    synth1.auto_assign_transformers(data)
    transformers1 = synth1.get_transformers()

    # Assert
    assert isinstance(transformers1['age'], FloatFormatter)
    assert transformers1['therapy'] == transformers1['alcohol'] is None

    # Run fit
    synth2 = CTGANSynthesizer(metadata, epochs=1)
    synth2.fit(data)
    transformers2 = synth2.get_transformers()

    # Assert
    assert isinstance(transformers2['age'], FloatFormatter)
    assert transformers2['therapy'] == transformers2['alcohol'] is None


def test_categorical_metadata_with_int_data():
    """Test ``CTGANSynthesizer`` with categorical values.

    Based on the issues [#1647](https://github.com/sdv-dev/SDV/issues/1647) and
    [#1648](https://github.com/sdv-dev/SDV/issues/1648), it sets up the metadata for the dataset,
    creates a sample data frame, and then runs the ``CTGANSynthesizer`` to generate synthetic data.
    Finally, it checks if the categorical variables in the synthetic data retain the same
    categories as the original data.
    """
    # Setup
    metadata_dict = {
        'METADATA_SPEC_VERSION': 'SINGLE_TABLE_V1',
        'columns': {
            'A': {'sdtype': 'categorical'},
            'B': {'sdtype': 'numerical'},
            'C': {'sdtype': 'categorical'},
        },
    }

    metadata = SingleTableMetadata.load_from_dict(metadata_dict)
    data = pd.DataFrame({
        'A': list(range(50)),
        'B': list(range(50)),
        'C': [str(i) for i in range(50)],
    })

    # Run
    synth = CTGANSynthesizer(metadata, epochs=10)
    synth.fit(data)
    synthetic_data = synth.sample(1000)

    # Assert
    original_categories = set(data['A'].unique())
    synthetic_categories_for_a = set(synthetic_data['A'].unique())
    new_categories_for_a = synthetic_categories_for_a - original_categories
    recycled_categories_for_a = original_categories & synthetic_categories_for_a

    original_categories = set(data['C'].unique())
    synthetic_categories_for_c = set(synthetic_data['C'].unique())
    new_categories_for_c = synthetic_categories_for_c - original_categories
    recycled_categories_for_c = original_categories & synthetic_categories_for_c

    assert len(new_categories_for_a) == 0
    assert len(recycled_categories_for_a) == 50
    assert len(new_categories_for_c) == 0
    assert len(recycled_categories_for_c) == 50


def test_category_dtype_errors():
    """Test CTGAN and TVAE error if data has 'category' dtype."""
    # Setup
    data, metadata = download_demo('single_table', 'fake_hotel_guests')
    data['room_type'] = data['room_type'].astype('category')
    data['has_rewards'] = data['has_rewards'].astype('category')

    ctgan = CTGANSynthesizer(metadata)
    tvae = TVAESynthesizer(metadata)

    # Run and Assert
    expected_msg = re.escape(
        "Columns ['has_rewards', 'room_type'] are stored as a 'category' type, which is not "
        "supported. Please cast these columns to an 'object' to continue."
    )
    with pytest.raises(InvalidDataTypeError, match=expected_msg):
        ctgan.fit(data)

    with pytest.raises(InvalidDataTypeError, match=expected_msg):
        tvae.fit(data)


def test_ctgansynthesizer_with_constraints_generating_categorical_values():
    """Test that ``CTGANSynthesizer`` does not crash when using constraints.

    Based on the issue [#1717](https://github.com/sdv-dev/SDV/issues/1717) this test
    ensures that the synthesizer does not crash with a constraint that generates ``categorical``
    data.
    """
    # Setup
    data, metadata = download_demo('single_table', 'student_placements')
    my_synthesizer = CTGANSynthesizer(metadata)
    my_constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {'column_names': ['high_spec', 'degree_type']},
    }
    my_synthesizer.add_constraints(constraints=[my_constraint])

    # Run
    my_synthesizer.fit(data)

    # Assert
    sampled_data = my_synthesizer.sample(10)
    assert len(sampled_data) == 10


def test_ctgan_with_dropped_columns():
    """Test CTGANSynthesizer doesn't crash when applied to columns that will be dropped. GH#1741"""
    # Setup
    data = pd.DataFrame(
        data={
            'user_id': ['100', '101', '102', '103', '104'],
            'user_ssn': ['111-11-1111', '222-22-2222', '333-33-3333', '444-44-4444', '555-55-5555'],
        }
    )

    metadata_dict = {
        'primary_key': 'user_id',
        'columns': {'user_id': {'sdtype': 'id'}, 'user_ssn': {'sdtype': 'ssn'}},
    }

    metadata = SingleTableMetadata.load_from_dict(metadata_dict)

    # Run
    synth = CTGANSynthesizer(metadata)
    synth.fit(data)
    samples = synth.sample(10)

    # Assert
    assert len(samples) == 10
    assert samples.columns.tolist() == ['user_id', 'user_ssn']
    assert all(id_val.startswith('sdv-id-') for id_val in samples['user_id'])
    pd.testing.assert_series_equal(
        samples['user_id'],
        pd.Series(
            [
                'sdv-id-IOsBJZ',
                'sdv-id-CFcIuA',
                'sdv-id-prYgtc',
                'sdv-id-yrTTYM',
                'sdv-id-kLtfIW',
                'sdv-id-nCFkOi',
                'sdv-id-kKQXYV',
                'sdv-id-aPRybP',
                'sdv-id-RHPiGX',
                'sdv-id-SJNtGY',
            ],
            name='user_id',
        ),
    )
