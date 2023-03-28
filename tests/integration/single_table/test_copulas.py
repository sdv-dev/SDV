from uuid import UUID

import pandas as pd
from rdt.transformers import (
    AnonymizedFaker, CustomLabelEncoder, FloatFormatter, PseudoAnonymizedFaker)

from sdv.datasets.demo import download_demo
from sdv.evaluation.single_table import evaluate_quality, get_column_pair_plot, get_column_plot
from sdv.sampling import Condition
from sdv.single_table import GaussianCopulaSynthesizer


def test_synthesize_table_gaussian_copula(tmp_path):
    """End to end test for the Gaussian Copula synthesizer.

    Tests fitting and sampling from the synthesizer, anonymization, quality reports, and
    synthesizer customization.
    """
    # Setup
    real_data, metadata = download_demo(
        modality='single_table',
        dataset_name='fake_hotel_guests'
    )
    synthesizer = GaussianCopulaSynthesizer(metadata)
    custom_synthesizer = GaussianCopulaSynthesizer(
        metadata,
        default_distribution='truncnorm',
        numerical_distributions={
            'checkin_date': 'uniform',
            'checkout_date': 'uniform',
            'room_rate': 'gaussian_kde'
        }
    )
    sensitive_columns = ['guest_email', 'billing_address', 'credit_card_number']
    model_path = tmp_path / 'synthesizer.pkl'

    suite_guests_with_rewards = Condition(
        num_rows=250,
        column_values={'room_type': 'SUITE', 'has_rewards': True}
    )

    suite_guests_without_rewards = Condition(
        num_rows=250,
        column_values={'room_type': 'SUITE', 'has_rewards': False}
    )

    # Run - fit
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(num_rows=500)

    # Run - evaluate
    quality_report = evaluate_quality(
        real_data,
        synthetic_data,
        metadata
    )

    column_plot = get_column_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        column_name='room_rate',
        metadata=metadata
    )

    pair_plot = get_column_pair_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        column_names=['room_rate', 'room_type'],
        metadata=metadata
    )

    # Run - save model
    synthesizer.save(model_path)

    # Run - custom synthesizer
    custom_synthesizer.fit(real_data)
    synthetic_data_customized = custom_synthesizer.sample(num_rows=500)
    learned_distributions = custom_synthesizer.get_learned_distributions()
    custom_quality_report = evaluate_quality(
        real_data,
        synthetic_data_customized,
        metadata
    )
    custom_column_plot = get_column_plot(
        real_data=real_data,
        synthetic_data=synthetic_data_customized,
        column_name='room_rate',
        metadata=metadata
    )
    simulated_synthetic_data = custom_synthesizer.sample_from_conditions(conditions=[
        suite_guests_with_rewards,
        suite_guests_without_rewards
    ])

    # Assert - fit
    assert set(real_data.columns) == set(synthetic_data.columns)
    assert real_data.shape[1] == synthetic_data.shape[1]
    assert len(synthetic_data) == 500
    for column in sensitive_columns:
        assert synthetic_data[column].isin(real_data[column]).sum() == 0

    # Assert - evaluate
    assert quality_report.get_score() > 0
    assert column_plot
    assert pair_plot

    # Assert - save/load model
    assert model_path.exists()
    assert model_path.is_file()
    loaded_synthesizer = GaussianCopulaSynthesizer.load(model_path)
    assert isinstance(synthesizer, GaussianCopulaSynthesizer)
    assert loaded_synthesizer.get_info() == synthesizer.get_info()
    assert loaded_synthesizer.metadata.to_dict() == metadata.to_dict()
    loaded_synthesizer.sample(20)

    # Assert - custom synthesizer
    assert custom_quality_report.get_score() > 0
    assert custom_column_plot
    assert list(learned_distributions['has_rewards']['learned_parameters']) == [
        'a',
        'b',
        'loc',
        'scale'
    ]
    assert learned_distributions['has_rewards']['distribution'] == 'truncnorm'
    assert set(real_data.columns) == set(simulated_synthetic_data.columns)
    assert real_data.shape[1] == simulated_synthetic_data.shape[1]


def test_adding_constraints(tmp_path):
    """End to end test for adding constraints to a ``BaseSingleTableSynthesizer``.

    The following functionalities are being tested:
        * Use an ``Inequality`` constraint.
        * Load custom constraint class from a file.
        * Add a custom constraint class to the model.
        * Validate that the custom constraint was applied properly.
        * Save, load and sample from the model storing both custom and pre-defined constraints.
    """
    # Setup
    real_data, metadata = download_demo(
        modality='single_table',
        dataset_name='fake_hotel_guests'
    )

    checkin_lessthan_checkout = {
        'constraint_class': 'Inequality',
        'constraint_parameters': {
            'low_column_name': 'checkin_date',
            'high_column_name': 'checkout_date'
        }
    }
    synthesizer = GaussianCopulaSynthesizer(metadata)

    # Run
    synthesizer.add_constraints([checkin_lessthan_checkout])
    synthesizer.fit(real_data)
    synthetic_data_constrained = synthesizer.sample(500)

    # Assert
    synthetic_dates = synthetic_data_constrained[['checkin_date', 'checkout_date']].dropna()
    checkin_dates = pd.to_datetime(synthetic_dates['checkin_date'])
    checkout_dates = pd.to_datetime(synthetic_dates['checkout_date'])
    violations = checkin_dates >= checkout_dates
    assert all(~violations)

    # Load custom constraint class
    synthesizer.load_custom_constraint_classes(
        'tests/integration/single_table/custom_constraints.py',
        ['IfTrueThenZero']
    )
    rewards_member_no_fee = {
        'constraint_class': 'IfTrueThenZero',
        'constraint_parameters': {
            'column_names': ['has_rewards', 'amenities_fee']
        }
    }
    synthesizer.add_constraints([rewards_member_no_fee])

    # Re-Fit the model
    synthesizer.fit(real_data)
    synthetic_data_custom_constraint = synthesizer.sample(500)

    # Assert
    validation = synthetic_data_custom_constraint[synthetic_data_custom_constraint['has_rewards']]
    assert validation['amenities_fee'].sum() == 0.0

    # Save and Load
    model_path = tmp_path / 'synthesizer.pkl'
    synthesizer.save(model_path)

    # Assert
    assert model_path.exists()
    assert model_path.is_file()
    loaded_synthesizer = GaussianCopulaSynthesizer.load(model_path)

    assert isinstance(loaded_synthesizer, GaussianCopulaSynthesizer)
    assert loaded_synthesizer.get_info() == synthesizer.get_info()
    assert loaded_synthesizer.metadata.to_dict() == metadata.to_dict()
    sampled_data = loaded_synthesizer.sample(100)
    validation = sampled_data[sampled_data['has_rewards']]
    assert validation['amenities_fee'].sum() == 0.0
    synthesizer.validate(sampled_data)
    loaded_synthesizer.validate(sampled_data)


def test_custom_processing_anonymization():
    """End to end testing for custom processing and anonymization.

    Tests the following functionality:
        * Pre-processing data
        * Fitting pre-processed data
        * Modifying transformers
        * Anonymization and pseudo-anonymization
    """
    # Setup
    real_data, metadata = download_demo(
        modality='single_table',
        dataset_name='fake_hotel_guests'
    )
    synthesizer = GaussianCopulaSynthesizer(metadata)
    transformers_synthesizer = GaussianCopulaSynthesizer(metadata)
    anonymization_synthesizer = GaussianCopulaSynthesizer(metadata)

    room_type_transformer = CustomLabelEncoder(
        order=['BASIC', 'DELUXE', 'SUITE'],
        add_noise=True
    )
    amenities_fee_transformer = FloatFormatter(
        learn_rounding_scheme=True,
        enforce_min_max_values=True,
        missing_value_replacement=0.00
    )

    sensitive_columns = ['guest_email', 'billing_address', 'credit_card_number']
    guest_email_transformer = AnonymizedFaker(
        provider_name='misc',
        function_name='uuid4',
        enforce_uniqueness=True
    )
    billing_address_transformer = PseudoAnonymizedFaker(
        provider_name='address',
        function_name='address'
    )

    # Run - Pre-process data
    pre_processed_data = synthesizer.preprocess(real_data)
    synthesizer.fit_processed_data(pre_processed_data)
    default_sample = synthesizer.sample(num_rows=100)

    # Run - Update transformers
    transformers_synthesizer.preprocess(real_data)
    transformers_synthesizer.update_transformers({
        'room_type': room_type_transformer,
        'amenities_fee': amenities_fee_transformer
    })
    transformers_synthesizer.fit(real_data)

    # Run - Anonymization
    anonymization_synthesizer.preprocess(real_data)
    anonymization_synthesizer.update_transformers({
        'guest_email': guest_email_transformer,
        'billing_address': billing_address_transformer
    })
    anonymization_synthesizer.fit(real_data)
    anonymized_sample = anonymization_synthesizer.sample(num_rows=100)

    # Assert - Pre-process data
    assert pre_processed_data.index.name == metadata.primary_key
    assert all(pre_processed_data.dtypes == 'float64')
    for column in sensitive_columns:
        assert default_sample[column].isin(real_data[column]).sum() == 0
        assert all(default_sample[column].value_counts() == 1)

    # Assert - Update transformers
    transformers = transformers_synthesizer.get_transformers()
    assert transformers['room_type'] == room_type_transformer
    assert transformers['amenities_fee'] == amenities_fee_transformer

    # Assert - Anonymization
    anonymized_transformers = anonymization_synthesizer.get_transformers()
    assert anonymized_transformers['guest_email'] == guest_email_transformer
    assert anonymized_transformers['billing_address'] == billing_address_transformer
    assert [UUID(uuid) for uuid in anonymized_sample['guest_email']]
    assert any(anonymized_sample['billing_address'].value_counts() > 1)
