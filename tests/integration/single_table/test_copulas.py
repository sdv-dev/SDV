from sdv.datasets.demo import download_demo
from sdv.evaluation.single_table import evaluate_quality, get_column_pair_plot, get_column_plot
from sdv.sampling import Condition
from sdv.single_table import GaussianCopulaSynthesizer


def test_synthesize_table_gaussian_copula(tmp_path):
    """End to end test for the ``SDV: Synthesize a Table (Gaussian Copula).ipynb``."""
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
