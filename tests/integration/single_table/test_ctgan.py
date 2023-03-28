from sdv.datasets.demo import download_demo
from sdv.evaluation.single_table import evaluate_quality, get_column_pair_plot, get_column_plot
from sdv.single_table import CTGANSynthesizer


def test_synthesize_table_ctgan(tmp_path):
    """End to end test for the CTGAN synthesizer.

    Tests quality reports, anonymization, and customizing the synthesizer.
    """
    # Setup
    real_data, metadata = download_demo(
        modality='single_table',
        dataset_name='fake_hotel_guests'
    )
    synthesizer = CTGANSynthesizer(metadata)
    custom_synthesizer = CTGANSynthesizer(
        metadata,
        epochs=100
    )
    sensitive_columns = ['guest_email', 'billing_address', 'credit_card_number']
    model_path = tmp_path / 'synthesizer.pkl'

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
        column_name='room_type',
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
    custom_quality_report = evaluate_quality(
        real_data,
        synthetic_data_customized,
        metadata
    )

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
    loaded_synthesizer = CTGANSynthesizer.load(model_path)
    assert isinstance(synthesizer, CTGANSynthesizer)
    assert loaded_synthesizer.get_info() == synthesizer.get_info()
    assert loaded_synthesizer.metadata.to_dict() == metadata.to_dict()
    loaded_synthesizer.sample(20)

    # Assert - custom synthesizer
    assert custom_quality_report.get_score() > 0
