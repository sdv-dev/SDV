import pandas as pd
from rdt.transformers import FloatFormatter

from sdv.datasets.demo import download_demo
from sdv.evaluation.single_table import evaluate_quality, get_column_pair_plot, get_column_plot
from sdv.metadata import SingleTableMetadata
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


def test_categoricals_are_not_preprocessed():
    """"""
    # Setup
    data = pd.DataFrame(data={
        'age': [56, 61, 36, 52, 42],
        'therapy': [True, False, True, False, True],
        'alcohol': ['medium', 'medium', 'low', 'high', 'low'],
    })
    metadata = SingleTableMetadata.load_from_dict({
        'columns': {
            'age': {'sdtype': 'numerical'},
            'therapy': {'sdtype': 'boolean'},
            'alcohol': {'sdtype': 'categorical'}
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
