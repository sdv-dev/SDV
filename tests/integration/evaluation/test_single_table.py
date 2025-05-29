import pandas as pd
import pytest

from sdv.datasets.demo import download_demo
from sdv.evaluation.single_table import evaluate_quality, get_column_pair_plot, run_diagnostic
from sdv.metadata.metadata import Metadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer


def test_evaluation():
    """Test ``evaluate_quality`` and ``run_diagnostic``."""
    # Setup
    data = pd.DataFrame({'col': [1, 2, 3]})
    metadata = Metadata()
    metadata.add_table('table')
    metadata.add_column('col', 'table', sdtype='numerical')
    synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution='truncnorm')

    # Run and Assert
    synthesizer.fit(data)
    samples = synthesizer.sample(10)
    score = evaluate_quality(data, samples, metadata).get_score()
    assert score == 0.8666666666666667

    report = run_diagnostic(data, samples, metadata)
    assert report.get_score() == 1
    pd.testing.assert_frame_equal(
        report.get_properties(),
        pd.DataFrame({
            'Property': ['Data Validity', 'Data Structure'],
            'Score': [1.0, 1.0],
        }),
    )


def test_evaluation_metadata():
    """Test ``evaluate_quality`` and ``run_diagnostic`` with Metadata."""
    # Setup
    data = pd.DataFrame({'col': [1, 2, 3]})
    metadata_dict = {'columns': {'col': {'sdtype': 'numerical'}}}
    metadata = Metadata.load_from_dict(metadata_dict)
    synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution='truncnorm')

    # Run and Assert
    synthesizer.fit(data)
    samples = synthesizer.sample(10)
    score = evaluate_quality(data, samples, metadata).get_score()
    assert score == 0.8666666666666667

    report = run_diagnostic(data, samples, metadata)
    assert report.get_score() == 1
    pd.testing.assert_frame_equal(
        report.get_properties(),
        pd.DataFrame({
            'Property': ['Data Validity', 'Data Structure'],
            'Score': [1.0, 1.0],
        }),
    )


def test_column_pair_plot_sample_size_parameter():
    """Test the sample_size parameter for the column pair plot."""
    # Setup
    real_data, metadata = download_demo(modality='single_table', dataset_name='fake_hotel_guests')
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(len(real_data))

    # Run
    fig = get_column_pair_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        column_names=['room_rate', 'amenities_fee'],
        metadata=metadata,
        sample_size=40,
    )

    # Assert
    assert len(synthetic_data) == 500
    assert len(fig.data[0].x) == 40
    assert len(fig.data[1].x) == 40


@pytest.mark.parametrize('plot_type', ['box', 'heatmap', 'scatter', 'violin'])
def test_column_pair_plot_different_plot_types(plot_type):
    """Test the method with all supported plot types."""
    # Setup
    real_data, metadata = download_demo(modality='single_table', dataset_name='fake_hotel_guests')
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(len(real_data))

    # Run
    fig = get_column_pair_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        column_names=['room_rate', 'amenities_fee'],
        metadata=metadata,
        sample_size=40,
        plot_type=plot_type,
    )

    # Assert
    assert len(synthetic_data) == 500
    assert len(fig.data[0].x) == 40
    assert len(fig.data[1].x) == 40
