
import pandas as pd

from sdv.datasets.demo import download_demo
from sdv.evaluation.single_table import evaluate_quality, get_column_pair_plot, run_diagnostic
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer


def test_evaluation():
    """Test ``evaluate_quality`` and ``run_diagnostic``."""
    # Setup
    data = pd.DataFrame({'col': [1, 2, 3]})
    metadata = SingleTableMetadata()
    metadata.add_column('col', sdtype='numerical')
    synthesizer = GaussianCopulaSynthesizer(metadata, default_distribution='truncnorm')

    # Run and Assert
    synthesizer.fit(data)
    samples = synthesizer.sample(10)
    score = evaluate_quality(data, samples, metadata).get_score()
    assert score == 0.8666666666666667

    diagnostic = run_diagnostic(data, samples, metadata).get_results()
    assert diagnostic == {
        'DANGER': ['More than 50% of the synthetic rows are copies of the real data'],
        'SUCCESS': [
            'The synthetic data covers over 90% of the numerical ranges present in the real data',
            'The synthetic data follows over 90% of the min/max boundaries set by the real data'
        ],
        'WARNING': []
    }


def test_column_pair_plot_sample_size_parameter():
    """Test the sample_size parameter for the column pair plot."""
    # Setup
    real_data, metadata = download_demo(
        modality='single_table',
        dataset_name='fake_hotel_guests'
    )
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(len(real_data))

    # Run
    fig = get_column_pair_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        column_names=['room_rate', 'amenities_fee'],
        metadata=metadata,
        sample_size=40
    )

    # Assert
    assert len(synthetic_data) == 500
    assert len(fig.data[0].x) == 40
    assert len(fig.data[1].x) == 40
