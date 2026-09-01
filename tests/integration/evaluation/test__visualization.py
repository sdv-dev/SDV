import pytest

from sdv.datasets.demo import download_demo
from sdv.evaluation import get_column_pair_plot


@pytest.mark.parametrize('plot_type', ['box', 'heatmap', 'scatter', 'violin'])
def test_column_pair_plot_different_plot_types(plot_type):
    """Test the method with all supported plot types."""
    # Setup
    real_data, metadata = download_demo(modality='multi_table', dataset_name='fake_hotels')

    # Run
    fig = get_column_pair_plot(
        real_data=real_data,
        synthetic_data=real_data.copy(),
        table_name='guests',
        column_names=['room_rate', 'amenities_fee'],
        metadata=metadata,
        sample_size=40,
        plot_type=plot_type,
    )

    # Assert
    assert len(fig.data[0].x) == 40
    assert len(fig.data[1].x) == 40
