"""Methods to compare the real and synthetic data for single-table."""

from sdmetrics import visualization
from sdmetrics.reports.single_table.diagnostic_report import DiagnosticReport
from sdmetrics.reports.single_table.quality_report import QualityReport

from sdv.errors import VisualizationUnavailableError
from sdv.evaluation._utils import _prepare_data_visualization
from sdv.metadata.metadata import Metadata


def evaluate_quality(real_data, synthetic_data, metadata, verbose=True):
    """Evaluate the quality of the synthetic data.

    Args:
        real_data (pd.DataFrame):
            The table containing the real data.
        synthetic_data (pd.DataFrame):
            The table containing the synthetic data.
        metadata (Metadata):
            The metadata object describing the real/synthetic data.
        verbose (bool):
            Whether or not to print report summary and progress.
            Defaults to True.

    Returns:
        QualityReport:
            Single table quality report object.
    """
    if isinstance(metadata, Metadata):
        metadata = metadata._convert_to_single_table()

    quality_report = QualityReport()
    if isinstance(metadata, Metadata):
        metadata = metadata._convert_to_single_table()

    quality_report.generate(real_data, synthetic_data, metadata.to_dict(), verbose)
    return quality_report


def run_diagnostic(real_data, synthetic_data, metadata, verbose=True):
    """Run diagnostic report for the synthetic data.

    Args:
        real_data (pd.DataFrame):
            The table containing the real data.
        synthetic_data (pd.DataFrame):
            The table containing the synthetic data.
        metadata (Metadata):
            The metadata object describing the real/synthetic data.
        verbose (bool):
            Whether or not to print report summary and progress.
            Defaults to True.

    Returns:
        DiagnosticReport:
            Single table diagnostic report object.
    """
    diagnostic_report = DiagnosticReport()
    if isinstance(metadata, Metadata):
        metadata = metadata._convert_to_single_table()

    diagnostic_report.generate(real_data, synthetic_data, metadata.to_dict(), verbose)
    return diagnostic_report


def get_column_plot(real_data, synthetic_data, metadata, column_name, plot_type=None):
    """Get a plot of the real and synthetic data for a given column.

    Args:
        real_data (pandas.DataFrame or None):
            The real table data.
        synthetic_data (pandas.DataFrame or None):
            The synthetic table data.
        metadata (Metadata):
            The table metadata.
        column_name (str):
            The name of the column.
        plot_type (str or None):
            The plot to be used. Can choose between ``distplot``, ``bar`` or ``None``. If ``None`
            select between ``distplot`` or ``bar`` depending on the data that the column contains,
            ``distplot`` for datetime and numerical values and ``bar`` for categorical.
            Defaults to ``None``.

    Returns:
        plotly.graph_objects._figure.Figure:
            1D marginal distribution plot (i.e. a histogram) of the columns.
    """
    if isinstance(metadata, Metadata):
        metadata = metadata._convert_to_single_table()

    sdtype = metadata.columns.get(column_name)['sdtype']
    if plot_type is None:
        if sdtype in ['datetime', 'numerical']:
            plot_type = 'distplot'
        elif sdtype in ['categorical', 'boolean']:
            plot_type = 'bar'

        else:
            raise VisualizationUnavailableError(
                f"The column '{column_name}' has sdtype '{sdtype}', which does not have a "
                'supported visualization. To visualize this data anyways, please add a '
                "'plot_type'."
            )

    real_data = _prepare_data_visualization(real_data, metadata, column_name, None)
    synthetic_data = _prepare_data_visualization(synthetic_data, metadata, column_name, None)

    return visualization.get_column_plot(
        real_data, synthetic_data, column_name, plot_type=plot_type
    )


def get_column_pair_plot(
    real_data, synthetic_data, metadata, column_names, plot_type=None, sample_size=None
):
    """Get a plot of the real and synthetic data for a given column pair.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_column (pandas.Dataframe):
            The synthetic table data.
        metadata (Metadata):
            The table metadata.
        column_names (list[string]):
            The names of the two columns to plot.
        plot_type (str or None):
            The plot to be used. Can choose between ``box``, ``heatmap``, ``scatter`` or ``None``.
            If ``None` select between ``box``, ``heatmap`` or ``scatter`` depending on the data
            that the column contains, ``scatter`` used for datetime and numerical values,
            ``heatmap`` for categorical and ``box`` for a mix of both. Defaults to ``None``.
        sample_size (int or None):
            The number of samples to use for the plot. If ``None`` use the whole dataset.
            Defaults to ``None``.

    Returns:
        plotly.graph_objects._figure.Figure:
            2D bivariate distribution plot (i.e. a scatterplot) of the columns.
    """
    if isinstance(metadata, Metadata):
        metadata = metadata._convert_to_single_table()

    if plot_type is None:
        plot_type = []
        for column_name in column_names:
            sdtype = metadata.columns.get(column_name)['sdtype']
            if sdtype in ['numerical', 'datetime']:
                plot_type.append('scatter')
            elif sdtype in ['categorical', 'boolean']:
                plot_type.append('heatmap')
            else:
                raise VisualizationUnavailableError(
                    f"The column '{column_name}' has sdtype '{sdtype}', which does not have a "
                    'supported visualization. To visualize this data anyways, please add a '
                    "'plot_type'."
                )

        if len(set(plot_type)) > 1:
            plot_type = 'box'
        else:
            plot_type = plot_type.pop()

    real_data = _prepare_data_visualization(real_data, metadata, column_names, sample_size)
    synthetic_data = _prepare_data_visualization(
        synthetic_data, metadata, column_names, sample_size
    )

    return visualization.get_column_pair_plot(real_data, synthetic_data, column_names, plot_type)

from sdv.datasets.demo import download_demo, get_available_demos

def plot_learning_curve(real_data, synthetic_generator, metadata, train_sizes=None, n_splits=5, 
                       random_state=None):
    """Plot learning curve showing how synthetic data quality varies with training data size.
    
    Args:
        real_data (pandas.DataFrame):
            The complete real dataset to use for generating learning curves.
        synthetic_generator (BaseGenerator):
            An instance of a synthetic data generator with a sample() method that takes
            a num_rows parameter. The generator should already be fitted on the appropriate data.
        metadata (Metadata):
            The metadata object describing the real/synthetic data structure.
        train_sizes (array-like or None):
            List of floats between 0.0 and 1.0 representing training set sizes to evaluate.
            If None, defaults to np.linspace(0.1, 1.0, 5). Defaults to None.
        n_splits (int):
            Number of times to repeat evaluation for each training size to compute confidence 
            intervals. Defaults to 5.
        random_state (int or None):
            Random seed for reproducibility. Defaults to None.
            
    Returns:
        plotly.graph_objects._figure.Figure:
            Interactive plot showing learning curves with confidence intervals.
    """
    
    # Set default training sizes if none provided
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)
    
    # Initialize arrays to store scores
    n_sizes = len(train_sizes)
    scores = np.zeros((n_splits, n_sizes))
    
    # Create cross-validation splits for each training size
    cv = ShuffleSplit(n_splits=n_splits, random_state=random_state)
    
    # For each training size
    for size_idx, train_size in enumerate(train_sizes):
        # Calculate actual number of samples for this training size
        n_samples = int(train_size * len(real_data))
        
        # For each CV split
        for split_idx, (train_idx, _) in enumerate(cv.split(real_data)):
            # Sample training data
            train_data = real_data.iloc[train_idx[:n_samples]]
            
            # Generate synthetic data using the same number of rows as training data
            synthetic_data = synthetic_generator.sample(num_rows=len(train_data))
            
            # Evaluate quality
            quality_report = evaluate_quality(train_data, synthetic_data, metadata, verbose=False)
            
            # Store score
            scores[split_idx, size_idx] = quality_report.get_score()
    
    # Calculate mean and std of scores
    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)
    
    # Create plot
    fig = go.Figure()
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=mean_scores,
        name='Quality Score',
        line=dict(color='blue'),
        mode='lines+markers'
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([mean_scores + std_scores, (mean_scores - std_scores)[::-1]]),
        fill='toself',
        fillcolor='blue',
        opacity=0.2,
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        name='Quality Score (±1 std)'
    ))
    
    # Update layout
    fig.update_layout(
        title='Synthetic Data Quality Learning Curve',
        xaxis_title='Training Set Size (fraction of full dataset)',
        yaxis_title='Quality Score',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig