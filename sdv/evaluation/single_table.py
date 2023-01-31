
from sdmetrics.reports.single_table.quality_report import QualityReport
from sdmetrics.reports.single_table.diagnostic_report import DiagnosticReport
import sdmetrics.reports.utils as report


def evaluate_quality(real_data, synthetic_data, metadata, verbose=True):
    """Evaluate the quality of the synthetic data.
    
    Args:
        real_data (pd.DataFrame):
            The table containing the real data.
        synthetic_data (pd.DataFrame):
            The table containing the synthetic data.
        metadata (SingleTableMetadata):
            The metadata object describing the real/synthetic data.
        verbose (bool):
            Whether or not to print report summary and progress.
            Defaults to True.
    
    Returns:
        float
            The overall quality score.
    """
    quality_report = QualityReport()
    quality_report.generate(real_data, synthetic_data, metadata.to_dict(), verbose)
    return quality_report.get_score()
    

def run_diagnostic(real_data, synthetic_data, metadata, verbose=True):
    """Wrapper around the initialization and evaluation of this class.
    
    Args:
        real_data (pd.DataFrame):
            The table containing the real data.
        synthetic_data (pd.DataFrame):
            The table containing the synthetic data.
        metadata (SingleTableMetadata):
            The metadata object describing the real/synthetic data.
        verbose (bool):
            Whether or not to print report summary and progress.
            Defaults to True.
    """
    quality_report = DiagnosticReport()
    quality_report.generate(real_data, synthetic_data, metadata.to_dict(), verbose)
    return quality_report.get_results()

def get_column_plot(real_data, synthetic_data, metadata, column_name):
    """Get a plot of the real and synthetic data for a given column.
    
    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_data (pandas.DataFrame):
            The synthetic table data.
        metadata (dict):
            The table metadata.
        column_name (str):
            The name of the column.

    Returns:
        plotly.graph_objects._figure.Figure:
            1D marginal distribution plot (i.e. a histogram) of the columns.
    """
    return report.get_column_plot(real_data, synthetic_data, column_name, metadata.to_dict())

def get_column_pair_plot(real_data, synthetic_data, metadata, column_names):
    """Get a plot of the real and synthetic data for a given column pair.

    Args:
        real_data (pandas.DataFrame):
            The real table data.
        synthetic_column (pandas.Dataframe):
            The synthetic table data.
        metadata (dict):
            The table metadata.
        column_names (list[string]):
            The names of the two columns to plot.

    Returns:
        plotly.graph_objects._figure.Figure:
            2D bivariate distribution plot (i.e. a scatterplot) of the columns.

    """
    return report.get_column_pair_plot(real_data, synthetic_data, column_names, metadata.to_dict())
