from unittest import TestCase
from unittest.mock import Mock

import pandas as pd
import sdmetrics

from sdv.evaluation import evaluate
from sdv.metadata.dataset import Metadata


class TestSDV(TestCase):

    def test_evaluate_single_table(self):
        """Test the ``evaluate`` method for single tables.

        Ensure the default metrics are called when no metrics are passed.

        Setup:
            - Mock ``sdmetrics.compute_metrics`` (but keep it's default behavior).
        Input:
            - The same dataframe twice.
        Output:
            - Score should be 1.
        Side Effect:
            - ``sdmetrics.compute_metrics`` should be called with the correct arguments.
        """
        # Setup
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        sdmetrics.compute_metrics = Mock(wraps=sdmetrics.compute_metrics)

        # Run
        score = evaluate(data, data)

        # Assert
        metrics = {
            'KSComplement': sdmetrics.single_table.multi_single_column.KSComplement,
            'CSTest': sdmetrics.single_table.multi_single_column.CSTest
        }
        metadata = {'fields': {
            'col1': {'type': 'numerical', 'subtype': 'integer'},
            'col2': {'type': 'numerical', 'subtype': 'integer'}
        }}
        sdmetrics.compute_metrics.assert_called_once_with(metrics, data, data, metadata=metadata)
        assert score == 1

    def test_evaluate_multi_table(self):
        """Test the ``evaluate`` method for mutlti tables.

        Ensure the default metrics are called when no metrics are passed.

        Setup:
            - Mock ``sdmetrics.compute_metrics`` (but keep it's default behavior).
        Input:
            - Dictionary of two dataframes.
            - The same dictionary of dataframes.
        Output:
            - Score should be 1.
        Side Effect:
            - ``sdmetrics.compute_metrics`` should be called with the correct arguments.
        """
        # Setup
        table = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        data = {'table1': table, 'table2': table}
        metadata = Metadata()
        metadata.add_table('table1', table)
        metadata.add_table('table2', table)
        sdmetrics.compute_metrics = Mock(wraps=sdmetrics.compute_metrics)

        # Run
        score = evaluate(data, data, metadata=metadata)

        # Assert
        metrics = {
            'KSComplement': sdmetrics.multi_table.multi_single_table.KSComplement,
            'CSTest': sdmetrics.multi_table.multi_single_table.CSTest
        }
        metadata = {'tables': {
            'table1': {
                'fields': {
                    'col1': {'type': 'numerical', 'subtype': 'integer'},
                    'col2': {'type': 'numerical', 'subtype': 'integer'}
                }
            },
            'table2': {
                'fields': {
                    'col1': {'type': 'numerical', 'subtype': 'integer'},
                    'col2': {'type': 'numerical', 'subtype': 'integer'}
                }
            }
        }}
        sdmetrics.compute_metrics.assert_called_once_with(metrics, data, data, metadata=metadata)
        assert score == 1

    def test_evaluate_single_table_with_metadata(self):
        """Test the ``evaluate`` method for single tables when metadata is passed.

        Setup:
            - Mock ``sdmetrics.compute_metrics`` (but keep it's default behavior).
        Input:
            - The same dataframe twice.
            - Metadata with one table.
        Output:
            - Score should be 1.
        Side Effect:
            - ``sdmetrics.compute_metrics`` should be called with the correct arguments.
        """
        # Setup
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        metadata = Metadata()
        metadata.add_table('table', data)
        sdmetrics.compute_metrics = Mock(wraps=sdmetrics.compute_metrics)

        # Run
        score = evaluate(data, data, metadata)

        # Assert
        metrics = {
            'KSComplement': sdmetrics.single_table.multi_single_column.KSComplement,
            'CSTest': sdmetrics.single_table.multi_single_column.CSTest
        }
        metadata = {'fields': {
            'col1': {'type': 'numerical', 'subtype': 'integer'},
            'col2': {'type': 'numerical', 'subtype': 'integer'}
        }}
        sdmetrics.compute_metrics.assert_called_once_with(metrics, data, data, metadata=metadata)
        assert score == 1
