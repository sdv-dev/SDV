from unittest import TestCase
from unittest.mock import Mock, patch
from sdv.evaluation import evaluate
import sdmetrics
import pytest

import pandas as pd


class TestSDV(TestCase):

    def test_evaluate_single_table(self):
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
        metadata={'fields': {
            'col1': {'type': 'numerical', 'subtype': 'integer'},
            'col2': {'type': 'numerical', 'subtype': 'integer'}
        }}
        sdmetrics.compute_metrics.assert_called_once_with(metrics, data, data, metadata=metadata)
        assert score == 1
