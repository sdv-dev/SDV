import logging

from sdv.constraints.errors import AggregateConstraintsError


def test_aggregate_constraints_error_logs_errors(caplog):
    """Test that an ``AggregateConstraintsError`` logs the stack trace for all its errors."""
    # Run
    with caplog.at_level(logging.DEBUG, logger='sdv.constraints.errors'):
        error = AggregateConstraintsError(errors=[ValueError('error 1'), ValueError('error 2')])
        message = str(error)

    # Assert
    log_messages = [record[2] for record in caplog.record_tuples]
    assert log_messages == ['ValueError: error 1\n', 'ValueError: error 2\n']
    assert message == '\nerror 1\n\nerror 2'
