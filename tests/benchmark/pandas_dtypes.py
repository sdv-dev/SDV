import pandas as pd

PANDAS_DTYPES = {
    'pd.Int8': pd.DataFrame({'pd.Int8': pd.Series([1, 2, -3, None, 4, 5], dtype='Int8')}),
    'pd.Int16': pd.DataFrame({'pd.Int16': pd.Series([1, 2, -3, None, 4, 5], dtype='Int16')}),
    'pd.Int32': pd.DataFrame({'pd.Int32': pd.Series([1, 2, -3, None, 4, 5], dtype='Int32')}),
    'pd.Int64': pd.DataFrame({'pd.Int64': pd.Series([1, 2, -3, None, 4, 5], dtype='Int64')}),
    'pd.UInt8': pd.DataFrame({'pd.UInt8': pd.Series([1, 2, 3, None, 4, 5], dtype='UInt8')}),
    'pd.UInt16': pd.DataFrame({'pd.UInt16': pd.Series([1, 2, 3, None, 4, 5], dtype='UInt16')}),
    'pd.UInt32': pd.DataFrame({'pd.UInt32': pd.Series([1, 2, 3, None, 4, 5], dtype='UInt32')}),
    'pd.UInt64': pd.DataFrame({'pd.UInt64': pd.Series([1, 2, 3, None, 4, 5], dtype='UInt64')}),
    'pd.Float32': pd.DataFrame({
        'pd.Float32': pd.Series([1.1, 1.2, 1.3, 1.4, None], dtype='Float32')
    }),
    'pd.Float64': pd.DataFrame({
        'pd.Float64': pd.Series([1.1, 1.2, 1.3, 1.4, None], dtype='Float64')
    }),
    'pd.boolean': pd.DataFrame({
        'pd.boolean': pd.Series([True, False, None, True, False], dtype='boolean')
    }),
    'pd.object': pd.DataFrame({'pd.object': pd.Series(['A', 'B', None, 'C'], dtype='object')}),
    'pd.string': pd.DataFrame({'pd.string': pd.Series(['A', 'B', None, 'C'], dtype='string')}),
    'pd.category': pd.DataFrame({
        'pd.category': pd.Series(['A', 'B', None, 'D'], dtype='category')
    }),
    'pd.datetime64': pd.DataFrame({
        'pd.datetime64': pd.Series(pd.date_range('2023-01-01', periods=3), dtype='datetime64[ns]')
    }),
    'pd.timedelta64': pd.DataFrame({
        'pd.timedelta64': pd.Series(
            [pd.Timedelta(days=1), pd.Timedelta(days=2), pd.Timedelta(days=3)],
            dtype='timedelta64[ns]',
        )
    }),
    'pd.Period': pd.DataFrame({
        'pd.Period': pd.Series(pd.period_range('2023-01', periods=3, freq='M')),
    }),
    'pd.Complex': pd.DataFrame({
        'pd.Complex': pd.Series([1 + 1j, 2 + 2j, 3 + 3j], dtype='complex128'),
    }),
}

PANDAS_DATETIME_DTYPES = {
    'pd.Int8 - datetime': pd.DataFrame({'pd.Int8 - datetime': pd.Series([1, 2, -3, None, 4, 5], dtype='Int8')}),
    'pd.Int16 - datetime': pd.DataFrame({'pd.Int16 - datetime': pd.Series([1, 2, -3, None, 4, 5], dtype='Int16')}),
    'pd.Int32 - datetime': pd.DataFrame({'pd.Int32 - datetime': pd.Series([20240101, 20240630, 20250101, None, 20250601, 20260101], dtype='Int32')}),
    'pd.Int64 - datetime': pd.DataFrame({'pd.Int64 - datetime': pd.Series([20240101, 20240630, 20250101, None, 20250601, 20260101], dtype='Int64')}),
    'pd.UInt8 - datetime': pd.DataFrame({'pd.UInt8 - datetime': pd.Series([1, 2, 3, None, 4, 5], dtype='UInt8')}),
    'pd.UInt16 - datetime': pd.DataFrame({'pd.UInt16 - datetime': pd.Series([1, 2, 3, None, 4, 5], dtype='UInt16')}),
    'pd.UInt32 - datetime': pd.DataFrame({'pd.UInt32 - datetime': pd.Series([20240101, 20240630, 20250101, None, 20250601, 20260101], dtype='UInt32')}),
    'pd.UInt64 - datetime': pd.DataFrame({'pd.UInt64 - datetime': pd.Series([20240101, 20240630, 20250101, None, 20250601, 20260101], dtype='UInt64')}),
    'pd.Float32 - datetime': pd.DataFrame({
        'pd.Float32 - datetime': pd.Series([20240101.0, 20240630.0, 20250101.0, None, 20250601.0, 20260101.0], dtype='Float32')
    }),
    'pd.Float64 - datetime': pd.DataFrame({
        'pd.Float64 - datetime': pd.Series([20240101.0, 20240630.0, 20250101.0, None, 20250601.0, 20260101.0], dtype='Float64')
    }),
    'pd.object - datetime': pd.DataFrame({'pd.object - datetime': pd.Series(['20240101', '20240630', '20250101', None, '20250601', '20260101'], dtype='object')}),
    'pd.string - datetime': pd.DataFrame({'pd.string - datetime': pd.Series(['20240101', '20240630', '20250101', None, '20250601', '20260101'], dtype='string')}),
    'pd.category - datetime': pd.DataFrame({
        'pd.category - datetime': pd.Series(['20240101', '20240630', '20250101', None, '20250601', '20260101'], dtype='category')
    }),
    'pd.datetime64 - datetime': pd.DataFrame({
        'pd.datetime64 - datetime': pd.Series(pd.date_range('2023-01-01', periods=3), dtype='datetime64[ns]')
    }),
    'pd.Period - datetime': pd.DataFrame({
        'pd.Period - datetime': pd.Series(pd.period_range('2023-01', periods=3, freq='M')),
    }),
}
