import decimal

import pandas as pd
import pyarrow as pa

PYARROW_DTYPES = {
    'pa.int8': pd.DataFrame({
        'pa.int8': pd.Series([1, 2, -3, None, 4, 5], dtype=pd.ArrowDtype(pa.int8()))
    }),
    'pa.int16': pd.DataFrame({
        'pa.int16': pd.Series([1, 2, -3, None, 4, 5], dtype=pd.ArrowDtype(pa.int16()))
    }),
    'pa.int32': pd.DataFrame({
        'pa.int32': pd.Series([1, 2, -3, None, 4, 5], dtype=pd.ArrowDtype(pa.int32()))
    }),
    'pa.int64': pd.DataFrame({
        'pa.int64': pd.Series([1, 2, -3, None, 4, 5], dtype=pd.ArrowDtype(pa.int64()))
    }),
    'pa.uint8': pd.DataFrame({
        'pa.uint8': pd.Series([1, 2, 3, None, 4, 5], dtype=pd.ArrowDtype(pa.uint8()))
    }),
    'pa.uint16': pd.DataFrame({
        'pa.uint16': pd.Series([1, 2, 3, None, 4, 5], dtype=pd.ArrowDtype(pa.uint16()))
    }),
    'pa.uint32': pd.DataFrame({
        'pa.uint32': pd.Series([1, 2, 3, None, 4, 5], dtype=pd.ArrowDtype(pa.uint32()))
    }),
    'pa.uint64': pd.DataFrame({
        'pa.uint64': pd.Series([1, 2, 3, None, 4, 5], dtype=pd.ArrowDtype(pa.uint64()))
    }),
    'pa.float32': pd.DataFrame({
        'pa.float32': pd.Series([1.1, 1.2, 1.3, None, 1.4], dtype=pd.ArrowDtype(pa.float32()))
    }),
    'pa.float64': pd.DataFrame({
        'pa.float64': pd.Series([1.1, 1.2, 1.3, None, 1.4], dtype=pd.ArrowDtype(pa.float64()))
    }),
    'pa.bool': pd.DataFrame({
        'pa.bool': pd.Series([True, False, None, True, False], dtype=pd.ArrowDtype(pa.bool_()))
    }),
    'pa.string': pd.DataFrame({
        'pa.string': pd.Series(['A', 'B', None, 'C'], dtype=pd.ArrowDtype(pa.string()))
    }),
    'pa.utf8': pd.DataFrame({
        'pa.utf8': pd.Series(['A', 'B', None, 'C'], dtype=pd.ArrowDtype(pa.utf8()))
    }),
    'pa.binary': pd.DataFrame({
        'pa.binary': pd.Series(
            [b'binary1', b'binary2', None, b'binary3'], dtype=pd.ArrowDtype(pa.binary())
        )
    }),
    'pa.large_binary': pd.DataFrame({
        'pa.large_binary': pd.Series(
            [b'large_binary1', b'large_binary2', None, b'large_binary3'],
            dtype=pd.ArrowDtype(pa.large_binary()),
        )
    }),
    'pa.large_string': pd.DataFrame({
        'pa.large_string': pd.Series(['A', 'B', None, 'C'], dtype=pd.ArrowDtype(pa.large_string()))
    }),
    'pa.date32': pd.DataFrame({
        'pa.date32': pd.Series(
            [pd.Timestamp('2023-01-01'), pd.Timestamp('2024-01-01'), None],
            dtype=pd.ArrowDtype(pa.date32()),
        )
    }),
    'pa.date64': pd.DataFrame({
        'pa.date64': pd.Series(
            [pd.Timestamp('2023-01-01'), pd.Timestamp('2024-01-01'), None],
            dtype=pd.ArrowDtype(pa.date64()),
        )
    }),
    'pa.timestamp': pd.DataFrame({
        'pa.timestamp': pd.Series(
            [pd.Timestamp('2023-01-01T00:00:00'), pd.Timestamp('2024-01-01T00:00:00'), None],
            dtype=pd.ArrowDtype(pa.timestamp('ms')),
        )
    }),
    'pa.duration': pd.DataFrame({
        'pa.duration': pd.Series(
            [pd.Timedelta(days=1), pd.Timedelta(hours=2), None],
            dtype=pd.ArrowDtype(pa.duration('s')),
        )
    }),
    'pa.time32': pd.DataFrame({
        'pa.time32': pd.Series(
            [
                pd.Timestamp('2023-01-01T01:00:00').time(),
                pd.Timestamp('2023-01-01T02:00:00').time(),
                None,
            ],
            dtype=pd.ArrowDtype(pa.time32('s')),
        )
    }),
    'pa.time64': pd.DataFrame({
        'pa.time64': pd.Series(
            [
                pd.Timestamp('2023-01-01T01:00:00').time(),
                pd.Timestamp('2023-01-01T02:00:00').time(),
                None,
            ],
            dtype=pd.ArrowDtype(pa.time64('ns')),
        )
    }),
    'pa.binary_view': pd.DataFrame({
        'pa.binary_view': pd.Series(
            [b'view1', b'view2', None, b'view3'], dtype=pd.ArrowDtype(pa.binary())
        )
    }),
    'pa.string_view': pd.DataFrame({
        'pa.string_view': pd.Series(['A', 'B', None, 'C'], dtype=pd.ArrowDtype(pa.string()))
    }),
    'pa.decimal128': pd.DataFrame({
        'pa.decimal128': pd.Series(
            [
                decimal.Decimal('123.45'),
                decimal.Decimal('88.90'),
                decimal.Decimal('78.90'),
                decimal.Decimal('98.90'),
                decimal.Decimal('678.90'),
                decimal.Decimal('6.90'),
                None,
            ],
            dtype=pd.ArrowDtype(pa.decimal128(precision=10, scale=2)),
        )
    }),
}
