import numpy as np
import pandas as pd

NUMPY_DTYPES = {
    'np.int8': pd.DataFrame({
        'np.int8': pd.Series([np.int8(1), np.int8(-1), np.int8(127)], dtype='int8')
    }),
    'np.int16': pd.DataFrame({
        'np.int16': pd.Series([np.int16(2), np.int16(-2), np.int16(32767)], dtype='int16')
    }),
    'np.int32': pd.DataFrame({
        'np.int32': pd.Series([np.int32(3), np.int32(-3), np.int32(2147483647)], dtype='int32')
    }),
    'np.int64': pd.DataFrame({
        'np.int64': pd.Series([np.int64(4), np.int64(-4), np.int64(922)], dtype='int64')
    }),
    'np.uint8': pd.DataFrame({
        'np.uint8': pd.Series([np.uint8(5), np.uint8(10), np.uint8(255)], dtype='uint8')
    }),
    'np.uint16': pd.DataFrame({
        'np.uint16': pd.Series([np.uint16(6), np.uint16(20), np.uint16(65535)], dtype='uint16')
    }),
    'np.uint32': pd.DataFrame({
        'np.uint32': pd.Series([np.uint32(7), np.uint32(30), np.uint32(42)], dtype='uint32')
    }),
    'np.uint64': pd.DataFrame({
        'np.uint64': pd.Series([np.uint64(8), np.uint64(40), np.uint64(184467)], dtype='uint64')
    }),
    'np.float16': pd.DataFrame({
        'np.float16': pd.Series(
            [np.float16(9.1), np.float16(-9.1), np.float16(65.0)], dtype='float16'
        )
    }),
    'np.float32': pd.DataFrame({
        'np.float32': pd.Series(
            [np.float32(1.2), np.float32(-1.2), np.float32(3.40)], dtype='float32'
        )
    }),
    'np.float64': pd.DataFrame({
        'np.float64': pd.Series(
            [np.float64(1.3), np.float64(-11.3), np.float64(1.7)], dtype='float64'
        )
    }),
    'np.complex64': pd.DataFrame({
        'np.complex64': pd.Series(
            [np.complex64(12 + 1j), np.complex64(-12 - 1j), np.complex64(3.4e38 + 1j)],
            dtype='complex64',
        )
    }),
    'np.complex128': pd.DataFrame({
        'np.complex128': pd.Series(
            [np.complex128(13 + 2j), np.complex128(-13 - 2j), np.complex128(1.7e308 + 2j)],
            dtype='complex128',
        )
    }),
    'np.bool': pd.DataFrame({
        'np.bool': pd.Series([np.bool_(True), np.bool_(False), np.bool_(True)], dtype='bool')
    }),
    'np.object': pd.DataFrame({
        'np.object': pd.Series(['object1', 'object2', 'object3'], dtype='object')
    }),
    'np.string': pd.DataFrame({
        'np.string': pd.Series([
            np.string_('string1'),
            np.string_('string2'),
            np.string_('string3'),
        ])
    }),
    'np.unicode': pd.DataFrame({
        'np.unicode': pd.Series(
            [np.unicode_('unicode1'), np.unicode_('unicode2'), np.unicode_('unicode3')],
            dtype='string',
        )
    }),
    'np.datetime64': pd.DataFrame({
        'np.datetime64': pd.Series([
            np.datetime64('2023-01-01T00:00:00'),
            np.datetime64('2024-01-01T00:00:00'),
            np.datetime64('2025-01-01T00:00:00'),
        ])
    }),
    'np.timedelta64': pd.DataFrame({
        'np.timedelta64': pd.Series(
            [np.timedelta64(1, 'D'), np.timedelta64(2, 'h'), np.timedelta64(3, 'm')],
        )
    }),
}
