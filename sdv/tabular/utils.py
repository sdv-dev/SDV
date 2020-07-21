"""Utility functions for tabular models."""

import numpy as np

IGNORED_DICT_KEYS = ['fitted', 'distribution', 'type']


def flatten_array(nested, prefix=''):
    """Flatten an array as a dict.

    Args:
        nested (list, numpy.array):
            Iterable to flatten.
        prefix (str):
            Name to append to the array indices. Defaults to ``''``.

    Returns:
        dict:
            Flattened array.
    """
    result = dict()
    for index in range(len(nested)):
        prefix_key = '__'.join([prefix, str(index)]) if len(prefix) else str(index)

        value = nested[index]
        if isinstance(value, (list, np.ndarray)):
            result.update(flatten_array(value, prefix=prefix_key))

        elif isinstance(value, dict):
            result.update(flatten_dict(value, prefix=prefix_key))

        else:
            result[prefix_key] = value

    return result


def flatten_dict(nested, prefix=''):
    """Flatten a dictionary.

    This method returns a flatten version of a dictionary, concatenating key names with
    double underscores.

    Args:
        nested (dict):
            Original dictionary to flatten.
        prefix (str):
            Prefix to append to key name. Defaults to ``''``.

    Returns:
        dict:
            Flattened dictionary.
    """
    result = dict()

    for key, value in nested.items():
        prefix_key = '__'.join([prefix, str(key)]) if len(prefix) else key

        if key in IGNORED_DICT_KEYS and not isinstance(value, (dict, list)):
            continue

        elif isinstance(value, dict):
            result.update(flatten_dict(value, prefix_key))

        elif isinstance(value, (np.ndarray, list)):
            result.update(flatten_array(value, prefix_key))

        else:
            result[prefix_key] = value

    return result


def _key_order(key_value):
    parts = list()
    for part in key_value[0].split('__'):
        if part.isdigit():
            part = int(part)

        parts.append(part)

    return parts


def unflatten_dict(flat):
    """Transform a flattened dict into its original form.

    Args:
        flat (dict):
            Flattened dict.

    Returns:
        dict:
            Nested dict (if corresponds)
    """
    unflattened = dict()

    for key, value in sorted(flat.items(), key=_key_order):
        if '__' in key:
            key, subkey = key.split('__', 1)
            subkey, name = subkey.rsplit('__', 1)

            if name.isdigit():
                column_index = int(name)
                row_index = int(subkey)

                array = unflattened.setdefault(key, list())

                if len(array) == row_index:
                    row = list()
                    array.append(row)
                elif len(array) == row_index + 1:
                    row = array[row_index]
                else:
                    # This should never happen
                    raise ValueError('There was an error unflattening the extension.')

                if len(row) == column_index:
                    row.append(value)
                else:
                    # This should never happen
                    raise ValueError('There was an error unflattening the extension.')

            else:
                subdict = unflattened.setdefault(key, dict())
                if subkey.isdigit():
                    subkey = int(subkey)

                inner = subdict.setdefault(subkey, dict())
                inner[name] = value

        else:
            unflattened[key] = value

    return unflattened


def impute(data):
    """Fill null values with the mean (numerical) or the mode (categorical)."""
    for column in data:
        column_data = data[column]
        if column_data.dtype in (np.int, np.float):
            fill_value = column_data.mean()
        else:
            fill_value = column_data.mode()[0]

        data[column] = data[column].fillna(fill_value)

    return data


def square_matrix(triangular_matrix):
    """Fill with zeros a triangular matrix to reshape it to a square one.

    Args:
        triangular_matrix (list [list [float]]):
            Array of arrays of

    Returns:
        list:
            Square matrix.
    """
    length = len(triangular_matrix)
    zero = [0.0]

    for item in triangular_matrix:
        item.extend(zero * (length - len(item)))

    return triangular_matrix


def check_matrix_symmetric_positive_definite(matrix):
    """Check if a matrix is symmetric positive-definite.

    Args:
        matrix (list or numpy.ndarray):
            Matrix to evaluate.

    Returns:
        bool
    """
    try:
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            # Not 2-dimensional or square, so not simmetric.
            return False

        np.linalg.cholesky(matrix)
        return True

    except np.linalg.LinAlgError:
        return False


def make_positive_definite(matrix):
    """Find the nearest positive-definite matrix to input.

    Args:
        matrix (numpy.ndarray):
            Matrix to transform

    Returns:
        numpy.ndarray:
            Closest symetric positive-definite matrix.
    """
    symetric_matrix = (matrix + matrix.T) / 2
    _, s, V = np.linalg.svd(symetric_matrix)
    symmetric_polar = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (symetric_matrix + symmetric_polar) / 2
    A3 = (A2 + A2.T) / 2

    if check_matrix_symmetric_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(matrix))
    identity = np.eye(matrix.shape[0])
    iterations = 1
    while not check_matrix_symmetric_positive_definite(A3):
        min_eigenvals = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += identity * (-min_eigenvals * iterations**2 + spacing)
        iterations += 1

    return A3
