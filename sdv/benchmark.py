# -*- coding: utf-8 -*-

"""SDV Benchmarking."""

import logging
import multiprocessing
import os
from datetime import datetime

import pandas as pd

from sdv import SDV, Metadata
from sdv.demo import get_available_demos, load_demo
from sdv.evaluation import evaluate

LOGGER = logging.getLogger(__name__)


def _score_dataset(dataset, datasets_path, output):
    start = datetime.now()

    try:
        if datasets_path is None:
            metadata, tables = load_demo(dataset, metadata=True)
        else:
            metadata = Metadata(os.path.join(datasets_path, dataset, 'metadata.json'))
            tables = metadata.load_tables()

        sdv = SDV()
        LOGGER.info('Modeling dataset %s', dataset)
        sdv.fit(metadata, tables)

        LOGGER.info('Sampling dataset %s', dataset)
        sampled = sdv.sample_all(10)

        LOGGER.info('Evaluating dataset %s', dataset)
        score = evaluate(sampled, metadata=metadata)

        LOGGER.info('%s: %s - ELAPSED: %s', dataset, score, datetime.now() - start)
        output.update({
            'dataset': dataset,
            'score': score,
        })

    except Exception as ex:
        error = '{}: {}'.format(type(ex).__name__, str(ex))
        LOGGER.error('%s: %s - ELAPSED: %s', dataset, error, datetime.now() - start)
        output.update({
            'dataset': dataset,
            'error': error
        })


def score_dataset(dataset, datasets_path, timeout=None):
    """Evaluate the performance of SDV on a dataset.

    Args:
        dataset (str):
            Name of the dataset on which to evaluate SDV.
        datasets_path (str):
            Path where the datasets can be found. If not passed,
            the demo datasets are used.
        timeout (int):
            Maximum number of seconds to wait. If not passed,
            wait until done.

    Returns:
        dict:
            Obtained scores or error.
    """
    with multiprocessing.Manager() as manager:
        output = manager.dict()
        process = multiprocessing.Process(
            target=_score_dataset,
            args=(dataset, datasets_path, output)
        )

        process.start()
        process.join(timeout)
        process.terminate()

        if not output:
            LOGGER.warn('%s: TIMEOUT', dataset)
            return {
                'dataset': dataset,
                'error': 'timeout'
            }

        return dict(output)


def benchmark(datasets=None, datasets_path=None, distributed=False, timeout=None):
    """Evaluate the performance of SDV over a collection of datasets.

    If ``distributed`` is ``True``, ``dask`` must be installed in order to
    distribute the load across multiple processors or workers.

    If a ``timeout`` is given, any dataset that takes longer than the indicated
    number of seconds to evaluate will be skipped and its score will be null.

    Args:
        datasets (list[str]):
            List of names of datasets to run on.
        datasets_path (str):
            Path where the datasets can be found. If not passed,
            the demo datasets are used.
        distributed (bool):
            Whether to use ``dask`` for to distribute the load.
            If ``True``, ``dask`` and ``distributed`` must also
            be installed.
        timeout (int):
            Maximum number of seconds to wait for each dataset to
            finish the evaluation process. If not passed, wait until
            all the datasets are done.

    Returns:
        pandas.DataFrame:
            Obtained scores or error.

    Raises:
        ImportError:
            If ``distributed`` is ``True`` and ``dask`` is not installed.
    """
    if datasets is None:
        if datasets_path is None:
            datasets = get_available_demos().name
        else:
            datasets = os.listdir(datasets_path)

    if distributed:
        try:
            import dask
        except ImportError as ie:
            ie.msg += (
                '\n\nIt seems like `dask` is not installed.\n'
                'Please install dask and distributed using:\n'
                '\n    pip install dask distributed'
            )
            raise

        score_dataset_ = dask.delayed(score_dataset)
    else:
        score_dataset_ = score_dataset

    scores = list()
    for dataset in datasets:
        scores.append(score_dataset_(dataset, datasets_path, timeout))

    if distributed:
        scores = dask.compute(*scores)

    return pd.DataFrame(scores)
