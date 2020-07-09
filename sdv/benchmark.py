# -*- coding: utf-8 -*-

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


def benchmark(datasets=None, datasets_path=None, distributed=True, timeout=None):
    if datasets is None:
        if datasets_path is None:
            datasets = get_available_demos().name
        else:
            datasets = os.listdir(datasets_path)

    if distributed:
        import dask

        global score_dataset
        score_dataset = dask.delayed(score_dataset)

    scores = list()
    for dataset in datasets:
        scores.append(score_dataset(dataset, datasets_path, timeout))

    if distributed:
        scores = dask.compute(*scores)

    return pd.DataFrame(scores)
