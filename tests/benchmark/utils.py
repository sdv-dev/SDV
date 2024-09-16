"""Utility functions for the benchmarking."""

import argparse
import json
import os
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

import git
import pandas as pd

from sdv.io.local import CSVHandler
from tests._external.gdrive_utils import get_latest_file, read_excel, save_to_gdrive
from tests._external.slack_utils import post_slack_message

GDRIVE_OUTPUT_FOLDER = '16SkTOyQ3xkJDPJbyZCusb168JwreW5bm'
PYTHON_VERSION = f'{sys.version_info.major}.{sys.version_info.minor}'
TEMPRESULTS = Path(f'results/{sys.version_info.major}.{sys.version_info.minor}.json')


def get_previous_dtype_result(dtype, method):
    """Return previous result for a given ``dtype`` and method."""
    data = get_previous_results()
    df = data[PYTHON_VERSION]
    try:
        filtered_row = df[df['dtype'] == dtype]
        value = filtered_row[method].to_numpy()[0]
    except IndexError:
        value = False

    return value


@lru_cache()
def get_previous_results():
    """Get the last run for the dtype benchmarking."""
    latest_file = get_latest_file(GDRIVE_OUTPUT_FOLDER)
    df = read_excel(latest_file['id'])
    return df


def _load_temp_results(filename):
    current_results = pd.read_json(filename)
    current_results = current_results.T.reset_index()
    current_results = current_results.rename(columns={'index': 'dtype'})
    return current_results


def _get_output_filename():
    repo = git.Repo(search_parent_directories=True)
    commit_id = repo.head.object.hexsha
    today = str(date.today())
    output_filename = f'{today}-{commit_id}'
    return output_filename


def compare_previous_result_with_current(args):
    """Compare the previous result with the current and post a message on slack."""
    output_filename = TEMPRESULTS
    current_results = _load_temp_results(output_filename)
    output_filename = _get_output_filename()
    csv_output = Path(f'results/{PYTHON_VERSION}.csv')
    current_results.to_csv(csv_output, index=False)

    new_supported_dtypes = []
    unsupported_dtypes = []

    for index, row in current_results.iterrows():
        dtype = row['dtype']
        for col in current_results.columns[1:]:
            current_value = row[col]
            stored_value = get_previous_dtype_result(dtype, col)

            if current_value and not stored_value:
                new_supported_dtypes.append(dtype)

            elif not current_value and stored_value:
                unsupported_dtypes.append(dtype)

    slack_message = ''
    if new_supported_dtypes:
        slack_message += (
            f':party_blob: New data types supported for python: {PYTHON_VERSION} '
            f'{set(new_supported_dtypes)}\n'
       )

    if unsupported_dtypes:
        slack_message += (
            f':party_blob: New data types supported for python: {PYTHON_VERSION} '
            f'{set(new_supported_dtypes)}\n'
       )
        slack_message += (
            f':fire: New unsupported data types for python: {PYTHON_VERSION} for the following '
            f'dtypes: {set(unsupported_dtype)}\n'
        )

    if slack_message:
        post_slack_message('sdv-alerts-debug', slack_message)
    else:
        slack_message = ':party_parrot: No new changes to the DTypes in SDV.'
        post_slack_message('sdv-alerts-debug', slack_message)


def save_results_to_json(results, filename=TEMPRESULTS):
    dtype = results.pop('dtype')
    if os.path.exists(filename):
        # Load existing data
        with open(filename, 'r') as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                json_data = {}

        if dtype in json_data:
            json_data[dtype].update(results)
        else:
            json_data[dtype] = results

        with open(filename, 'w') as file:
            json.dump(json_data, file, indent=4)


def store_results_in_gdrive(args):
    csv_handler = CSVHandler()
    results = csv_handler.read('results/')
    save_to_gdrive(GDRIVE_OUTPUT_FOLDER, results)


def _get_parser():
    """Return argparser to capture inputs from the command line."""
    parser = argparse.ArgumentParser(description='Benchmark utility arg parsing')
    parser.set_defaults(action=None)
    action = parser.add_subparsers(title='action')
    action.required = True

    # Compare with previous results
    compare = action.add_parser('compare', help='Compare previous results to the current one.')
    compare.set_defaults(action=compare_previous_result_with_current)

    # Command Line package creation
    upload = action.add_parser('upload', help='Upload a new spreadsheet with the results.')

    upload.set_defaults(action=store_results_in_gdrive)
    return parser


if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()
    args.action(args)
