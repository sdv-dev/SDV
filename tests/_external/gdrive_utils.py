"""Google Drive utils."""

import io
import json
import os
import random
import time
from datetime import date
from functools import lru_cache, wraps

import git
import pandas as pd
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ['https://www.googleapis.com/auth/drive']
PYDRIVE_CREDENTIALS = 'PYDRIVE_CREDENTIALS'

MAX_RETRIES = 5
MAXIMUM_BACKOFF = 64


def exponential_backoff(func):
    """Exponential backoff decorator to prevent google drive timeout."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        tries = 0
        while tries < MAX_RETRIES:
            try:
                return func(*args, **kwargs)

            except Exception as exception:
                tries += 1
                if tries == MAX_RETRIES:
                    raise exception

            random_milliseconds = random.randint(0, 1000) / 1000
            backoff_time = min((2**tries) + random_milliseconds, MAXIMUM_BACKOFF)
            time.sleep(backoff_time)

    return wrapper


def _generate_filename():
    """Generate a filename with today's date and the commit id."""
    repo = git.Repo(search_parent_directories=True)
    commit_id = repo.head.object.hexsha
    today = str(date.today())
    return f'{today}-{commit_id}.xlsx'


@lru_cache()
def _get_drive_service():
    tmp_credentials = os.getenv('PYDRIVE_CREDENTIALS')
    credentials_json = json.loads(tmp_credentials)
    credentials = Credentials.from_authorized_user_info(credentials_json, SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    return service


@exponential_backoff
def get_latest_file(folder_id):
    """Get the latest file from the given Google Drive folder."""
    service = _get_drive_service()

    query = f"'{folder_id}' in parents and trashed = false"
    results = (
        service.files()
        .list(q=query, orderBy='modifiedTime desc', pageSize=1, fields='files(id, name)')
        .execute()
    )

    files = results.get('files', [])
    if files:
        return files[0]


@exponential_backoff
def read_excel(file_id):
    """Read a file as an XLSX from Google Drive.

    Args:
        file_id (str):
            The ID of the file to load.

    Returns:
        pd.DataFrame or dict[pd.DataFrame]:
            A DataFrame containing the body of file if single sheet else dict of DataFrames one for
            each sheet

    """
    service = _get_drive_service()

    # Get file metadata to check mimeType
    file_metadata = service.files().get(fileId=file_id, fields='mimeType').execute()
    mime_type = file_metadata.get('mimeType')

    if mime_type == 'application/vnd.google-apps.spreadsheet':
        # If it's a Google Sheet, export it to XLSX
        request = service.files().export_media(
            fileId=file_id,
            mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
    else:
        # If it's already an XLSX or other binary format, download it directly
        request = service.files().get_media(fileId=file_id)

    # Download file content
    file_io = io.BytesIO()
    downloader = MediaIoBaseDownload(file_io, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    file_io.seek(0)  # Reset stream position

    # Load the file content into pandas
    return pd.read_excel(file_io, sheet_name=None)


def _set_column_width(writer, results, sheet_name):
    for column in results:
        column_width = max(results[column].astype(str).map(len).max(), len(column))
        col_idx = results.columns.get_loc(column)
        writer.sheets[sheet_name].set_column(col_idx, col_idx, column_width + 2)


def _set_color_fields(worksheet, data, marked_data, writer, color_code):
    for _, row in marked_data.iterrows():
        dtype = row['dtype']
        sdtype = row['sdtype']
        method = row['method']

        format_code = writer.book.add_format({'bg_color': color_code})

        for data_row in range(len(data)):
            if data.loc[data_row, 'dtype'] == dtype and data.loc[data_row, 'sdtype'] == sdtype:
                method_col = data.columns.get_loc(method)
                worksheet.write(data_row + 1, method_col, data.loc[data_row, method], format_code)


@exponential_backoff
def save_to_gdrive(output_folder, results, output_filename=None, mark_results=None):
    """Save a ``DataFrame`` to google drive folder as ``xlsx`` (spreadsheet).

    Given the output folder id (google drive folder id), store the given ``results`` as
    ``spreadsheet``. If not ``output_filename`` is given, the spreadsheet is saved with the
    current date and commit as name.

    Args:
        output_folder (str):
            String representing a google drive folder id.
        results (pd.DataFrame or dict[pd.DataFrame]):
            Dataframe to be stored as ``xlsx``, or dictionary mapping sheet names to dataframes for
            storage in one ``xlsx`` file.
        output_filename (str, optional):
            String representing the filename to be used for the results spreadsheet. If None,
            uses to the current date and commit as the name. Defaults to None.
        mark_results (dict, optional):
            A dictionary that maps hex color codes to dataframes. Each dataframe associates
            specific `data types`, `sdtypes` and methods to be highlighted with the hex color.

    Returns:
        str:
            Google drive file id of uploaded file.
    """
    if not output_filename:
        output_filename = _generate_filename()

    output = io.BytesIO()
    with pd.ExcelWriter(
        output, engine='xlsxwriter', engine_kwargs={'options': {'nan_inf_to_errors': True}}
    ) as writer:  # pylint: disable=E0110
        for sheet_name, data in results.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
            _set_column_width(writer, data, sheet_name)
            if mark_results:
                for color_code, marked_results in mark_results.items():
                    marked_data = marked_results[marked_results['python_version'] == sheet_name]
                    if not marked_data.empty:
                        worksheet = writer.sheets[sheet_name]
                        _set_color_fields(worksheet, data, marked_data, writer, color_code)

    output.seek(0)

    file_metadata = {
        'name': output_filename,
        'parents': [output_folder],
        'mimeType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    }

    service = _get_drive_service()
    media = MediaIoBaseUpload(output, mimetype=file_metadata['mimeType'], resumable=True)

    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')

    return file_id
