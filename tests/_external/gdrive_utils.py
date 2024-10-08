"""Google Drive utils."""

import io
import json
import os
import pathlib
import tempfile
from datetime import date

import git
import pandas as pd
import yaml
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

PYDRIVE_CREDENTIALS = 'PYDRIVE_CREDENTIALS'


def _generate_filename():
    """Generate a filename with today's date and the commit id."""
    repo = git.Repo(search_parent_directories=True)
    commit_id = repo.head.object.hexsha
    today = str(date.today())
    return f'{today}-{commit_id}.xlsx'


def _get_drive_client():
    tmp_credentials = os.getenv(PYDRIVE_CREDENTIALS)
    if not tmp_credentials:
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
    else:
        with tempfile.TemporaryDirectory() as tempdir:
            credentials_file_path = pathlib.Path(tempdir) / 'credentials.json'
            credentials_file_path.write_text(tmp_credentials)

            credentials = json.loads(tmp_credentials)

            settings = {
                'client_config_backend': 'settings',
                'client_config': {
                    'client_id': credentials['client_id'],
                    'client_secret': credentials['client_secret'],
                },
                'save_credentials': True,
                'save_credentials_backend': 'file',
                'save_credentials_file': str(credentials_file_path),
                'get_refresh_token': True,
            }
            settings_file = pathlib.Path(tempdir) / 'settings.yaml'
            settings_file.write_text(yaml.safe_dump(settings))

            gauth = GoogleAuth(str(settings_file))
            gauth.LocalWebserverAuth()

    return GoogleDrive(gauth)


def get_latest_file(folder_id):
    """Get the latest file from the given Google Drive folder.

    Args:
        folder (str):
            The string Google Drive folder ID.
    """
    drive = _get_drive_client()
    drive_query = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=False",
        'orderBy': 'modifiedDate desc',
        'maxResults': 1,
    })
    file_list = drive_query.GetList()
    if len(file_list) > 0:
        return file_list[0]


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
    client = _get_drive_client()
    drive_file = client.CreateFile({'id': file_id})
    xlsx_mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    drive_file.FetchContent(mimetype=xlsx_mime)
    return pd.read_excel(drive_file.content, sheet_name=None)


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
                worksheet.write(
                    data_row + 1, method_col, bool(data.loc[data_row, method]), format_code
                )


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
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:  # pylint: disable=E0110
        for sheet_name, data in results.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
            _set_column_width(writer, data, sheet_name)
            if mark_results:
                for color_code, marked_results in mark_results.items():
                    marked_data = marked_results[marked_results['python_version'] == sheet_name]
                    if not marked_data.empty:
                        worksheet = writer.sheets[sheet_name]
                        _set_color_fields(worksheet, data, marked_data, writer, color_code)

    file_config = {'title': output_filename, 'parents': [{'id': output_folder}]}
    drive = _get_drive_client()
    drive_file = drive.CreateFile(file_config)
    drive_file.content = output
    drive_file.Upload({'convert': True})
    return drive_file['id']
