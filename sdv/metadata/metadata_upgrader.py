"""Methods to help with upgrading metadata."""


def _upgrade_columns_and_keys(old_metadata):
    new_metadata = {}
    columns = {}
    fields = old_metadata.get('fields')
    alternate_keys = []
    primary_key = old_metadata.get('primary_key')
    for field, field_meta in fields.items():
        column_meta = {}
        old_type = field_meta['type']
        subtype = field_meta.get('subtype')
        column_meta['sdtype'] = old_type

        if old_type == 'numerical':
            if subtype == 'float':
                column_meta['computer_representation'] = 'Float'
            elif subtype == 'integer':
                column_meta['computer_representation'] = 'Int64'

        elif old_type == 'datetime':
            datetime_format = field_meta.get('format')
            if datetime_format:
                column_meta['datetime_format'] = datetime_format

        elif old_type == 'id':
            column_meta['sdtype'] = 'id'
            if subtype == 'integer':
                regex_format = r'\d{30}'
            else:
                regex_format = field_meta.get('regex', '[A-Za-z]{5}')

            if not field_meta.get('pii'):
                column_meta['regex_format'] = regex_format

            if field != primary_key and field_meta.get('ref') is None:
                alternate_keys.append(field)

        if field_meta.get('pii'):
            column_meta['pii'] = True
            pii_category = field_meta.get('pii_category')
            if isinstance(pii_category, list):
                column_meta['sdtype'] = pii_category[0]
            elif pii_category:
                column_meta['sdtype'] = pii_category

        columns[field] = column_meta

    new_metadata['columns'] = columns
    new_metadata['primary_key'] = primary_key
    if alternate_keys:
        new_metadata['alternate_keys'] = alternate_keys

    return new_metadata


def convert_metadata(old_metadata):
    """Convert old metadata to the new metadata format.

    Args:
        old_metadata (dict):
            A dict version of the old metadata.

    Returns:
        An equivalent dict in the new metadata format.
    """
    new_metadata = _upgrade_columns_and_keys(old_metadata)
    return new_metadata
