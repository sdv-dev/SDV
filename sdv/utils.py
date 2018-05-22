def format_table_meta(table_meta):
    """ reformats table meta to turn fields into dictionary """
    new_fields = {}
    for field in table_meta['fields']:
        field_name = field['name']
        new_fields[field_name] = field
    table_meta['fields'] = new_fields
    return table_meta
