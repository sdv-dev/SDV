from io import StringIO

import pandas as pd

from sdv.data_navigator import DataNavigator, Table


class DummyDataNavigator(DataNavigator):
    """Dummy DataNavigator subclass to test."""
    def _anonymize_data(self):
        pass


def get_table_customers_meta():
    """Get meta.json DEMO_CUSTOMERS table."""
    return {
        "fields": {
            'CUSTOMER_ID': {'name': 'CUSTOMER_ID',
                            'subtype': 'integer',
                            'type': 'number',
                            'uniques': 0,
                            'regex': '^[0-9]{10}$'},
            'PHONE_NUMBER1': {'name': 'PHONE_NUMBER1',
                              'subtype': 'integer',
                              'type': 'number',
                              'pii': True,
                              'pii_category': 'phone_number',
                              'uniques': 0}
        },
        "headers": True,
        "name": "DEMO_CUSTOMERS",
        "path": "customers.csv",
        "primary_key": "CUSTOMER_ID",
        "use": True
    }


def get_table_customers_data():
    """Get pandas DEMO_CUSTOMERS table data."""
    return pd.read_csv(StringIO("""
    CUSTOMER_ID,PHONE_NUMBER1
    50,6175553295
    """))


def get_table_customers_data_anonymized():
    """Get pandas DEMO_CUSTOMERS table data anonymized."""
    return pd.read_csv(StringIO("""
    CUSTOMER_ID,PHONE_NUMBER1
    50,1234567890
    """))


def get_table_orders_meta():
    """Get meta.json DEMO_ORDERS table."""
    return {
        "fields": {
            'ORDER_ID': {'name': 'ORDER_ID',
                         'subtype': 'integer',
                         'type': 'number',
                         'uniques': 0,
                         'regex': '^[0-9]{2}$'},
            'CUSTOMER_ID': {'name': 'CUSTOMER_ID',
                            'ref': {'field': 'CUSTOMER_ID', 'table': 'DEMO_CUSTOMERS'},
                            'subtype': 'integer',
                            'type': 'number',
                            'uniques': 0}
        },
        "headers": True,
        "name": "DEMO_ORDERS",
        "path": "orders.csv",
        "primary_key": "ORDER_ID",
        "use": True
    }


def get_table_orders_data():
    """Get pandas DEMO_ORDERS table data."""
    return pd.read_csv(StringIO("""
    ORDER_ID,CUSTOMER_ID
    1,50
    """))


def get_table_order_items_meta():
    """Get meta.json DEMO_ORDER_ITEMS table."""
    return {
        "fields": {
            'ORDER_ITEM_ID': {'name': 'ORDER_ITEM_ID',
                              'subtype': 'integer',
                              'type': 'number',
                              'uniques': 0,
                              'regex': '^[0-9]{3}$'},
            'ORDER_ID': {'name': 'ORDER_ID',
                         'ref': {'field': 'ORDER_ID', 'table': 'DEMO_ORDERS'},
                         'subtype': 'integer',
                         'type': 'number',
                         'uniques': 0}
        },
        "headers": True,
        "name": "DEMO_ORDER_ITEMS",
        "path": "order_items.csv",
        "primary_key": "ORDER_ITEM_ID",
        "use": True
    }


def get_table_order_items_data():
    """Get pandas DEMO_ORDER_ITEMS table data."""
    return pd.read_csv(StringIO("""
    ORDER_ITEM_ID,ORDER_ID
    102,1
    """))


def get_relations_tuple():
    """Get relations tuple."""
    return (
        {
            'DEMO_CUSTOMERS': {'DEMO_ORDERS'},
            'DEMO_ORDERS': {'DEMO_ORDER_ITEMS'}
        },
        {
            'DEMO_ORDERS': {'DEMO_CUSTOMERS'},
            'DEMO_ORDER_ITEMS': {'DEMO_ORDERS'}
        },
        {
            ('DEMO_ORDERS', 'DEMO_CUSTOMERS'): ('CUSTOMER_ID', 'CUSTOMER_ID'),
            ('DEMO_ORDER_ITEMS', 'DEMO_ORDERS'): ('ORDER_ID', 'ORDER_ID')
        }
    )


def get_ht_table_dict():
    """Get hyper transformer table_dict."""
    return {
        'DEMO_CUSTOMERS': (get_table_customers_data_anonymized(), 'meta_customers'),
        'DEMO_ORDERS': (get_table_orders_data(), 'meta_orders'),
        'DEMO_ORDER_ITEMS': (get_table_order_items_data(), 'meta_order_items')
    }


def get_ht_fit_transform():
    """Get hyper transformer fit_transform."""
    return {
        'DEMO_CUSTOMERS': get_table_customers_data(),
        'DEMO_ORDERS': get_table_orders_data(),
        'DEMO_ORDER_ITEMS': get_table_order_items_data()
    }


def build_meta(path=''):
    """Build simple meta.json structure.

    meta = {
        'path': str,
        'tables': list
    }

    Args:
        path (str): used as base path to search the dataset files.

    Return: meta.json dict
    """
    return {
        'path': path,
        'tables': [
            get_table_customers_meta(),
            get_table_orders_meta(),
            get_table_order_items_meta()
        ]
    }


def build_tables():
    """Build dict of tables.

    tables = {
        'TABLE_NAME': Tables(table_meta, table_data),
        ...
    }

    """
    return {
        'DEMO_CUSTOMERS': Table(
            get_table_customers_data(),
            get_table_customers_meta()
        ),
        'DEMO_ORDERS': Table(
            get_table_orders_data(),
            get_table_orders_meta()
        ),
        'DEMO_ORDER_ITEMS': Table(
            get_table_order_items_data(),
            get_table_order_items_meta()
        )
    }
