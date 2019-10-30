import datetime
import random

import numpy as np
import pandas as pd

COUNTRIES = [
    'Bulgaria',
    'Canada',
    'France',
    'Germany',
    'Spain',
    'United States',
]

GENDERS = ['M', 'F', None]

DEVICE_TYPES = ['mobile', 'tablet']

OPERATIVE_SYSTEMS = ['ios', 'android', 'windows']


def _get_random_choices(choices, amount):
    return np.array([random.choice(choices) for _ in range(amount)])


def _get_datetime(amount):
    return np.array([
        datetime.date(random.randint(2016, 2018), random.randint(1, 12), random.randint(1, 28))
        for _ in range(amount)
    ])


def get_demo(amount=10):
    """Generate three tables related between them.

    Generate three tables, ``users``, ``sessions`` and ``transactions`` with random generated rows.

    Args:
        amount (int):
            The amount of rows that we would like to generate.
    Returns:
        tuple of dicts:
            Returns a tuple that contains two dictionaries, metadata and tables.
    """

    users = pd.DataFrame({
        'user_id': np.array(range(amount)),
        'country': _get_random_choices(COUNTRIES, amount),
        'gender': _get_random_choices(GENDERS, amount),
        'age': np.array([random.randint(18, 50) for _ in range(amount)]),
    })

    sessions = pd.DataFrame({
        'session_id': np.array(range(amount)),
        'user_id': np.array([random.randint(0, amount - 1) for _ in range(amount)]),
        'device_type': _get_random_choices(DEVICE_TYPES, amount),
        'operative_system': _get_random_choices(OPERATIVE_SYSTEMS, amount),
    })

    transactions = pd.DataFrame({
        'transaction_id': np.array(range(amount)),
        'session_id': np.array([random.randint(0, amount - 1) for _ in range(amount)]),
        'datetime': pd.to_datetime(_get_datetime(amount)),
        'amount': np.array([round(random.random() * 1000, 2) for _ in range(amount)]),
        'approved': _get_random_choices([True, False], amount),
    })

    metadata = {
        "path": "",
        "tables": [
            {
                "headers": True,
                "name": "users",
                "path": "users.csv",
                "primary_key": "user_id",
                "use": True,
                "fields": [
                    {
                        "name": "user_id",
                        "type": "id",
                        "subtype": "number"
                    },
                    {
                        "name": "country",
                        "type": "categorical",
                        "subtype": "categorical"
                    },
                    {
                        "name": "gender",
                        "type": "categorical",
                        "subtype": "categorical"
                    },
                    {
                        "name": "age",
                        "type": "number",
                        "subtype": "integer"
                    }
                ]
            },
            {
                "headers": True,
                "name": "sessions",
                "path": "sessions.csv",
                "primary_key": "session_id",
                "use": True,
                "fields": [
                    {
                        "name": "session_id",
                        "type": "id",
                        "subtype": "number"
                    },
                    {
                        "name": "user_id",
                        "ref": {
                            "field": "user_id",
                            "table": "users"
                        },
                        "type": "id",
                        "subtype": "number"
                    },
                    {
                        "name": "device_type",
                        "type": "categorical",
                        "subtype": "categorical"
                    },
                    {
                        "name": "operative_system",
                        "type": "categorical",
                        "subtype": "categorical"
                    }
                ]
            },
            {
                "headers": True,
                "name": "transactions",
                "path": "transactions.csv",
                "primary_key": "transaction_id",
                "use": True,
                "fields": [
                    {
                        "name": "transaction_id",
                        "type": "id",
                        "subtype": "number"
                    },
                    {
                        "name": "session_id",
                        "ref": {
                            "field": "session_id",
                            "table": "sessions"
                        },
                        "type": "id",
                        "subtype": "number"
                    },
                    {
                        "name": "datetime",
                        "type": "datetime",
                        "format": "%Y-%m-%d"
                    },
                    {
                        "name": "amount",
                        "type": "number",
                        "subtype": "float"
                    },
                    {
                        "name": "approved",
                        "type": "categorical",
                        "subtype": "bool"
                    }
                ]
            }
        ]
    }

    tables = {
        'users': users,
        'sessions': sessions,
        'transactions': transactions
    }

    return metadata, tables
