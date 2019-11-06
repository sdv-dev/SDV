Metadata
========

In order to have **SDV** process your dataset, you will need its **Metadata**:

.. code-block:: python

    {
        "tables": [
            {
                "fields": [
                    {"name": "user_id", "type": "id"},
                    {"name": "country", "type": "categorical"},
                    {"name": "gender", "type": "categorical"},
                    {"name": "age", "type": "numerical", "subtype": "integer"}
                ],
                "headers": True,
                "name": "users",
                "path": "users.csv",
                "primary_key": "user_id"
            },
            {
                "fields": [
                    {"name": "session_id", "type": "id"},
                    {"name": "user_id", "type": "id", "ref": {
                        "field": "user_id", "table": "users"},
                    },
                    {"name": "device", "type": "categorical"},
                    {"name": "os", "type": "categorical"}
                ],
                "headers": True,
                "name": "sessions",
                "path": "sessions.csv",
                "primary_key": "session_id"
            },
            {
                "fields": [
                    {"name": "transaction_id", "type": "id"},
                    {"name": "session_id", "type": "id", "ref": {
                        "field": "session_id", "table": "sessions"},
                    },
                    {"name": "timestamp", "format": "%Y-%m-%d", "type": "datetime"},
                    {"name": "amount", "type": "numerical", "subtype": "float"},
                    {"name": "approved", "type": "boolean"}
                ],
                "headers": True,
                "name": "transactions",
                "path": "transactions.csv",
                "primary_key": "transaction_id"
            }
        ]
    }


This can either be provided as a python `dict` object or as a JSON file, and it
mush have the following schema:

Top Level
^^^^^^^^^

At the topmost level of the **Metadata** dictionary, there is only one element:

:Tables:
    List of tables in the dataset, each one represented as a subdocument.

Table
^^^^^

A node ``table`` should be made for each table in our dataset. It contains the configuration on
how to handle this table. It has the following elements:

.. code-block:: python

    "tables": [
        {
            "fields": [...],
            "headers": true,
            "name": "users",
            "path": "users.csv",
            "primary_key": "user_id"
        },
        ...
    ]

:Fields:
    List of fields of the table.

:Headers:
    Whether or not load the headers from the csv file. This can be skipped if the
    data is being passed as ``pandas.DataFrames``.

:Name:
    Name of the table.

:Path:
    Relative path to the ``.csv`` file from the data root folder. This can be skipped if the
    data is being passed as ``pandas.DataFrames``.

:Primary_key:
    Name of the field that act as a primary key of the table.

:Use:
    Optional. If set to false, skip this table when modeling and sampling the dataset.


Field details
^^^^^^^^^^^^^

Each field within a table needs to have its name, its type and sometimes its subtype
specified.

The available types and subtypes are in this table:

+---------------+---------------+
| Type          | Subtype       |
+===============+===============+
| numerical     | integer       |
+---------------+---------------+
| numerical     | float         |
+---------------+---------------+
| datetime      | datetime      |
+---------------+---------------+
| categorical   |               |
+---------------+---------------+
| boolean       |               |
+---------------+---------------+
| id            | integer       |
+---------------+---------------+
| id            | string        |
+---------------+---------------+

.. code-block:: python

    "tables": [{
        "fields": [
            {
                "name": "country",
                "type": "categorical"
            },
            ...
        ],
        ...
    }]

:Name:
    Name of the field.

:Type:
    The type of the field.

:Subtype:
    Optional. The subtype of the field.

Datetime fields
"""""""""""""""

For  ``datetime`` types, a ``format`` key should be included containing the date format using
`strftime`_ format.

.. code-block:: python

    "tables": [{
        "fields": [
            {
                "name": "timestamp",
                "type": "datetime",
                "format": "%Y-%m-%d"
            },
            ...
        ],
        ...
    }]


Categorical fields ( Data anonymization)
""""""""""""""""""""""""""""""""""""""""

For ``categorical`` types, there is an option to anonymize data labeled as Personally Identifiable
Information, ``pii``, but keeping its statistical properties. To anonymize a field, you should use
the following keys.

.. code-block:: python

    'tables': [{
        'fields': [
            {
                'name': 'social_scurity_number',
                'type': 'categorical',
                'pii': True,
                'pii_category': 'ssn'
            },
            ...
        ],
        ...
    }]

The most common supported values of ``pii_category`` are in the following table,
but any value supported by faker can be used:

+---------------------------+
| name                      |
+---------------------------+
| first_name                |
+---------------------------+
| last_name                 |
+---------------------------+
| phone_number              |
+---------------------------+
| ssn                       |
+---------------------------+
| credit_card_number        |
+---------------------------+
| credit_card_security_code |
+---------------------------+

For a full list of available categories please check the `Faker documentation site`_

.. note:: Sometime ``Faker`` categories admit a `type`, which can be passed as an additional
          argument. If that is the case, you set a ``list`` containing both the category and
          the type instead of only the string: ``'pii_category': ['credict_card_number', 'visa']``

Primary key fields
""""""""""""""""""

If a field is specified as a ``primary_key`` of the table, then the field must be of type ``id``:

.. code-block:: python

    "tables": [{
        "fields": [
            {
                "name": "user_id",
                "type": "id"
            },
            ...
        ],
        ...
    }]

If the subtype of the primary key is integer, an optional regular expression can be passed to
generate keys that match it:

.. code-block:: python

    "tables": [{
        "fields": [
            {
                "name": "user_id",
                "type": "id",
                "subtype": "string",
                "regex": "[a-zA-Z]{10}"
            },
            ...
        ],
        ...
    }]


Foreign key fields
""""""""""""""""""

If a field is a foreign key to another table, then it has to also be of type ``id``, and
define define a relationship using the ``ref`` field:

.. code-block:: python

    "tables": [{
        "fields": [
            {
                "name": "user_id",
                "ref": {
                    "field": "user_id",
                    "table": "users"
                },
                "type": "id"
            },
            ...
        ],
        ...
    }]

:table: Parent table name.
:field: Parent table field name.


.. _strftime: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
.. _Faker documentation site: https://faker.readthedocs.io/en/master/providers.html
