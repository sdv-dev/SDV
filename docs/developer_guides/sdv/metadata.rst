.. _metadata_schema:

Metadata Schema
===============

This section explains the format of the metadata JSON file.

Top Level
---------

At the topmost level of the **Metadata** dictionary, there is only one element:

:Tables:
    Mapping of tables in the dataset, each one represented as a sub-document, with
    the table name as the corresponding key.

Table
-----

A node ``table`` should be made for each table in your dataset. It contains the configuration on
how to handle this table. It has the following elements:

.. code-block:: python

    "tables": {
        "users": {
            "fields": {...},
            "path": "users.csv",
            "primary_key": "user_id"
        },
        ...
    }

:Fields:
    Mapping of fields in the table.

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
-------------

Each field within a table needs to have its type specified,
Additionally, some field types need additional details, such as the subtype or
other properties.

The available types and subtypes are in this table:

+---------------+---------------+-----------------------+
| Type          | Subtype       | Additional Properties |
+===============+===============+=======================+
| numerical     | integer       | integer               |
+---------------+---------------+-----------------------+
| numerical     | float         | float                 |
+---------------+---------------+-----------------------+
| datetime      |               | format                |
+---------------+---------------+-----------------------+
| categorical   |               | pii, pii_category     |
+---------------+---------------+-----------------------+
| boolean       |               |                       |
+---------------+---------------+-----------------------+
| id            | integer       | ref                   |
+---------------+---------------+-----------------------+
| id            | string        | ref, regex            |
+---------------+---------------+-----------------------+

.. code-block:: python

    "tables": {
        "users": {
            "fields": {
                "country": {
                    "type": "categorical"
                },
                ...
            },
            ...
        },
        ...
    }

:Type:
    The type of the field.

Datetime fields
***************

For  ``datetime`` types, a ``format`` key should be included containing the date format using
`strftime`_ format.

.. code-block:: python

    "tables": {
        "transactions": {
            "fields": {
                "timestamp": {
                    "type": "datetime",
                    "format": "%Y-%m-%d"
                },
                ...
            },
            ...
        },
        ...
    }


Categorical fields (Data anonymization)
****************************************

For ``categorical`` types, there is an option to anonymize data labeled as Personally Identifiable
Information, ``pii``, but keeping its statistical properties. To anonymize a field, you should use
the following keys.

.. code-block:: python

    "tables": {
        "users": {
            "fields": {
                "social_security_number": {
                    "type": "categorical",
                    "pii": True,
                    "pii_category": "ssn"
                },
                ...
            },
            ...
        },
        ...
    }

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
******************

If a field is specified as a ``primary_key`` of the table, then the field must be of type ``id``:

.. code-block:: python

    "tables": {
        "users": {
            "fields": {
                "user_id": {
                    "name": "user_id"
                },
                ...
            },
            ...
        },
        ...
    }

If the subtype of the primary key is integer, an optional regular expression can be passed to
generate keys that match it:

.. code-block:: python

    "tables": {
        "users": {
            "fields": {
                "user_id": {
                    "name": "user_id",
                    "type": "id",
                    "subtype": "string",
                    "regex": "[a-zA-Z]{10}"
                },
                ...
            },
            ...
        },
        ...
    }


Foreign key fields
******************

If a field is a foreign key to another table, then it has to also be of type ``id``, and
define define a relationship using the ``ref`` field:

.. code-block:: python

    "tables": {
        "sessions": {
            "fields": {
                "user_id": {
                    "type": "id"
                    "ref": {
                        "field": "user_id",
                        "table": "users"
                    },
                },
                ...
            },
            ...
        },
        ...
    }]

:table: Parent table name.
:field: Parent table field name.


.. _strftime: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
.. _Faker documentation site: https://faker.readthedocs.io/en/master/providers.html
