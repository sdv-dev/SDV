Metadata
========

In order to use **SDV** you will need a ``Metadata`` object alongside your data.

This document explains how to create it, as well as how to represent it as a JSON file.

Generate Metadata from your data
--------------------------------

In this step by step guide you will show you how to create a ``Metadata`` object
by letting **SDV** analyze your tables and figure out the data types of your columns.

Load your data
**************

The first step is to load your data as ``pandas.DataFrame`` objects.
In this example, you will be loading the demo data using the ``load_demo`` function.

   .. code-block:: python

      from sdv.demo import load_demo

      tables = load_demo()

Create a Metadata instance
**************************

The next step is to create an empty instance of the ``Metadata`` class without
passing it any arguments.

   .. code-block:: python

      from sdv import Metadata

      metadata = Metadata()

Add the first table
*******************

Once you have your ``Metadata`` instance ready you can start adding tables.

In this example, you will add the table ``users``, which is the parent table of your
dataset, indicating which is its Primary Key field, ``user_id``.

Indicating the Primary Key is optional and can be skipped if your table has none, but
then you will not be able to specify any child table.

The ``Metadata`` instance will analyze all the columns in the passed table and identify
they different data types and subtypes, and indicate that the ``user_id`` column is
the table primary key.

   .. code-block:: python

      metadata.add_table('users', data=tables['users'], primary_key='user_id')

Add a table specifying its parent
*********************************

In this second example, you will add the table ``sessions``, which is related to the
``users`` table in a parent-child relationship, where each user can have multiple
sessions, and each session belongs to one and only one user.

In order to specify this, while creating the ``sessions`` table you have to indicate the
name of the parent table, ``users``, and the field from the ``sessions`` table that
acts as the foreign key, ``user_id``.

With this, a part from analyzing all the columsn and indicating the primary key like in
the previous step, the ``Metadata`` instance will specify a relationship between the
two tables by adding a property to the ``user_id`` field that indicates that it is related
to the ``user_id`` field in the ``users`` table.

   .. code-block:: python

      metadata.add_table('sessions', data=tables['sessions'], primary_key='session_id',
                         parent='users', foreign_key='user_id')

The ``foreign_key`` field is optional, and can be skipped when the name of the child foreign
key field is exactly the same as the parent primary key field.

Add a table specifying field properties
***************************************

There are situations where the ``Metadata`` analysis is not able to figure out
some data types or subtypes, or to deduce some properties of the field such as the
datetime format.

In these situations, you can pass a dictionary with the exact metadata of those fields,
which will overwrite the deductions from the analysis process.

In this next example, you will be adding a ``transactions`` table, which is related to
the previous ``sessions`` table, and contains a ``datetime`` field which needs to have
the datetime format specified.

   .. code-block:: python

       transactions_fields = {
           'timestamp': {
               'type': 'datetime',
               'format': '%Y-%m-%d'
           }
       }
       metadata.add_table('transactions', data=tables['transactions'],
                          fields_metadata=transactions_fields,
                          primary_key='transaction_id', parent='sessions')

.. note:: When analyzing an integer column that also has null values in it, the type will
          be correct, ``numerical``, but the subtype will be mistakenly set as ``float``.
          This can be fixed by passing the ``integer`` subtype.


Store your Metadata in a JSON file
**********************************

Once you have finished configuring your ``Metadata`` instance, you can use it with ``SDV``.

However, in some occasions you will want to store it as a JSON file, so you do not need to
configure it again the next time that you want to work on this dataset.

This can be esily done using the ``to_json`` method of your ``Metadata`` instance, passing
it the path and name of the file where you want your JSON metadata stored.

   .. code-block:: python

      metadata.to_json('paht/to/metadata.json')

This will create a file with this contents:

   .. code-block:: json

      {
          "tables": {
              "users": {
                  "primary_key": "user_id",
                  "fields": {
                      "user_id": {
                          "type": "id",
                          "subtype": "integer"
                      },
                      "country": {
                          "type": "categorical"
                      },
                      "gender": {
                          "type": "categorical"
                      },
                      "age": {
                          "type": "numerical",
                          "subtype": "integer"
                      }
                  }
              },
              "sessions": {
                  "primary_key": "session_id",
                  "fields": {
                      "session_id": {
                          "type": "id",
                          "subtype": "integer"
                      },
                      "user_id": {
                          "ref": {
                              "field": "user_id",
                              "table": "users"
                          },
                          "type": "id",
                          "subtype": "integer"
                      },
                      "device": {
                          "type": "categorical"
                      },
                      "os": {
                          "type": "categorical"
                      }
                  }
              },
              "transactions": {
                  "primary_key": "transaction_id",
                  "fields": {
                      "transaction_id": {
                          "type": "id",
                          "subtype": "integer"
                      },
                      "session_id": {
                          "ref": {
                              "field": "session_id",
                              "table": "sessions"
                          },
                          "type": "id",
                          "subtype": "integer"
                      },
                      "timestamp": {
                          "type": "datetime",
                          "format": "%Y-%m-%d"
                      },
                      "amount": {
                          "type": "numerical",
                          "subtype": "float"
                      },
                      "approved": {
                          "type": "boolean"
                      }
                  }
              }
          }
      }

Later on, you can recover your ``Metadata`` by passing the path to your ``metadata.json`` file
as an argument when creating a new ``Metadata`` instance:

   .. code-block:: python

      metadata = Metadata('metadata.json')


Metadata Schema
---------------

This section explains the format of the metadata JSON file.

Top Level
---------

At the topmost level of the **Metadata** dictionary, there is only one element:

:Tables:
    Mapping of tables in the dataset, each one represented as a subdocument, with
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
