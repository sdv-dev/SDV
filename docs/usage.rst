=====
Usage
=====

Preparation of the data
-----------------------

SDV is used to model and sample data from relational datasets along with their relational
behaviors. To do so it needs the data in a format in which it can process their relationships.

To prepare a dataset to work with SDV you need to do the following:

1. Dump your tables in ``.csv`` files, named after them. There have to be **one single file** for
   table.
2. Create a ``meta.json`` file with the specification of your dataset.
3. Load it with ``SDV``


Metadata file specification
---------------------------

The ``meta.json`` is a file containing a single JSON with all the information SDV requires to
work, its base schema is as follows:

.. code-block:: python

    {
        "path": "",
        "tables": [
            {
                "fields": [
                    {
                        "name": "CUSTOMER_ID",
                        "subtype": "integer",
                        "type": "number",
                        "uniques": 0,
                        "regex": "^[0-9]{10}$"
                    },
                    ...
                ],
                "headers": true,
                "name": "DEMO_CUSTOMERS",
                "path": "customers.csv",
                "primary_key": "CUSTOMER_ID",
                "use": true
            },
            ...
        ]
    }


:Path:
    Relative path from this file to the root folder of the datasets. Leave empty if the
    datasets are on the same folder than the ``meta.json`` file.

:Tables:
    List of tables in the dataset. Each table should have

Table details
^^^^^^^^^^^^^

A node ``table`` should be made for each table in our dataset. It contains the configuration on
how to handle this table. It has the following elements:

.. code-block:: python

    "tables": [
        {
            "fields": [...],
            "headers": true,
            "name": "DEMO_CUSTOMERS",
            "path": "customers.csv",
            "primary_key": "CUSTOMER_ID",
            "use": true
        },
        ...
    ]

:Fields:
    List of fields of the table.

:Headers:
    Whether or not load the headers from the csv file.

:Name:
    Name of the table.

:Path:
    Relative path to the ``.csv`` file from the data root folder.

:Primary_key:
    Name of the field that act as a primary key of the table.

:Use:
    Wheter or not use this table when sampling.


Field details
^^^^^^^^^^^^^

.. code-block:: python

    "tables": [{
        "fields": [
            {
                "name": "CREDIT_LIMIT",
                "subtype": "integer",
                "type": "number",
                "uniques": 0
            },
            ...
        ],
        ...
    }]

:Name:
    Name of the field.

:Uniques:
    Number of unique values in this field.

:Type:
    The type of the field. See table below.

:Subtype:
    The subtype of the field. See table below


+---------------+---------------+
| Type          | Subtype       |
+===============+===============+
| number        | integer       |
+---------------+---------------+
| number        | float         |
+---------------+---------------+
| datetime      | datetime      |
+---------------+---------------+
| categorical   | categorical   |
+---------------+---------------+
| categorical   | boolean       |
+---------------+---------------+

Datetime fields
"""""""""""""""

For  ``datetime`` types, a ``format`` key should be included containing the date format using
`strftime`_ format.

.. code-block:: python

    "tables": [{
        "fields": [
            {
                "name": "timestamp_first_active",
                "type": "datetime",
                "format": "%Y%m%d%H%M%S",
                "uniques": 213451
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
                'pii': True, # expected a bool
                'pii_category': 'ssn' # expected a string
            },
            ...
        ],
        ...
    }]

The most common supported values of ``pii_category`` are:

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

But any value supported by faker can be used. A full list can be found here: `Faker`_


Primary key fields
""""""""""""""""""

If a field is specified as a ``primary_key`` of the table, then a key ``regex`` matching its format
should be included.

.. code-block:: python

    "tables": [{
        "fields": [
            {
                "name": "CUSTOMER_ID",
                "subtype": "integer",
                "type": "number",
                "uniques": 0,
                "regex": "^[0-9]{10}$"
            },
            ...
        ],
        ...
    }]


Foreign key fields
""""""""""""""""""

If a field is a foreign key to another table, then it has to be specified using the ``ref``.

.. code-block:: python

    "tables": [{
        "fields": [
            {
                "name": "CUSTOMER_ID",
                "ref": {
                    "field": "CUSTOMER_ID",
                    "table": "DEMO_CUSTOMERS"
                },
                "subtype": "integer",
                "type": "number",
                "uniques": 0
            },
            ...
        ],
        ...
    }]

:table: Origin table name.
:field: Origin table field name.


Examples
^^^^^^^^
A full working example can be found on the `tests`_ folder.


Sampling new data
-----------------
To use SDV in a project

.. code-block:: python

    >>> from sdv import SDV

    >>> vault = SDV('meta.json')
    >>> vault.fit()
    >>> vault.sample()


.. _RDT: https://github.com/HDI-Project/RDT
.. _strftime: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
.. _tests: https://github.com/HDI-Project/SDV/blob/master/tests/data/meta.json
.. _Faker: https://faker.readthedocs.io/en/master/providers.html
