<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“SDV” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![PyPi][pypi-img]][pypi-url]
[![Travis][travis-img]][travis-url]
[![CodeCov][codecov-img]][codecov-url]
[![Downloads][downloads-img]][downloads-url]

[pypi-img]: https://img.shields.io/pypi/v/sdv.svg
[pypi-url]: https://pypi.python.org/pypi/sdv
[travis-img]: https://travis-ci.org/HDI-Project/SDV.svg?branch=master
[travis-url]: https://travis-ci.org/HDI-Project/SDV
[codecov-img]: https://codecov.io/gh/HDI-Project/SDV/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/HDI-Project/SDV
[downloads-img]: https://pepy.tech/badge/sdv
[downloads-url]: https://pepy.tech/project/sdv

<h1>SDV - Synthetic Data Vault</h1>

- Free Software: MIT License
- Documentation: https://HDI-Project.github.io/SDV
- Homepage: https://github.com/HDI-Project/SDV

# Overview

**SDV** is an automated generative modeling and sampling tool that allows the users to generate
synthetic data after creating generative models for multi-table, relational datasets.

# Install

## Requirements

**SDV** has been developed and tested on [Python 3.5, 3.6 and 3.7](https://www.python.org/downloads)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **SDV** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **SDV**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) sdv-venv
```

Afterwards, you have to execute this command to have the virtualenv activated:

```bash
source sdv-venv/bin/activate
```

Remember about executing it every time you start a new console to work on **SDV**!

## Install with pip

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **SDV**:

```bash
pip install sdv
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

## Install from source

With your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:HDI-Project/SDV.git
cd SDV
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://HDI-Project.github.io/SDV/contributing.html#get-started)
for more details about this process.

# Data Requirements

**SDV** allows the users to sample single tables, multi-tables or relantional tables. In the case
of the relational tables **only one parent is supported at the moment**.

There are two ways of passing the data to **SDV**:

## Datasets in memory

One way to fulfill **SDV**'s data requierements is to pass your data as a python `dict` where
the `key` is the `name` of the table and the `value` is a `pandas.DataFrame` instance of this
table.

## Datasets in CSV format

The other way to fulfill **SDV**'s data requierements is to have your datatables in `.csv` format
separated by `,`. The path tho this `csv` files must be specified in the `metadata` file or `dict`,
more about metadata in the following [section](#metadata).

# Metadata

The `metadata` is requiered by **SDV** in order to obtain information for the datasets and the
relations between them if there is any. This information can be given to **SDV** as a python `dict`
or a `json` file consisting of multiple parts. At the highest level of the object, there is
information about the path to the dataset and a list of table objects each representing a data
table in the dataset. Each table object contains information about its row and other important
information. The structure of the meta.json object is described below.

## Meta Object

- **tables** - A list of table objects.

## Table Object

- **path** - A string containing the path to the table's `csv` file.
- **name** - A string representing the name of the table for reference.
- **primary_key** - A string containing the name of the primary key column.
- **fields** - A list of field objects in the table.

## Field Object

- **name** - A string representing the name of the field.
- **type** - A string representing the type of the field. Those types are as follow:
    - boolean
    - categorical
    - datetime
    - numerical
- **subtype** - A string representing the subtype. Only the following subtypes are present:
    - Numerical:
        - Integer
        - Float
- **ref** - An object that represennts a foreign key, a reference to another table's primary key.

## Ref Object

- **table** - A string representing the name of the table that's primary key is being referenced.
- **field** - A string representing the name of the field that is the primary key.

**Bear in mind** that primary keys can only be of `type` `id` and those can only be `int` values.
More detailed information about how to generate a proper `metadata` can be found at the
[project documentation site](https://HDI-Project.github.io/SDV/).

# Quickstart

In this short series of tutorials we will guide you through a series of steps that will help you
getting started using **SDV** to sample columns, tables and datasets.

## 1. Load some demo datasets and metadata

As we explained before, we will need some data and the metadata corresponding to this data in order
to work with **SDV**. In this example, we will use the function `get_demo` from `sdv.demo` module.
This will return us `metadata` and `tables`.

```python
from sdv import load_demo

metadata, tables = load_demo()
```

The returned objects contain the following information:

- `tables`: python `dict` that contains three tables (`users`, `sessions` and `transactions`).
    ```
    {
        'users':
                user_id country gender  age
              0        0     USA      M   34
              1        1      UK      F   23
              2        2      ES   None   44
              3        3      UK      M   22
              4        4     USA      F   54
              5        5      DE      M   57
              6        6      BG      F   45
              7        7      ES   None   41
              8        8      FR      F   23
              9        9      UK   None   30,
      'sessions':
              session_id  user_id  device       os
              0           0        0  mobile  android
              1           1        1  tablet      ios
              2           2        1  tablet  android
              3           3        2  mobile  android
              4           4        4  mobile      ios
              5           5        5  mobile  android
              6           6        6  mobile      ios
              7           7        6  tablet      ios
              8           8        6  mobile      ios
              9           9        8  tablet      ios,
      'transactions':
              transaction_id  session_id           timestamp  amount  approved
              0               0           0 2019-01-01 12:34:32   100.0      True
              1               1           0 2019-01-01 12:42:21    55.3      True
              2               2           1 2019-01-07 17:23:11    79.5      True
              3               3           3 2019-01-10 11:08:57   112.1     False
              4               4           5 2019-01-10 21:54:08   110.0     False
              5               5           5 2019-01-11 11:21:20    76.3      True
              6               6           7 2019-01-22 14:44:10    89.5      True
              7               7           8 2019-01-23 10:14:09   132.1     False
              8               8           9 2019-01-27 16:09:17    68.0      True
              9               9           9 2019-01-29 12:10:48    99.9      True
    }
    ```
- `metadata`: python `dict` that contains the information about the fields, primary keys and
foreign keys for those tables as described in the [metadata section](#metadata).
    ```
    {'tables': [{'fields': [{'name': 'user_id', 'type': 'id'},
                        {'name': 'country', 'type': 'categorical'},
                        {'name': 'gender', 'type': 'categorical'},
                        {'name': 'age',
                         'subtype': 'integer',
                         'type': 'numerical'}],
             'name': 'users',
             'primary_key': 'user_id'},
            {'fields': [{'name': 'session_id', 'type': 'id'},
                        {'name': 'user_id',
                         'ref': {'field': 'user_id', 'table': 'users'},
                         'type': 'id'},
                        {'name': 'device', 'type': 'categorical'},
                        {'name': 'os', 'type': 'categorical'}],
             'name': 'sessions',
             'primary_key': 'session_id'},
            {'fields': [{'name': 'transaction_id', 'type': 'id'},
                        {'name': 'session_id',
                         'ref': {'field': 'session_id', 'table': 'sessions'},
                         'type': 'id'},
                        {'format': '%Y-%m-%d',
                         'name': 'timestamp',
                         'type': 'datetime'},
                        {'name': 'amount',
                         'subtype': 'float',
                         'type': 'numerical'},
                        {'name': 'approved', 'type': 'boolean'}],
             'name': 'transactions',
             'primary_key': 'transaction_id'}]
    }
    ```

### 2. Create SDV instance and fit

Before sampling, first we have to `fit` our `SDV`. In order to do so we have to import it,
instantiate it and fit it with the `metadata` and `tables` that we obtained before:

```python
from sdv import SDV

sdv = SDV()
sdv.fit(metadata, tables)
```

Once we start the fitting process, logger messages with the status will be displayed, if those
are allowed:

```
INFO - modeler - Modeling data
INFO - modeler - Modeling Complete
```

Once `Modeling Complete` is displayed, or the process of fitting is completed, we can process to
sample data.

### 3. Sample data

Sampling data once we have fitted our `sdv` instance is as simple as calling the `sample_all`
method with the desired amount of samples, by default is `5`:

```python
samples = sdv.sample_all()
```

This will generate `5` samples of all the `dataframes` that we had in our `tables`.
**Bear in mind** that this is sampled data, so you will probably obtain different results as the
ones shown below.

```
samples['users']

   user_id country gender  age
0        0      FR    NaN   59
1        1     USA      F   42
2        2      ES      F   38
3        3      ES      M   44
4        4     USA      M   30
```

```
samples['sessions']

   session_id  user_id  device       os
0           0        0  mobile      ios
1           1        1  tablet      ios
2           2        3  mobile      ios
3           3        4  tablet  android
```

```
samples['transactions']

    transaction_id  session_id                     timestamp      amount  approved
0                0           0 2019-01-04 13:17:04.294821120   93.705583      True
1                1           0 2019-01-04 13:17:03.939597824  111.825632      True
2                2           1 2019-01-15 22:57:00.760709376  107.607127      True
3                3           1 2019-01-15 23:08:28.680436224   64.862206      True
4                4           1 2019-01-15 23:03:14.034901504  113.418620      True
5                5           1 2019-01-15 23:16:01.957336320   63.494097      True
6                6           0 2019-01-04 13:17:04.215221248  101.454611      True
7                7           1 2019-01-15 23:22:02.995152128   94.099341      True
8                8           1 2019-01-15 23:13:38.940339968   91.468082      True
9                9           1 2019-01-15 23:09:28.751503616   77.090736      True
```

**Notice that** as there is a relation between the tables, `SDV`, may generate different amounts
of rows for the `child` tables. When the tables are related between them, `SDV` learns this
distribution and generates similar output. Only the parent table is ensured to have as many rows
as we specified.

# What's next?

If you would like to see more usage examples, please head to
[examples](https://github.com/HDI-Project/SDV/tree/master/examples).

However, if you would like more details about **SDV** and all its possibilities and features,
please check the [project documentation site](https://HDI-Project.github.io/SDV/)!
