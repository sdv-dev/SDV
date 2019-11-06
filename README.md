<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“SDV” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![PyPi Shield](https://img.shields.io/pypi/v/SDV.svg)](https://pypi.python.org/pypi/SDV)
[![Travis CI Shield](https://travis-ci.org/HDI-Project/SDV.svg?branch=master)](https://travis-ci.org/HDI-Project/SDV)
[![Coverage Status](https://codecov.io/gh/HDI-Project/SDV/branch/master/graph/badge.svg)](https://codecov.io/gh/HDI-Project/SDV)
[![Downloads](https://pepy.tech/badge/sdv)](https://pepy.tech/project/sdv)

# SDV - Synthetic Data Vault

* Free Software: MIT License
* Documentation: https://HDI-Project.github.io/SDV
* Homepage: https://github.com/HDI-Project/SDV

## Overview

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

**SDV** can work with both single table and multi table, relational datasets, as far as they
comply with the following data requirements:

* All the data columns must be either numerical, categorical, boolean or datatimes. Mixed value
  types are not supported, but columns **can have null values**.
* All the tables in the dataset can have **at most one primary key**, which can either be
  numerical or categorical. Datetime indexes might be supported in future versions.
* All the tables can have **at most one foreign key to a parent table**, meaning that each table
  can have at most **one parent**.
* Tables are either loaded as `pandas.DataFrame` objects or stored as CSV files.

## Metadata

Alongside the actual tables, **SDV** needs to be provided some metadata about the dataset,
which can either be provided as a python `dict` object or as a JSON file.

For more details about the Metadata format, please refer to [the corresponding section
of the documentation](https://hdi-project.github.io/SDV/metadata.html)

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started using **SDV**.

## 1. Load some data

**SDV** comes with a toy dataset to play with, which can be loaded using the `sdv.load_demo`
function:

```python
from sdv import load_demo

metadata, tables = load_demo()
```

This will return two objects:

1. A `metadata` dictionary with all the information that **SDV** needs to know about the dataset:

```
{
    "tables": [
        {
            "fields": [
                {"name": "user_id", "type": "id"},
                {"name": "country", "type": "categorical"},
                {"name": "gender", "type": "categorical"},
                {"name": "age", "type": "numerical", "subtype": "integer"}
            ],
            "name": "users",
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
            "name": "sessions",
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
            "name": "transactions",
            "primary_key": "transaction_id"
        }
    ]
}
```

2. A dictionary containing three `pandas.DataFrames` with the tables described in the
metadata dictionary.

The returned objects contain the following information:

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

## 2. Create and fit an SDV instance

Before sampling, **SDV** needs to learn about your data in a process called *Database Modeling*.

During this process, **SDV** will walk across all the tables in your dataset learning
about the table relationships and the probability distributions of their values.

To do this, we create an instance of the `sdv.SDV` class and call its `fit`
method passing it both the `metadata` and `tables` that we obtained before:

```python
from sdv import SDV

sdv = SDV()
sdv.fit(metadata, tables)
```

## 3. Sample data

Once the modeling has finished, we can sample new data using our fitted `SDV` instance.

In order to do this, we call its `sample_all` method passing the number of rows that we
want to sample.

```python
samples = sdv.sample_all(5)
```

The output will be a dictionary with the same structure as the original `tables` dict,
but filled with synthetic data instead of the real one.

**Notice** that only the parent tables of your dataset will have the specified number of rows,
as the number of child rows that each row in the parent table has is also sampled following
the original distribution of your dataset.

## What's next?

If you would like to see more usage examples, please have a look at the [examples folder](
https://github.com/HDI-Project/SDV/tree/master/examples).

Also do not forget to check the [project documentation site](https://HDI-Project.github.io/SDV/)!
