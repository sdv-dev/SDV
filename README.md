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

- License: MIT
- Documentation: https://HDI-Project.github.io/SDV
- Homepage: https://github.com/HDI-Project/SDV

## Overview

**SDV** is an automated generative modeling and sampling tool that allows the users to generate
synthetic data after creating generative models for their data.

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

# Data Format

In order to work with **SDV** you will need to generate a `MetaData` that contains information
about the tables that you would like to sample and also provide those tables.

## Metadata

The `Metadata` can be a python `dict` or a `json` file consisting of multiple parts. At the highest
level of the object, there is information about the path to the dataset and a list of table
objects each representing a data table in the dataset. Each table object contains information
about its row and other important information. The structure of the meta.json object is described
below.

### Meta Object

- **path** - A string representing the path to the dataset.
- **tables** - A list of table objects.

### Table Object

- **path** - A string containing the path to the table's `csv` file.
- **name** - A string representing the name of the table for reference.
- **primary_key** - A string containing the name of the primary key column.
- **headers** - Boolean that represents wheither or not the table contains a header row.
- **use** - Boolean that represents wheither or not to use this table.
- **fields** - A list of field objects in the table.

### Field Object

- **name** - A string representing the name of the field.
- **type** - A string representing the type of the field.
- **subtype** - A string representing the subtype.
- **ref** - An object that represennts a foreign key, a reference to another table's primary key.

### Ref Object

- **table** - A string representing the name of the table that's primary key is being referenced.
- **field** - A string representing the name of the field that is the primary key.

**Bear in mind** that primary keys can only be of `type` `id` and subtype `number` or
`categorical`. More detailed information about how to generate a more proper `metadata` can be
found at the [project documentation site](https://HDI-Project.github.io/SDV/).

## Dataset / Datatable

In order to work with `SDV` you will need your tables to be a `.csv` file separeted with `,` and
it's path specified in the `metadata` as described above. Also, you can create a python `dict`
object containing as `key` the given `name` in the `metadata` and as value an instance of
a `pandas.DataFrame`.

# Quickstart

In this short series of tutorials we will guide you through a series of steps that will help you
getting started using **SDV** to sample columns, tables and datasets.

## 1. Load some demo datasets and metadata

As we explained before, we will need some data and the metadata corresponding to this data in order
to work with **SDV**. In this example, we will use the function `get_demo` from `sdv.demo` module.
This will return us `metadata` and `tables`.

```python
from sdv.demo import get_demo

metadata, tables = get_demo()
```

The returned objects contain the following information:

- `tables`: python `dict` that contains three tables (`users`, `sessions` and `transactions`).
- `metadata`: python `dict` that contains the information about the fields, primary keys and
foreign keys for those tables as described in the [metadata section](#metadata).


### 2. Create SDV instance and fit

Before sampling, first we have to `fit` our `SDV`. In order to do so we have to import it,
instantiate it and fit it with the `metadata` and `tables` that we obtained before:

```python
from sdv import SDV

sdv = SDV()
sdv.fit(metadata, tables)
```

Once we start the fitting process, logger messages with the status will be displayed:

```
INFO - modeler - Modeling data
INFO - modeler - Modeling Complete
```

Once `Modeling Complete` is displayed, we can process to sample data.

### 3. Sample data

Sampling data once we have fitted our `sdv` instance is as simple as:

```python
samples = sdv.sample_all()
```

This will generate `5` samples of all the `dataframes` that we had in our `tables`.
**Bear in mind** that this is sampled data, so you will probably obtain different results as the
ones shown below.

```
samples['users']

   user_id  country gender  age
0        0   Canada      F   60
1        1  Germany      M   45
2        2   France      F   44
3        3  Germany      F   40
4        4   Canada      M   46
```

```
samples['sessions']

   session_id  user_id device_type operative_system
0           0        0      mobile          windows
1           1        1      mobile          windows
2           2        1      mobile          android
3           3        2      tablet              ios
4           4        2      tablet              ios
```

```
samples['transactions']

   transaction_id  session_id                      datetime      amount  approved
0               0           0 2018-04-13 10:01:25.053699072  701.361518      True
1               1           0 2018-04-13 10:01:25.053699072  699.960940      True
2               2           0 2018-04-13 10:01:25.053699072  700.801190      True
3               3           0 2018-04-13 10:01:25.053699072  700.932238      True
4               4           1                           NaT  558.657914      True
```

# What's next?

For more details about **SDV** and all its possibilities and features, please check the
[project documentation site](https://HDI-Project.github.io/SDV/)!
