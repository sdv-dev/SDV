<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab" />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![PyPi Shield](https://img.shields.io/pypi/v/SDV.svg)](https://pypi.python.org/pypi/SDV)
[![Travis CI Shield](https://travis-ci.org/sdv-dev/SDV.svg?branch=master)](https://travis-ci.org/sdv-dev/SDV)
[![Coverage Status](https://codecov.io/gh/sdv-dev/SDV/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/SDV)
[![Downloads](https://pepy.tech/badge/sdv)](https://pepy.tech/project/sdv)

# SDV - Synthetic Data Vault

* License: [MIT](https://github.com/sdv-dev/SDV/blob/master/LICENSE)
* Documentation: https://sdv-dev.github.io/SDV
* Homepage: https://github.com/sdv-dev/SDV

## Overview

The Synthetic Data Vault (SDV) is a tool that allows users to statistically model an entire
multi-table, relational dataset. Users can then use the statistical model to generate a
synthetic dataset. Synthetic data can be used to supplement, augment and in some cases replace
real data when training machine learning models. Additionally, it enables the testing of machine
learning or other data dependent software systems without the risk of exposure that comes with
data disclosure. Underneath the hood it uses a unique hierarchical generative modeling and
recursive sampling techniques.

# Install

## Requirements

**SDV** has been developed and tested on [Python 3.5, 3.6 and 3.7](https://www.python.org/downloads)

Also, although it is not strictly required, the usage of a
[virtualenv](https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **SDV** is run.

## Install with pip

The easiest and recommended way to install **SDV** is using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install sdv
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

If you want to install from source or contribute to the project please read the
[Contributing Guide](https://sdv-dev.github.io/SDV/contributing.html#get-started).


# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started using **SDV**.

## 1. Model the dataset using SDV

To model a multi table, relational dataset, we follow two steps. In the first step, we will load
the data and configures the meta data. In the second step, we will use the sdv API to fit and
save a hierarchical model. We will cover these two steps in this section using an example dataset.

### Step 1: Load example data

**SDV** comes with a toy dataset to play with, which can be loaded using the `sdv.load_demo`
function:

```python
from sdv import load_demo

metadata, tables = load_demo(metadata=True)
```

This will return two objects:

1. A `Metadata` object with all the information that **SDV** needs to know about the dataset.

For more details about how to build the `Metadata` for your own dataset, please refer to the
[Metadata](https://sdv-dev.github.io/SDV/metadata.html) section of the documentation.

2. A dictionary containing three `pandas.DataFrames` with the tables described in the
metadata object.

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

### 2. Fit a model using the SDV API.

First, we build a hierarchical statistical model of the data using **SDV**. For this we will
create an instance of the `sdv.SDV` class and use its `fit` method.

During this process, **SDV** will traverse across all the tables in your dataset following the
primary key-foreign key relationships and learn the probability distributions of the values in
the columns.

```python
from sdv import SDV

sdv = SDV()
sdv.fit(metadata, tables)
```

Once the modeling has finished, you can save your fitted `SDV` instance for later usage
using the `save` method of your instance.

```python
sdv.save('path/to/sdv.pkl')
```

The generated `pkl` file will not include any of the original data in it, so it can be
safely sent to where the synthetic data will be generated without any privacy concerns.

## 2. Sample data from the fitted model

In order to sample data from the fitted model, we will first need to load it from its
`pkl` file. Note that you can skip this step if you are running all the steps sequentially
within the same python session.

```python
sdv = SDV.load('path/to/sdv.pkl')
```

After loading the instance, we can sample synthetic data using its `sample_all` method,
passing the number of rows that we want to generate.

```python
samples = sdv.sample_all(5)
```

The output will be a dictionary with the same structure as the original `tables` dict,
but filled with synthetic data instead of the real one.

**Note** that only the parent tables of your dataset will have the specified number of rows,
as the number of child rows that each row in the parent table has is also sampled following
the original distribution of your dataset.

# Join out community

1. If you would like to see more usage examples, please have a look at the [examples folder](
https://github.com/sdv-dev/SDV/tree/master/examples) or the repository. Please contact us
if you have a usage example that you would want to share with the community.
2. Please head to the [Contributing Guide](https://sdv-dev.github.io/SDV/contributing.html#get-started)
for more details about this process.
3. If you have any doubts, feature requests or detect an error, please [open an issue on github](
https://github.com/sdv-dev/SDV/issues)
4. Also do not forget to check the [project documentation site](https://sdv-dev.github.io/SDV/)!

# Citation

If you use **SDV** for your research, please consider citing the following paper:

Neha Patki, Roy Wedge, Kalyan Veeramachaneni. [The Synthetic Data Vault](https://dai.lids.mit.edu/wp-content/uploads/2018/03/SDV.pdf). [IEEE DSAA 2016](https://ieeexplore.ieee.org/document/7796926).

```
@inproceedings{
    7796926,
    author={N. {Patki} and R. {Wedge} and K. {Veeramachaneni}},
    booktitle={2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA)},
    title={The Synthetic Data Vault},
    year={2016},
    volume={},
    number={},
    pages={399-410},
    keywords={data analysis;relational databases;synthetic data vault;SDV;generative model;relational database;multivariate modelling;predictive model;data analysis;data science;Data models;Databases;Computational modeling;Predictive models;Hidden Markov models;Numerical models;Synthetic data generation;crowd sourcing;data science;predictive modeling},
    doi={10.1109/DSAA.2016.49},
    ISSN={},
    month={Oct}
}
```
