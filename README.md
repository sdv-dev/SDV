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

# SDV - Synthetic Data Vault

Automated generative modeling and sampling

- Free software: MIT license
- Documentation: https://HDI-Project.github.io/SDV

## Overview

The goal of the Synthetic Data Vault (SDV) is to allow data scientists to navigate, model and
sample relational databases. The main access point of the library is  the class `SDV`, that wraps
the functionality of the three core classes: the `DataNavigator`, the `Modeler` and the `Sampler`.

Using these classes, users can get easy access to information about the relational database,
create generative models for tables in the database and sample rows from these models to produce
synthetic data.

## Installation

### Install with pip

The easiest way to install SDV is using `pip`

```
pip install sdv
```

### Install from sources

You can also clone the repository and install it from sources

```
git clone git@github.com:HDI-Project/SDV.git
```

After cloning the repository, it's recommended that you create a virtualenv.
In this example, we will create it using [VirtualEnvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/):

```
cd SDV
mkvirtualenv -p $(which python3.6) -a $(pwd) sdv
```

After creating the virtualenv and activating it, you can install the project by runing the following command:

```
make install
```

For development, use the following command instead, which will install some additional dependencies for code linting and testing.

```
make install-develop
```

## Usage Example

Below there is a short example about how to use SDV to model and sample a dataset composed of
relational tables.

**NOTE**: In order to be able to run this example, please make sure to have cloned the repository
and execute these commands inside it, as they rely on some of the demo data included in it.

## Using the SDV class

The easiest way to use SDV in Python is using the SDV class imported from the root of the package:

```python
from sdv import SDV

data_vault = SDV('tests/data/meta.json')
data_vault.fit()
samples = data_vault.sample_all()
```

The output of `SDV.sample_all` is a dictionary with the name of the tables as keys, and
`pandas.DataFrame` with the synthesized table as values. Below we can see a few rows of each
table:

```text
   CUSTOMER_ID  CUST_POSTAL_CODE  PHONE_NUMBER1  CREDIT_LIMIT COUNTRY
0            0           61026.0   5.410825e+09        1017.0  FRANCE
1            1           20166.0   7.446005e+09        1316.0      US
2            2           11371.0   8.993345e+09        1839.0      US

   ORDER_ID  CUSTOMER_ID  ORDER_TOTAL
0         0            0       1251.0
1         1            0       1691.0
2         2            0       1126.0

   ORDER_ITEM_ID  ORDER_ID  PRODUCT_ID  UNIT_PRICE  QUANTITY
0              0         0         9.0        20.0       0.0
1              1         0         8.0        79.0       3.0
2              2         0         8.0        66.0       1.0

```

With this, we will be able to generate sintetic samples of data. The only argument we pass to `SDV`
is a path to a JSON file containing the information of the different tables, their fields and
relations. Further explanation of how to generate this file can be found on the docs.

After instantiating the class, we call to the `fit()` method in order to transform and model the
data, and after that we are ready to sample rows, tables or the whole database.

## Using each class manually

The modelling and sampling process using SDV follows these steps:

1. We use a `DataNavigator` instance to extract relevant information from the dataset, as well as
   to transform their contents into numeric values.

2. The `DataNavigator` is then used to create a `Modeler` instance, which uses the information in
   the `DataNavigator` to create generative models of the tables.

3. The `Modeler` instance can be passed to a `Sampler` to sample rows of synthetic data.

### Using the DataNavigator

The `DataNavigator` can be used to extract useful information about a dataset, such as the
relationships between tables. Here we will use it to load the test data from the CSV files
and apply some transformations to it.

First, we will create an instance of `CSVDataLoader`, that will load the data and prepare it to use it with `DataNavigator`.
To create an instance of the `CSVDataLoader` class, the filepath to the meta.json file must be provided.

```python
from sdv import CSVDataLoader
data_loader = CSVDataLoader('tests/data/meta.json')
```

The `load_data()` function can then be used to create an instance of a `DataNavigator`.

```python
data_navigator = data_loader.load_data()
```

The `DataNavigator` stores the data as a dictionary mapping the table names to a tuple of the data
itself (represented as a `pandas.Dataframe`) and the meta information for that table. You can access
the data using the `DataNavigator.get_data` method:

```python
data_navigator.get_data('DEMO_CUSTOMERS')
```

The output of `get_data` will be a `pandas.DataFrame` containing the requested table:

```text
                           0           1           2
CUSTOMER_ID               50           4    97338810
CUST_POSTAL_CODE       11371       63145        6096
PHONE_NUMBER1     6175553295  8605551835  7035552143
CREDIT_LIMIT            1000         500        1000
COUNTRY                   UK          US      CANADA
```

The metadata can be accessed with the analogous method `DataNavigator.get_meta_data`:

```python
data_navigator.get_meta_data('DEMO_CUSTOMERS')
```

The output of this method is a `dict` with the metadata for given table:

```text
{
   'fields': {
      'CUSTOMER_ID': {
         'name': 'CUSTOMER_ID',
         'subtype': 'integer',
         'type': 'number',
         'uniques': 0,
         'regex': '^[0-9]{10}$'
      },
      'CUST_POSTAL_CODE': {
         'name': 'CUST_POSTAL_CODE',
         'subtype': 'integer',
         'type': 'categorical',
         'uniques': 0
      },
      'PHONE_NUMBER1': {
         'name': 'PHONE_NUMBER1',
         'subtype': 'integer',
         'type': 'number',
         'uniques': 0
      },
      'CREDIT_LIMIT': {
         'name': 'CREDIT_LIMIT',
         'subtype': 'integer',
         'type': 'number',
         'uniques': 0
      },
      'COUNTRY': {
         'name': 'COUNTRY',
         'type': 'categorical',
         'uniques': 0
      }
   },
   'headers': True,
   'name': 'DEMO_CUSTOMERS',
   'path': 'customers.csv',
   'primary_key': 'CUSTOMER_ID',
   'use': True
}
```

You can also use the `DataNavigator` to get parents or children of a table.

```python
data_navigator.get_parents('DEMO_ORDERS')
```

The output of `get_parents` is a `set` containing the name of the parent tables.
Analogously, the `get_children` returns the children tables.

Finally, we can use the `transform_data()` function to apply transformations from the
[RDT library](https://github.com/HDI-Project/rdt) to our data. If no transformations are provided,
the function will convert all categorical types and datetime types to numeric values by default.

```python
transformed_data = data_navigator.transform_data()
```

It will return a dictionary mapping the table name to the transformed data represented as a
`pandas.Dataframe`.

```text
{
   'DEMO_CUSTOMERS':

         CUSTOMER_ID  CUST_POSTAL_CODE  PHONE_NUMBER1  CREDIT_LIMIT   COUNTRY
      0           50          0.286191     6175553295          1000  0.610635
      1            4          0.764276     8605551835           500  0.856004
      2     97338810          0.087041     7035552143          1000  0.015171
      3       630407          0.828319     7035552143          2000  0.786255
      4       826362          0.357857     6175553295          1000  0.557042
      5     55996144          0.534755     4045553285          1000  0.213988,

   'DEMO_ORDERS':

         ORDER_ID  CUSTOMER_ID  ORDER_TOTAL
      0         1           50         2310
      1         2            4         1507
      2        10     97338810          730
      3         6     55996144          730
      4         3     55996144          939
      5         4           50         2380,

   'DEMO_ORDER_ITEMS':

         ORDER_ITEM_ID  ORDER_ID  PRODUCT_ID  UNIT_PRICE  QUANTITY
      0             100        10           7          52         8
      1             101         8           6         125         4
      2             102         1           6         125         4
      3             103         4           9         125         4
      4             104         1           9         113         4
      5             105         9          10          87         2
}
```

### Using the Modeler

The `Modeler` can be used to recursively model the data. This is important because the tables in
the data have relationships between them, that should also be modeled in order to have reliable
sampling. Let's look at the test data for example. There are three tables in this data set:
`DEMO_CUSTOMERS`, `DEMO_ORDERS` and `DEMO_ORDER_ITEMS`.


The `DEMO_ORDERS` table has a field labelled `CUSTOMER_ID`, that references the "id" field
of the `DEMO_CUSTOMERS` table. SDV wants to model not only the data, but these relationships as
well. The Modeler class is responsible for carrying out this task.

To do so, first, import from the Modeler and create an instance of the class. The Modeler must
be provided the DataNavigator and the type of model to use. If no model type is provided, it will
use a [copulas.multivariate.Gaussian Copula](https://github.com/DAI-Lab/copulas) by default. Note that in order for
the modeler to work, the DataNavigator must have already transformed its data.

```python
from sdv import Modeler
modeler = Modeler(data_navigator)
```

Then you can model the entire database. The `Modeler` will store models for every table in the
dataset, but return no result.

```python
modeler.model_database()
```

The modeler can also be saved to a file using the `save()` method. This will save a pickle file
on the specified path.

```python
modeler.save('demo_model.pkl')
```

If you have stored a model in a previous session using the command above, you can load the model
using the `load()` method:

```python
modeler = Modeler.load('demo_model.pkl')
```

### Using the Sampler

The `Sampler` takes in a `Modeler` and `DataNavigator`. Using the models created in the last step,
the `Sampler` can recursively move through the tables in the dataset, and sample synthetic data.
It can be used to sample rows from specified tables, sample an entire table at once or sample the
whole database.

Let's do an example with our dataset. First import the Sampler and create an instance of
the class.

```python
from sdv import Sampler
sampler = Sampler(data_navigator, modeler)
```

To sample rows from a table, and their related childs, use the method `Sampler.sample_rows`.

```python
sampler.sample_rows('DEMO_CUSTOMERS', 5)
```

It will return a `dict` with the table name as keys and a `pandas.DataFrame` containing the
synthesized data as values, as we can see below. Please note that not all rows are shown.

```text
{
   'DEMO_CUSTOMERS':
         CUSTOMER_ID  CUST_POSTAL_CODE  PHONE_NUMBER1  CREDIT_LIMIT COUNTRY
      0            5             63145     6286474239          1778      US
      1            6             11371     4526647758           526  FRANCE
      2            7             11371     6027109756           958      UK
      3            8             63145     5712861733          1965      UK
      4            9             11371     5728456040          1383   SPAIN,

   'DEMO_ORDERS':
          ORDER_ID  CUSTOMER_ID  ORDER_TOTAL
      0         11            5         1331
      1         12            5         1952
      2         13            5         2179
      3         14            6         1160
      4         15            6         1069
      5         16            7         1090

   'DEMO_ORDER_ITEMS':
           ORDER_ITEM_ID  ORDER_ID  PRODUCT_ID  UNIT_PRICE  QUANTITY
      0              189        11          11         127         0
      1              190        11          12         122         2
      2              191        11           8          78         2
      3              192        11          12         233         3
      4              193        12          11          96         6
      5              194        12          11          65         2
}
```

To sample a whole table use the method `Sampler.sample_table`. This will create as many rows as there where in the
original database.

```python
sampler.sample_table('DEMO_CUSTOMERS')
```

The output of sampe_table is a `pandas.DataFrame` containing the synthesized table:

```text
   CUSTOMER_ID  CUST_POSTAL_CODE  PHONE_NUMBER1  CREDIT_LIMIT COUNTRY
0            0           27937.0   8.095336e+09        1029.0  CANADA
1            1           18183.0   2.761015e+09         891.0  CANADA
2            2           16402.0   4.956798e+09        1313.0   SPAIN
3            3            7116.0   8.072395e+09        1124.0  FRANCE
4            4             368.0   4.330203e+09        1186.0  FRANCE
5            5           64304.0   6.256936e+09        1113.0      US
6            6           94698.0   8.271224e+09        1086.0  CANADA

```

Finally, the entire database can be sampled using `Sampler.sample_all(num_rows)`. The `num_rows`
parameter specifies how many parent rows generate, the amount of child rows will be sampled based
on the sampled parents.

```python
samples = sampler.sample_all()
```

The variable `samples` will contain a `dict` mapping table names to the `pandas.dataFrames`
containing the sampled data.

```
{
   'DEMO_CUSTOMERS':
         CUSTOMER_ID  CUST_POSTAL_CODE  PHONE_NUMBER1  CREDIT_LIMIT COUNTRY
      0            5             63145     6286474239          1778      US
      1            6             11371     4526647758           526  FRANCE
      2            7             11371     6027109756           958      UK
      3            8             63145     5712861733          1965      UK
      4            9             11371     5728456040          1383   SPAIN,

   'DEMO_ORDERS':
          ORDER_ID  CUSTOMER_ID  ORDER_TOTAL
      0         11            5         1331
      1         12            5         1952
      2         13            5         2179
      3         14            6         1160
      4         15            6         1069
      5         16            7         1090

   'DEMO_ORDER_ITEMS':
           ORDER_ITEM_ID  ORDER_ID  PRODUCT_ID  UNIT_PRICE  QUANTITY
      0              189        11          11         127         0
      1              190        11          12         122         2
      2              191        11           8          78         2
      3              192        11          12         233         3
      4              193        12          11          96         6
      5              194        12          11          65         2
}
```

## What's next?

For more details about **SDV** and all its possibilities and features, please check the
[project documentation site](https://HDI-Project.github.io/SDV/)!
