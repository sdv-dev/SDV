<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“SDV” />
  <i>An open source project from Data to AI Lab at MIT.</i>
</p>



[![][pypi-img]][pypi-url] [![][travis-img]][travis-url]

# SDV - Synthetic Data Vault


Automated generative modeling and sampling

- Free software: MIT license
- Documentation: https://HDI-Project.github.io/sdv

[travis-img]: https://travis-ci.org/HDI-Project/sdv.svg?branch=master
[travis-url]: https://travis-ci.org/HDI-Project/sdv
[pypi-img]: https://img.shields.io/pypi/v/sdv.svg
[pypi-url]: https://pypi.python.org/pypi/sdv

## Summary

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
cd SDV
pip install -e .
```

## Usage Example

Below there is a short example about how to use SDV to model and sample a dataset composed of
relational tables.

### Using the SDV class

The easiest way to use SDV is using the SDV class from the root of the package:

```python
>>> from sdv import SDV

>>> data_vault = SDV('tests/data/meta.json')
>>> data_vault.fit()
>>> samples = data_vault.sample_all()
```

With this, we will be able to generate sintetic samples of data. The only argument we pass to `CSV`
is a path to a JSON file containing the information of the different tables, their fields and
relations. Further explanation of how to generate this file can be found on the docs.

After instantiating the class, we call to the `fit()` method in order to transform and model the
data, and after that we are ready to sample rows, tables or the whole database.

### Using each class manually

The overall flow of SDV is as follows: the DataNavigator extracts relevant information from the
dataset, as well as applies desired transformations. This class is then passed into a `Modeler`,
which uses the information in the `DataNavigator` to create generative models of the tables.
Finally, a `Modeler` can be given to a `Sampler` to actual sample rows of synthetic data.

#### DataNavigator

The `DataNavigator` can be used to extract useful information about a dataset, such as the
relationships between tables. It can also be used to apply transformations. Here we will use it to
load the test data from the CSV files and apply some transformations to it.

First, we will instantiate the `CSVDataLoader` class, that will load the data and prepare it to use it with `DataNavigator`.
To create an instance of the `CSVDataLoader` class, the filepath to the meta.json file must be provided.

```python
>>> from sdv import CSVDataLoader
>>> data_loader = CSVDataLoader('tests/data/meta.json')
```

The `load_data()` function can then be used to create an instance of a `DataNavigator`.

```python
>>> data_navigator = data_loader.load_data()
```

The `DataNavigator` stores the data as a dictionary mapping the table names to a tuple of the data
itself (represented as a `pandas.Dataframe`) and the meta information for that table. You can access
the data using the following command:

```python
>>> customer_table = data_navigator.tables['DEMO_CUSTOMERS']
>>> customer_data = customer_table.data
>>> customer_data.head(3).T

                           0           1           2
CUSTOMER_ID               50           4    97338810
CUST_POSTAL_CODE       11371       63145        6096
PHONE_NUMBER1     6175553295  8605551835  7035552143
CREDIT_LIMIT            1000         500        1000

>>> customers_meta = customer_table.meta
>>> customers_meta.keys()
dict_keys(['fields', 'headers', 'name', 'path', 'primary_key', 'use'])
>>> customers_meta['fields']
{'CUSTOMER_ID': {'name': 'CUSTOMER_ID',
  'subtype': 'integer',
  'type': 'number',
  'uniques': 0},
 'CUST_POSTAL_CODE': {'name': 'CUST_POSTAL_CODE',
  'subtype': 'integer',
  'type': 'number',
  'uniques': 0},
 'PHONE_NUMBER1': {'name': 'PHONE_NUMBER1',
  'subtype': 'integer',
  'type': 'number',
  'uniques': 0},
 'CREDIT_LIMIT': {'name': 'CREDIT_LIMIT',
  'subtype': 'integer',
  'type': 'number',
  'uniques': 0}}
```

You can also use the data navigator to get parents or children of a table.

```python
>>> data_navigator.get_parents('DEMO_ORDERS')
{'DEMO_CUSTOMERS'}

>>> data_navigator.get_children('DEMO_CUSTOMERS')
{'DEMO_ORDERS'}
```

Finally, we can use the `transform_data()` function to apply transformations from the
[RDT library](https://github.com/HDI-Project/rdt) to our data. If no transformations are provided,
the function will convert all categorical types and datetime types to numeric values by default.
It will return a dictionary mapping the table name to the transformed data represented as a
`pandas.Dataframe`.

```python
>>> transformed_data = data_navigator.transform_data()
>>> transformed_data['DEMO_CUSTOMERS'].head(3).T
                           0           1           2
CUSTOMER_ID               50           4    97338810
CUST_POSTAL_CODE       11371       63145        6096
PHONE_NUMBER1     6175553295  8605551835  7035552143
CREDIT_LIMIT            1000         500        1000
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
>>> from sdv import Modeler
>>> modeler = Modeler(data_navigator)
```

Then you can model the entire database. The modeler will store models for every table in the
dataset.

```python
>>> modeler.model_database()
```

The models that were created for each table can be accessed using the following command:

```python
>>> customers_model = modeler.models['DEMO_CUSTOMERS']
>>> print(customers_model)
CUSTOMER_ID
==============
Distribution Type: Gaussian
Variable name: CUSTOMER_ID
Mean: 22198555.57142857
Standard deviation: 30269487.960631203
Max: 97338810.0
Min: 4.0

CUST_POSTAL_CODE
==============
Distribution Type: Gaussian
Variable name: CUST_POSTAL_CODE
Mean: 34062.71428571428
Standard deviation: 21312.957555038134
Max: 63145.0
Min: 6096.0

PHONE_NUMBER1
==============
Distribution Type: Gaussian
Variable name: PHONE_NUMBER1
Mean: 6464124184.428572
Standard deviation: 1064804060.6865476
...
```
The output above shows the parameters that got stored for every column in the users table.

The modeler can also be saved to a file using the `save()` method. This will save a pickle file
on the specified path.

```python
>>> modeler.save('models/demo_model.pkl')
```

If you have stored a model in a previous session using the command above, you can load the model
using the `load()` method:

```python
>>> modeler = Modeler.load('models/demo_model.pkl')
```

### Using the Sampler

The `Sampler` takes in a `Modeler` and `DataNavigator`. Using the mdels created in the last step,
the `Sampler` can recursively move through the tables in the dataset, and sample synthetic data.
It can be used to sample rows from specified tables, sample an entire table at once or sample the
whole database.

Let's do an example with out dataset. First import the Sampler and create an instance of
the class.

```python
>>> from sdv import Sampler
>>> sampler = Sampler(data_navigator, modeler)
```

To sample from a row, use the command `sample_rows()`. Note that before sampling from a child table, one of
its parent tables must be sampled from.

```python
>>> sampler.sample_rows('DEMO_CUSTOMERS', 1).T
0           2014-01-02         20140102175145         2014-01-30  -unknown-

   age signup_method  signup_flow language affiliate_channel  \
0   20         basic            0       en            direct

  affiliate_provider first_affiliate_tracked signup_app first_device_type  \
0             google                     omg        Web       Mac Desktop

   first_browser
0  Mobile Safari
```

To sample a whole table use `sample_table()`. This will create as many rows as in the original
database.

```python
>>> sampler.sample_table('DEMO_CUSTOMERS')
date_account_created timestamp_first_active date_first_booking     gender  \
0             2014-01-04         20140104172305         2014-01-09       MALE
1             2014-01-01         20140101145313         2014-04-18  -unknown-
2             2014-01-01         20140101233803         2013-12-23       MALE
3             2014-01-02         20140102173933         2014-03-23  -unknown-
4             2014-01-03         20140104071157         2014-01-31  -unknown-
5             2013-12-31         20131231224951         2013-12-09  -unknown-
6             2014-01-05         20140105205012         2014-05-01  -unknown-
...
```

Finally, the entire database can be sampled using `sample_all(num_rows)`. The `num_rows` parameter
specifies how many child rows to create per parent row. This function returns a dictionary mapping
table names to the generated dataframes.

```python
>>> samples = sampler.sample_all()
>>> samples['DEMO_CUSTOMERS']
{'DEMO_CUSTOMERS':   date_account_created timestamp_first_active date_first_booking     gender  \
0           2014-01-01         20140102081228         2014-02-12  -unknown-

   age signup_method  signup_flow language affiliate_channel  \
0   60         basic            0       en         sem-brand

  affiliate_provider first_affiliate_tracked signup_app first_device_type  \
0             direct               untracked        Web       Mac Desktop

  first_browser
0        Safari  , 'sessions':           action  action_type  action_detail      device_type  \
0         create          NaN            NaN  Windows Desktop
1         search          NaN            NaN  Windows Desktop
2  confirm_email          NaN            NaN  Windows Desktop
3           edit          NaN            NaN  Windows Desktop
4   authenticate          NaN            NaN  Windows Desktop

          secs_elapsed
0  9223372036854775807
1  9223372036854775807
2  9223372036854775807
3  9223372036854775807
4  9223372036854775807  }
```
