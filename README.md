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
The goal of the Synthetic Data Vault (SDV) is to allow data scientists to navigate, model, and sample relational databases. The library is divided into three core classes: the DataNavigator, the Modeler and the Sampler. Using these classes, users can get easy access to information about the relational database, create generative models for tables in the database and sample rows from these models to produce synthetic data.

The overall flow of SDV is as follows: the DataNavigator extracts relevant information from the dataset, as well as applies desired transformations. This class is then passed into a Modeler, which uses the information in the DataNavigator to create generative models of the tables. Finally, a Modeler can be given to a Sampler to actual sample rows of synthetic data.

### DataNavigator
The DataNavigator.py file has two classes: the DataLoader, and the DataNavigator itself. The DataLoader class is responsible takes in correctly formatted [meta.json](https://hdi-project.github.io/MetaData.json/index) file and returns an instance of a DataNavigator. The DataNavigator class uses the information in the meta.json to find relationships between tables in the database, and provides methods to access this information. The DataNavigator also stores an instance of the HyperTransformer class, from [RDT](https://github.com/HDI-Project/RDT), that is used to transform and reverse transform the data.

### Modeler
The Modeler class takes in a DataNavigator as well as the desired model type. The default model used is a [Gaussian Copula](https://github.com/DAI-Lab/Copulas). The Modeler recursively models every table in the database using the specified model type. These models can then be saved and loaded for later use.

### Sampler
The Sampler class takes in an instance of a Modeler and uses that to sample rows of synthetic data. It also uses the HyperTransformer in the DataNavigator to reverse transform all of the data back into the original format.

## Setup/Installation
### Using a virtual environment
Although not necessary, a virtual environment makes it simpler to install and run the code in SDV.
- Example using Conda
```bash
$ conda create -n sdv_env python=3.6
```
To activate the environment use
```bash
$ conda activate sdv_demo
```
To deactivate the environment use
```bash
$ conda deactivate
```
### Installing requirements using pip
The easiest way to install all of the requirements for SDV is to use pip.
```bash
$ pip install -r requirements.txt
```
### Installing demo files
If you desire to use the demo files to test out SDV, you can install them using the following command:
```bash
$ python demo_downloader.py
```

## Usage

### Using the DataNavigator
The DataNavigator can be used to extract useful information about a dataset. It can also be used to apply transformations.

First, import everything from DataNavigator and load the data.
```python
>>> from sdv.DataNavigator import *
>>> data_loader = CSVDataLoader('demo/Airbnb_demo_meta.json')
>>> data_navigator = data_loader.load_data()
```
You can use the data navigator to get parents of a certain table, get the children, access the data or transform data.
```python
>>> data_navigator.get_parents('sessions')
{'users'}

>>> data_navigator.get_children('users')
{'sessions'}

>>> data_navigator.get_data()
{'users': (             id date_account_created  timestamp_first_active  \
0    d1mm9tcy42           2014-01-01          20140101000936   
1    yo8nz8bqcq           2014-01-01          20140101001558   
2    4grx6yxeby           2014-01-01          20140101001639   
3    ncf87guaf0           2014-01-01          20140101002146   
4    4rvqpxoh3h           2014-01-01          20140101002619   
...

>>> data_navigator.transform_data()
{'users':      date_account_created  timestamp_first_active  date_first_booking  \
0            1.388552e+18            1.388553e+18        1.388812e+18   
1            1.388552e+18            1.388553e+18                 NaN   
2            1.388552e+18            1.388553e+18                 NaN   
3            1.388552e+18            1.388554e+18                 NaN   
4            1.388552e+18            1.388554e+18        1.388639e+18   
5            1.388552e+18            1.388554e+18                 NaN   
6            1.388552e+18            1.388554e+18        1.389071e+18   
7            1.388552e+18            1.388555e+18                 NaN   
...
```

### Using the Modeler
The Modeler can be used to recursively model the database. An instance of the class can be saved for easy access.

First, import from the Modeler and create an instance of the class. Note that in order for the modeler to work, the DataNavigator must have transformed its data.
```python
>>> from sdv.Modeler import Modeler
>>> modeler = Modeler(data_navigator)
```
Then you can model the entire database.
```python
>>> modeler.model_database()
```
The modeler can then be saved.
```python
>>> modeler.save_model('demo_model')
```
To load a model, import the load_model function from utils.
```python
>>> from sdv.utils import load_model
>>> modeler = load_model('sdv/models/demo_model.pkl')
```
### Using the Sampler
The sampler takes in a Modeler and DataNavigator. It can be used to sample rows from specified tables, sample an entire table at once or sample the whole database.

First import the Sampler and create an instance of the class.
```python
>>> from sdv.Sampler import Sampler
>>> sampler = Sampler(data_navigator, modeler)
```

To sample from a row, use the command below. Note that before sampling from a child table, one of its parent tables must be sampled from.
```python
>>> sampler.sample_rows('users', 1)
0           2014-01-02         20140102175145         2014-01-30  -unknown-   

   age signup_method  signup_flow language affiliate_channel  \
0   20         basic            0       en            direct   

  affiliate_provider first_affiliate_tracked signup_app first_device_type  \
0             google                     omg        Web       Mac Desktop   

   first_browser  
0  Mobile Safari
```

To sample a whole table use sample_table. This will create as many rows as in the original database.
```python
>>> sampler.sample_table('users')
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

Finally, the entire database can be sampled using sample_all(num_rows). The num_rows parameter specifies how many child rows to create per parent row. This function returns a dictionary mapping table names to the generated dataframes.

```python
>>> sampler.sample_all()
{'users':   date_account_created timestamp_first_active date_first_booking     gender  \
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
