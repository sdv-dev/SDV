.. _sdgym_datasets:

SDGym Datasets
==============

**SDGym** uses SDV datasets to benchmark the **Synthesizers** which are
in three data modalities:

-  Single Table Datasets: Datasets that contain only one table with no
   inter-row dependencies.
-  Multi Table Datasets: Datasets that contain more than one table,
   potentially with relationships between them.
-  Time Series Datasets: Datasets that contain a single table that
   represents sequences of rows.

Dataset Format
--------------

The **SDV Datasets** are comprised of two elements:

-  A ``metadata.json`` file which describes the data found in the
   dataset. This file follows the `SDV Metadata
   schema <https://sdv.dev/SDV/developer_guides/sdv/metadata.html>`__
-  A collection of ``CSV`` files stored in a format which can be loaded
   by the ``pandas.read_csv`` function without any additional arguments
   than the csv path.

Using the datasets
------------------

All the datasets can also be found for download inside the `sdv-datasets
S3 bucket <http://sdv-datasets.s3.amazonaws.com/index.html>`__ in the
form of a ``.zip`` file that contains both the ``metadata.json`` and the
``CSV`` file collection.

In order to load these datasets in the same format as they will be
passed to your synthesizer you can use the ``sdgym.load_dataset``
function passing the name of the dataset to load.

In this example, we will load the ``adult`` dataset:

.. code:: python3

   In [1]: from sdgym.datasets import load_dataset

   In [2]: metadata = load_dataset('adult')

This will read the ``metadata.json`` file and return it as a
``sdv.Metadata`` instance.

.. code:: python

   In [3]: metadata
   Out[3]:
   Metadata
     root_path: /home/xals/.local/share/SDGym/datasets/adult
     tables: ['adult']
     relationships:

Afterwards, you can load the tables from the dataset passing the loaded
``metadata`` to the ``sdgym.load_tables`` function:

.. code:: python3

   In [4]: from sdgym.datasets import load_tables

   In [5]: tables = load_tables(metadata)

This will return a ``dict`` containing the tables loaded as
``pandas.DataFrames``.

.. code:: python3

   In [6]: tables
   Out[6]:
   {'adult':        age  workclass  fnlwgt     education  education-num  ... capital-gain capital-loss hours-per-week native-country  label
    0       27    Private  177119  Some-college             10  ...            0            0             44  United-States  <=50K
    1       27    Private  216481     Bachelors             13  ...            0            0             40  United-States  <=50K
    2       25    Private  256263    Assoc-acdm             12  ...            0            0             40  United-States  <=50K
    3       46    Private  147640       5th-6th              3  ...            0         1902             40  United-States  <=50K
    4       45    Private  172822          11th              7  ...            0         2824             76  United-States   >50K
    ...    ...        ...     ...           ...            ...  ...          ...          ...            ...            ...    ...
    32556   43  Local-gov   33331       Masters             14  ...            0            0             40  United-States   >50K
    32557   44    Private   98466          10th              6  ...            0            0             35  United-States  <=50K
    32558   23    Private   45317  Some-college             10  ...            0            0             40  United-States  <=50K
    32559   45  Local-gov  215862     Doctorate             16  ...         7688            0             45  United-States   >50K
    32560   25    Private  186925  Some-college             10  ...         2597            0             48  United-States  <=50K

    [32561 rows x 15 columns]}

Getting the list of all the datasets
------------------------------------

If you want to obtain the list of all the available datasets you can use
the ``sdgym.datasets.get_available_datasets`` function:

.. code:: python

   In [7]: from sdgym.datasets import get_available_datasets

   In [8]: get_available_datasets()
   Out[8]:
                             name      size
   0                 Accidents_v1  44717026
   1    ArticularyWordRecognition   1928334
   2           Atherosclerosis_v1    521308
   3           AtrialFibrillation    111036
   4        AustralianFootball_v1   3500419
   ..                         ...       ...
   99      student_placements_pii     11602
   100                  trains_v1      1772
   101              university_v1      3226
   102                    walmart   3566966
   103                   world_v1    110291

   [104 rows x 2 columns]

How to add your own dataset to SDGym?
-------------------------------------

Coming soon!
