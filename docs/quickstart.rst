Quickstart
==========

The easiest way to use SDV is using the ``SDV`` class from the root of the package:


.. code-block:: python

    >>> from sdv import SDV

    >>> data_vault = SDV('tests/data/meta.json')
    >>> data_vault.fit()
    >>> samples = data_vault.sample_all()
    >>> for dataset in samples:
    ...    print(samples[dataset].head(3), '\n')
    CUSTOMER_ID  CUST_POSTAL_CODE  PHONE_NUMBER1  CREDIT_LIMIT COUNTRY
    0            0           61026.0   5.410825e+09        1017.0  FRANCE

    ORDER_ID  CUSTOMER_ID  ORDER_TOTAL
    0         0            0       1251.0
    1         1            0       1691.0
    2         2            0       1126.0

    ORDER_ITEM_ID  ORDER_ID  PRODUCT_ID  UNIT_PRICE  QUANTITY
    0              0         0         9.0        20.0       0.0
    1              1         0         8.0        79.0       3.0
    2              2         0         8.0        66.0       1.0



With this, we will be able to generate sintetic samples of data. The only argument we pass to
``SDV`` is a path to a JSON file containing the information of the different tables, their fields
and relations. Further explanation of how to generate this file can be found on the docs.

After instantiating the class, we call to the ``fit()`` method in order to transform and model the
data, and after that we are ready to sample rows, tables or the whole database.
