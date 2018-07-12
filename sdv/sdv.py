# -*- coding: utf-8 -*-

"""Main module."""
from DataNavigator import DataNavigator
from Modeler import Modeler
from Sampler import Sampler

# try out data navigator and modeler
# dn = DataNavigator('../demo/Airbnb_demo_meta.json')
dn = DataNavigator('../tests/manual_data/meta.json')
print('Data', dn.data)
print('child map', dn.child_map)
print('parent map', dn.parent_map)
sampler = Sampler(dn)
modeler = Modeler(dn)
# modeler.CPA('DEMO_CUSTOMERS')
modeler.RCPA('DEMO_ORDERS')
# modeler.model_database()
print(modeler.tables)
# print(modeler.models)

# # create copula model
# model = GaussianCopula()
# data = pd.read_csv('../tests/manual_data/customers.csv')
# model.fit(data)
# params = modeler.flatten_model(model)
# print(params)

# # model the database
# modeler.model_database()
# print(modeler.tables)

# # Sample the data base
# print('Sample rows before parent', sampler.sample_rows('sessions', 10))
# print('Sample Table', sampler.sample_table('users'))
# print('sample rows', sampler.sample_rows('users', 5))
# print('Sample rows after parent', sampler.sample_rows('sessions', 10))
# print('sample all', sampler.sample_all())
