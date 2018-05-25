# -*- coding: utf-8 -*-

"""Main module."""
from DataNavigator import DataNavigator
from Sampler import Sampler
from Modeler import Modeler
from copulas.multivariate.GaussianCopula import GaussianCopula


# try out data navigator and modeler
dn = DataNavigator('../demo/Airbnb_demo_meta.json')
print(dn.data)
print(dn.child_map)
print(dn.parent_map)
sampler = Sampler(dn)
modeler = Modeler(dn)
# modeler.model_database()
# print(modeler.tables)

# print('Sample rows before parent', sampler.sample_rows('sessions', 10))
# print('Sample Table', sampler.sample_table('users'))
# print('sample rows', sampler.sample_rows('users', 5))
# print('Sample rows after parent', sampler.sample_rows('sessions', 10))
print('sample all', sampler.sample_all())
