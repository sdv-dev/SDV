# -*- coding: utf-8 -*-

"""Main module."""
from DataNavigator import DataNavigator
from Sampler import Sampler
from Modeler import Modeler

dn = DataNavigator('../demo/Airbnb_demo_meta.json')
print(dn.data)
print(dn.child_map)
print(dn.parent_map)
sampler = Sampler(dn)
modeler = Modeler(dn)
modeler.model_database()
print(modeler.tables)
# print(sampler.sample_table('users'))
