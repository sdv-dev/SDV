import logging
from sdv.DataNavigator import *
from sdv.Modeler import Modeler
from sdv.Sampler import Sampler

# Configure logger
logger = logging.getLogger('sdv.Modeler')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


# Demo for Airbnb
data_loader = CSVDataLoader('demo/Airbnb_demo_meta.json')
data_navigator = data_loader.load_data()
data_navigator.transform_data()
modeler = Modeler(data_navigator)
modeler.model_database()
# modeler.save_model('demo_model')
sampler = Sampler(data_navigator, modeler)
print(sampler.sample_rows('users', 1))
print(sampler.sample_rows('sessions', 1))

# # Demo for Telstra
# data_loader = CSVDataLoader('demo/telstra/Telstra_manual_meta.json')
# data_navigator = data_loader.load_data()
# print(data_navigator.parent_map)
# print(data_navigator.transform_data())
# modeler = Modeler(data_navigator)
# modeler.model_database()
# modeler.save_model('telstra_model')
# sampler = Sampler(data_navigator, modeler)
# print(sampler.sample_rows('severity_type', 2))
# print(sampler.sample_rows('log_feature', 1))
# print(sampler.sample_rows('resource_type', 1))
# print(sampler.sample_rows('event_type', 1))

# # Demo for Biodegradability
# data_loader = CSVDataLoader('demo/biodegradability/Biodegradability_manual_meta.json')
# data_navigator = data_loader.load_data()
# print(data_navigator.parent_map)
# print(data_navigator.transform_data())
# modeler = Modeler(data_navigator)
# modeler.model_database()
# sampler = Sampler(data_navigator, modeler)
# print('molecule', sampler.sample_rows('molecule', 2))
# print('atom', sampler.sample_rows('atom', 1))
# print('bond', sampler.sample_rows('bond', 1))
