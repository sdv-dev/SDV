import logging
from sdv.DataNavigator import CSVDataLoader
from sdv.Modeler import Modeler
from sdv.Sampler import Sampler

# Configure logger
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
LOGGER.addHandler(ch)


def demo_airbnb():
    # Demo for Airbnb
    data_loader = CSVDataLoader('demo/Airbnb_demo_meta.json')
    data_navigator = data_loader.load_data()
    data_navigator.transform_data()
    modeler = Modeler(data_navigator)
    modeler.model_database()
    sampler = Sampler(data_navigator, modeler)
    LOGGER.info('Sampled users row: %s',
                sampler.sample_rows('users', 1))
    LOGGER.info('Sampled sessions row: %s',
                sampler.sample_rows('sessions', 1))


def demo_telstra():
    # Demo for Telstra
    data_loader = CSVDataLoader('demo/telstra/Telstra_manual_meta.json')
    data_navigator = data_loader.load_data()
    LOGGER.info('Parent map: %s',
                data_navigator.parent_map)
    LOGGER.info('Transformed Data: %s',
                data_navigator.transform_data())
    modeler = Modeler(data_navigator)
    modeler.model_database()
    modeler.save_model('telstra_model')
    sampler = Sampler(data_navigator, modeler)
    LOGGER.info('Sampled severity type rows: %s',
                sampler.sample_rows('severity_type', 2))
    LOGGER.info('Sampled log feature row: %s',
                sampler.sample_rows('log_feature', 1))
    LOGGER.info('Sampled resource type row: %s',
                sampler.sample_rows('resource_type', 1))
    LOGGER.info('Sampeld event type row: %s',
                sampler.sample_rows('event_type', 1))


def demo_biodegradability():
    # Demo for Biodegradability
    data_loader = CSVDataLoader('demo/biodegradability/Biodegradability_manual_meta.json')
    data_navigator = data_loader.load_data()
    LOGGER.info('Parent map: %s',
                data_navigator.parent_map)
    LOGGER.info('Transformed data: %s',
                data_navigator.transform_data())
    modeler = Modeler(data_navigator)
    modeler.model_database()
    sampler = Sampler(data_navigator, modeler)
    LOGGER.info('Sampled molecule rows: %s',
                sampler.sample_rows('molecule', 2))
    LOGGER.info('Sampled atom row: %s',
                sampler.sample_rows('atom', 1))
    LOGGER.info('Sampled bond row: %s',
                sampler.sample_rows('bond', 1))


if __name__ == '__main__':
    # demo_airbnb()
    # demo_telstra()
    demo_biodegradability()
