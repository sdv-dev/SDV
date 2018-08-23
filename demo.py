import logging
from sdv.sdv import SDV


def get_logger():
    logger = logging.getLogger()
    # We create a formatter
    fmt = '%(asctime)s - %(process)d - %(module)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)

    # Setup Handler
    console_handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


LOGGER = get_logger()


def demo_airbnb():
    # Demo for Airbnb
    airbnb_sdv = SDV('demo/airbnb_demo/Airbnb_demo_meta.json')
    airbnb_sdv.fit()
    sampled_rows = {}
    sampled_rows['users'] = airbnb_sdv.sample_rows('users', 1)
    sampled_rows['sessions'] = airbnb_sdv.sample_rows('sessions', 1)
    LOGGER.info('Sampled users row: %s',
                sampled_rows['users'])
    LOGGER.info('Sampled sessions row: %s',
                sampled_rows['sessions'])
    return sampled_rows


def demo_telstra():
    # Demo for Telstra
    telstra_sdv = SDV('demo/telstra/Telstra_manual_meta.json')
    telstra_sdv.fit()
    sampled_rows = {}
    LOGGER.info('Parent map: %s',
                telstra_sdv.dn.parent_map)
    LOGGER.info('Transformed Data: %s',
                telstra_sdv.dn.transform_data())
    sampled_rows['severity_type'] = telstra_sdv.sample_rows('severity_type', 2)
    sampled_rows['resource_type'] = telstra_sdv.sample_rows('resource_type', 1)
    sampled_rows['event_type'] = telstra_sdv.sample_rows('event_type', 1)
    sampled_rows['log_feature'] = telstra_sdv.sample_rows('log_feature', 1)
    LOGGER.info('Sampled severity type rows: %s',
                sampled_rows['severity_type'])
    LOGGER.info('Sampled log feature row: %s',
                sampled_rows['log_feature'])
    LOGGER.info('Sampled resource type row: %s',
                sampled_rows['resource_type'])
    LOGGER.info('Sampeld event type row: %s',
                sampled_rows['event_type'])
    return sampled_rows


def demo_biodegradability():
    # Demo for Biodegradability
    bio_sdv = SDV('demo/biodegradability/Biodegradability_manual_meta.json')
    bio_sdv.fit()
    sampled_rows = {}
    LOGGER.info('Parent map: %s',
                bio_sdv.dn.parent_map)
    LOGGER.info('Transformed data: %s',
                bio_sdv.dn.transformed_data)
    sampled_rows['molecule'] = bio_sdv.sample_rows('molecule', 2)
    sampled_rows['atom'] = bio_sdv.sample_rows('atom', 1)
    sampled_rows['bond'] = bio_sdv.sample_rows('bond', 1)
    LOGGER.info('Sampled molecule rows: %s',
                sampled_rows['molecule'])
    LOGGER.info('Sampled atom row: %s',
                sampled_rows['atom'])
    LOGGER.info('Sampled bond row: %s',
                sampled_rows['bond'])
    return sampled_rows


if __name__ == '__main__':
    demo_airbnb()
    # demo_telstra()
    # demo_biodegradability()
