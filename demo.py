import logging
from sdv.sdv import SDV


def get_logger():
    logger = logging.getLogger()  # Nameless
    # We create a formatter
    fmt = '%(asctime)s - %(process)d - %(module)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)

    # Setup Handler
    console_handler = logging.StreamHandler()  # This one is for consoles.
    logger.setLevel(logging.INFO)    # Level of messages that will filtered out
    console_handler.setFormatter(formatter)   # Message format
    logger.addHandler(console_handler)
    return logger


LOGGER = get_logger()


def demo_airbnb():
    # Demo for Airbnb
    airbnb_sdv = SDV('demo/Airbnb_demo_meta.json')
    airbnb_sdv.fit()
    LOGGER.info('Sampled users row: %s',
                airbnb_sdv.sample_rows('users', 1))
    LOGGER.info('Sampled sessions row: %s',
                airbnb_sdv.sample_rows('sessions', 1))


def demo_telstra():
    # Demo for Telstra
    telstra_sdv = SDV('demo/telstra/Telstra_manual_meta.json')
    telstra_sdv.fit()
    LOGGER.info('Parent map: %s',
                telstra_sdv.dn.parent_map)
    LOGGER.info('Transformed Data: %s',
                telstra_sdv.dn.transform_data())
    LOGGER.info('Sampled severity type rows: %s',
                telstra_sdv.sample_rows('severity_type', 2))
    LOGGER.info('Sampled log feature row: %s',
                telstra_sdv.sample_rows('log_feature', 1))
    LOGGER.info('Sampled resource type row: %s',
                telstra_sdv.sample_rows('resource_type', 1))
    LOGGER.info('Sampeld event type row: %s',
                telstra_sdv.sample_rows('event_type', 1))


def demo_biodegradability():
    # Demo for Biodegradability
    bio_sdv = SDV('demo/biodegradability/Biodegradability_manual_meta.json')
    bio_sdv.fit()
    LOGGER.info('Parent map: %s',
                bio_sdv.dn.parent_map)
    LOGGER.info('Transformed data: %s',
                bio_sdv.dn.transformed_data)
    LOGGER.info('Sampled molecule rows: %s',
                bio_sdv.sample_rows('molecule', 2))
    LOGGER.info('Sampled atom row: %s',
                bio_sdv.sample_rows('atom', 1))
    LOGGER.info('Sampled bond row: %s',
                bio_sdv.sample_rows('bond', 1))


if __name__ == '__main__':
    # demo_airbnb()
    # demo_telstra()
    demo_biodegradability()
