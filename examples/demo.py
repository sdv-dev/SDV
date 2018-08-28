import logging
import os
import pickle
import sys

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
    airbnb_sdv = SDV('../demo/airbnb_demo/Airbnb_demo_meta.json')
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
    telstra_sdv = SDV('../demo/telstra/Telstra_manual_meta.json')
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
    bio_sdv = SDV('../demo/biodegradability/Biodegradability_manual_meta.json')
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


def demo_rossman():
    # Demo for Rossman
    rossman_sdv = SDV('../demo/rossmann/Rossman_manual_meta.json')
    rossman_sdv.fit()
    sampled_rows = {}
    LOGGER.info('Parent map: %s',
                rossman_sdv.dn.parent_map)
    sampled_rows['store'] = rossman_sdv.sample_rows('store', 1)
    sampled_rows['train'] = rossman_sdv.sample_rows('train', 1)
    LOGGER.info('Sampled store rows: %s',
                sampled_rows['store'])
    LOGGER.info('Sampled train row: %s',
                sampled_rows['train'])
    return sampled_rows


def demo_mutagenesis():
    # Demo for Biodegradability
    mutagenesis_sdv = SDV('../demo/mutagenesis/Mutagenesis_manual_meta.json')
    mutagenesis_sdv.fit()
    sampled_rows = {}
    LOGGER.info('Parent map: %s',
                mutagenesis_sdv.dn.parent_map)
    LOGGER.info('Transformed data: %s',
                mutagenesis_sdv.dn.transformed_data)
    sampled_rows['molecule'] = mutagenesis_sdv.sample_rows('molecule', 2)
    sampled_rows['atom'] = mutagenesis_sdv.sample_rows('atom', 1)
    sampled_rows['bond'] = mutagenesis_sdv.sample_rows('bond', 1)
    LOGGER.info('Sampled molecule rows: %s',
                sampled_rows['molecule'])
    LOGGER.info('Sampled atom row: %s',
                sampled_rows['atom'])
    LOGGER.info('Sampled bond row: %s',
                sampled_rows['bond'])
    return sampled_rows


def demo_walmart():
    # Demo for Biodegradability
    walmart_sdv = SDV('../demo/walmart/Walmart_manual_meta.json')
    walmart_sdv.fit()
    sampled_rows = {}
    LOGGER.info('Parent map: %s',
                walmart_sdv.dn.parent_map)
    LOGGER.info('Transformed data: %s',
                walmart_sdv.dn.transformed_data)
    sampled_rows['stores'] = walmart_sdv.sample_rows('stores', 2)
    sampled_rows['train'] = walmart_sdv.sample_rows('train', 1)
    sampled_rows['features'] = walmart_sdv.sample_rows('features', 1)
    LOGGER.info('Sampled stores rows: %s',
                sampled_rows['stores'])
    LOGGER.info('Sampled train row: %s',
                sampled_rows['train'])
    LOGGER.info('Sampled features row: %s',
                sampled_rows['features'])
    return sampled_rows


def demo_coupon():
    # Demo for coupon purchase prediction
    coupon_sdv = SDV('../demo/coupon_purchase_prediction/Coupon_purchase_prediction_manual_meta.json')
    coupon_sdv.fit()
    sampled_rows = {}
    LOGGER.info('Parent map: %s',
                coupon_sdv.dn.parent_map)
    LOGGER.info('Transformed data: %s',
                coupon_sdv.dn.transformed_data)
    sampled_rows['prefecture_locations'] = coupon_sdv.sample_rows('prefecture_locations', 1)
    sampled_rows['user_list'] = coupon_sdv.sample_rows('user_list', 1)
    sampled_rows['coupon_list'] = coupon_sdv.sample_rows('coupon_list_train', 1)
    sampled_rows['coupon_visit'] = coupon_sdv.sample_rows('coupon_visit_train', 1)
    sampled_rows['coupon_detail'] = coupon_sdv.sample_rows('coupon_detail_train', 1)
    sampled_rows['coupon_area'] = coupon_sdv.sample_rows('coupon_area_train', 1)
    LOGGER.info('Sampled user list rows: %s',
                sampled_rows['user_list'])
    LOGGER.info('Sampled coupon list row: %s',
                sampled_rows['coupon_list'])
    LOGGER.info('Sampled prefecture locations row: %s',
                sampled_rows['prefecture_locations'])
    LOGGER.info('Sampled coupon visit rows: %s',
                sampled_rows['coupon_visit'])
    LOGGER.info('Sampled coupon detail row: %s',
                sampled_rows['coupon_detail'])
    LOGGER.info('Sampled coupon area row: %s',
                sampled_rows['coupon_area'])
    return sampled_rows


def demo_nips():
    # Demo for nips papers
    nips_sdv = SDV('../demo/nips_2015_papers/nips_2015_papers_manual_meta.json')
    nips_sdv.fit()
    sampled_rows = {}
    LOGGER.info('Parent map: %s',
                nips_sdv.dn.parent_map)
    LOGGER.info('Transformed data: %s',
                nips_sdv.dn.transformed_data)
    sampled_rows['Authors'] = nips_sdv.sample_rows('Authors', 1)
    sampled_rows['Papers'] = nips_sdv.sample_rows('Papers', 1)
    sampled_rows['PaperAuthors'] = nips_sdv.sample_rows('PaperAuthors', 1)
    LOGGER.info('Sampled authors rows: %s',
                sampled_rows['Authors'])
    LOGGER.info('Sampled papers row: %s',
                sampled_rows['Papers'])
    LOGGER.info('Sampled paper authors row: %s',
                sampled_rows['PaperAuthors'])
    return sampled_rows


def demo_clinton():
    # Demo or coupon purchase prediction
    clinton_sdv = SDV('../demo/hilary_clinton_emails/Hilary_clinton_emails_manual_meta.json')
    clinton_sdv.fit()
    sampled_rows = {}
    LOGGER.info('Parent map: %s',
                clinton_sdv.dn.parent_map)
    LOGGER.info('Transformed data: %s',
                clinton_sdv.dn.transformed_data)
    sampled_rows['Persons'] = clinton_sdv.sample_rows('Persons', 2)
    sampled_rows['Emails'] = clinton_sdv.sample_rows('Emails', 1)
    sampled_rows['Aliases'] = clinton_sdv.sample_rows('Aliases', 1)
    sampled_rows['EmailReceivers'] = clinton_sdv.sample_rows('EmailReceivers', 1)
    LOGGER.info('Sampled persons rows: %s',
                sampled_rows['Persons'])
    LOGGER.info('Sampled emails row: %s',
                sampled_rows['Emails'])
    LOGGER.info('Sampled Aliases row: %s',
                sampled_rows['Aliases'])
    LOGGER.info('Sampled email receivers row: %s',
                sampled_rows['EmailReceivers'])
    return sampled_rows


def demo_world_dev():
    # Demo for world development indicators
    world_dev_sdv = SDV('../demo/world_development_indicators/World_development_indicators_manual_meta.json')
    world_dev_sdv.fit()
    sampled_rows = {}
    LOGGER.info('Parent map: %s',
                world_dev_sdv.dn.parent_map)
    LOGGER.info('Transformed data: %s',
                world_dev_sdv.dn.transformed_data)
    sampled_rows['Country'] = world_dev_sdv.sample_rows('Country', 1)
    sampled_rows['Series'] = world_dev_sdv.sample_rows('Series', 1)
    sampled_rows['CountryNotes'] = world_dev_sdv.sample_rows('CountryNotes', 1)
    sampled_rows['Footnotes'] = world_dev_sdv.sample_rows('Footnotes', 1)
    sampled_rows['SeriesNotes'] = world_dev_sdv.sample_rows('SeriesNotes', 1)
    sampled_rows['Indicators'] = world_dev_sdv.sample_rows('Indicators', 1)
    LOGGER.info('Sampled country rows: %s',
                sampled_rows['Country'])
    LOGGER.info('Sampled series row: %s',
                sampled_rows['Series'])
    LOGGER.info('Sampled Country Note row: %s',
                sampled_rows['CountryNotes'])
    LOGGER.info('Sampled Footnote row: %s',
                sampled_rows['Footnotes'])
    LOGGER.info('Sampled Series Note row: %s',
                sampled_rows['SeriesNotes'])
    LOGGER.info('Sampled indicator row: %s',
                sampled_rows['Indicators'])
    return sampled_rows


if __name__ == '__main__':
    demo_dict = {
        'airbnb_demo': demo_airbnb,
        'biodegradability': demo_biodegradability,
        'coupon_purchase_prediction': demo_coupon,
        'hilary_clinton_emails': demo_clinton,
        'mutagenesis': demo_mutagenesis,
        'nips_2015_papers': demo_nips,
        'rossmann': demo_rossman,
        'telstra': demo_telstra,
        'walmart': demo_walmart,
        'world_development_indicators': demo_world_dev
    }
    demo = demo_dict[sys.argv[1]]
    demo()
