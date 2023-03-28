
import pandas as pd

from sdv.datasets.demo import get_available_demos


def test_get_available_demos_single_table():
    """Test it can get demos for single table."""
    # Run
    tables_info = get_available_demos('single_table')

    # Assert
    expected_table = pd.DataFrame({
        'dataset_name': [
            'adult', 'alarm', 'census',
            'child', 'covtype', 'expedia_hotel_logs',
            'insurance', 'intrusion', 'news'
        ],
        'size_MB': [
            '3.907448', '4.520128', '98.165608',
            '3.200128', '255.645408', '0.200128',
            '3.340128', '162.039016', '18.712096'
        ],
        'num_tables': ['1'] * 9
    })
    expected_table['size_MB'] = expected_table['size_MB'].astype(float).round(2)
    assert len(expected_table.merge(tables_info)) == len(expected_table)


def test_get_available_demos_multi_table():
    """Test it can get demos for multi table."""
    # Run
    tables_info = get_available_demos('multi_table')

    # Assert
    expected_table = pd.DataFrame({
        'dataset_name': [
            'Accidents_v1', 'Atherosclerosis_v1', 'AustralianFootball_v1',
            'Biodegradability_v1', 'Bupa_v1', 'CORA_v1', 'Carcinogenesis_v1',
            'Chess_v1', 'Countries_v1', 'DCG_v1', 'Dunur_v1', 'Elti_v1',
            'FNHK_v1', 'Facebook_v1', 'Hepatitis_std_v1', 'Mesh_v1',
            'Mooney_Family_v1', 'MuskSmall_v1', 'NBA_v1', 'NCAA_v1',
            'PTE_v1', 'Pima_v1', 'PremierLeague_v1', 'Pyrimidine_v1',
            'SAP_v1', 'SAT_v1', 'SalesDB_v1', 'Same_gen_v1',
            'Student_loan_v1', 'Telstra_v1', 'Toxicology_v1', 'Triazine_v1',
            'TubePricing_v1', 'UTube_v1', 'UW_std_v1', 'WebKP_v1',
            'airbnb-simplified', 'financial_v1', 'ftp_v1', 'genes_v1',
            'got_families', 'imdb_MovieLens_v1', 'imdb_ijs_v1', 'imdb_small_v1',
            'legalActs_v1', 'mutagenesis_v1', 'nations_v1', 'restbase_v1',
            'rossmann', 'trains_v1', 'university_v1', 'walmart', 'world_v1'
        ],
        'size_MB': [
            '296.202744', '7.916808', '32.534832', '0.692008', '0.059144', '1.987328', '1.642592',
            '0.403784', '10.52272', '0.321536', '0.020224', '0.054912', '141.560872', '1.481056',
            '0.809472', '0.101856', '0.121784', '0.646752', '0.16632', '29.137896', '1.31464',
            '0.160896', '17.37664', '0.038144', '196.479272', '0.500224', '325.19768', '0.056176',
            '0.180256', '5.503512', '1.495496', '0.156496', '15.414536', '0.135912', '0.0576',
            '1.9718', '293.14392', '94.718016', '5.45568', '0.440016', '0.001', '55.253264',
            '259.140656', '0.205728', '186.132944', '0.618088', '0.540336', '1.01452', '73.328504',
            '0.00644', '0.009632', '14.642184', '0.295032'
        ],
        'num_tables': [
            '3', '4', '4', '5', '9', '3', '6', '2', '4', '2', '17', '11', '3', '2',
            '7', '29', '68', '2', '4', '9', '38', '9', '4', '2', '4', '36', '4',
            '4', '10', '5', '4', '2', '20', '2', '4', '3', '2', '8', '2', '3', '3',
            '7', '7', '7', '5', '3', '3', '3', '2', '2', '5', '3', '3'
        ]
    })
    expected_table['size_MB'] = expected_table['size_MB'].astype(float).round(2)
    assert len(expected_table.merge(tables_info, on='dataset_name')) == len(expected_table)
