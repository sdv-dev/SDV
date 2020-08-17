from sdv import load_demo
from sdv.tabular import GaussianCopula
model = GaussianCopula(
    primary_key='user_id',
    anonymize_fields={'country':'country_code'}
)
users = load_demo()['users']

#test__get_distribution_none
cosa = model._get_distribution(True)
