from sdv import load_demo
from sdv.tabular import GaussianCopula
from sdv import Metadata

users = load_demo()['users']
model = GaussianCopula(
    primary_key='user_id',
    anonymize_fields={'country':'country_code'}
)
#model._metadata._model_kwargs.keys()
model.fit(users)
model._sample(2)

pass
