import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.data_processing import DataProcessor
import numpy as np
from rdt.transformers import FloatFormatter, AnonymizedFaker, UniformEncoder
from sdv.single_table import GaussianCopulaSynthesizer

metadata = SingleTableMetadata().load_from_dict({
    'columns': {
        'A': {'sdtype': 'numerical'},
        'B': {'sdtype': 'numerical'},
        'C': {'sdtype': 'numerical'},
        'D': {'sdtype': 'categorical'},
    }
})
data = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9],
    'D': ['a', 'b', 'c'],
})
metadata.update_column('A', sdtype='categorical')
synthesizer = GaussianCopulaSynthesizer(metadata)
metadata.update_column('B', sdtype='categorical')
metadata.update_column('C', sdtype='categorical')
synthesizer.fit(data)