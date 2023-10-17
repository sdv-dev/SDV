import pandas as pd

from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer

data = pd.DataFrame({'a': [], 'b': [], 'c': []})
metadata = SingleTableMetadata.load_from_dict({
    'columns': {
        'a': { 'sdtype': 'numerical' },
        'b': { 'sdtype': 'numerical' },
        'c': { 'sdtype': 'numerical' }
    }
})
constraint = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'a', 
        'high_column_name': 'b'
    }
}

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.add_constraints(constraints=[constraint])
synthesizer.fit(data)
sample = synthesizer.sample(num_rows=5)
print(sample.to_string())