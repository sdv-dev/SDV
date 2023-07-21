from sdv.data_processing.data_processor import DataProcessor
from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table.copulas import GaussianCopulaSynthesizer
import pandas as pd

data = pd.DataFrame({
    'low': [1, 2, 3],
})
metadata = SingleTableMetadata()
metadata.add_column('low', sdtype='numerical')
metadata.update_column('low', sdtype='job', pii=True)

dp = DataProcessor(metadata)
dp.fit(data)
transformed = dp.transform(data)
reverse_transformed = dp.reverse_transform(transformed)
print(reverse_transformed)
