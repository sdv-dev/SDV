from sdv.metadata.single_table import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import pandas as pd

data = pd.DataFrame({'age': [56, 61, 36, 52, 42],})
metadata = SingleTableMetadata.load_from_dict({
    'columns': {'age': {'sdtype': 'numerical'}}
})
synthesizer = CTGANSynthesizer(metadata)
synthesizer.fit(data)
print(synthesizer.sample(10))