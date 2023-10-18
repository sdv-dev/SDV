from sdv.datasets.demo import get_available_demos

demos = get_available_demos(modality='multi_table')
print(demos.dtypes)