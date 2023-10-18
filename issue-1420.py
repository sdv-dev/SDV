from sdv.datasets.demo import get_available_demos

demos1 = get_available_demos(modality='multi_table')
print(demos1.dtypes)
print(demos1['num_tables'])

demos2 = get_available_demos(modality='single_table')
print(demos2.dtypes)
print(demos2['num_tables'])

demos3 = get_available_demos(modality='sequential')
print(demos3.dtypes)
print(demos3['num_tables'])
