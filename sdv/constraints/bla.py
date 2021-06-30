import pandas as pd

a = pd.Series({
    'a': [1, 1, 1]
})

d2 =  pd.DataFrame({
    'a': [0, 1, 2]
})

d3 = 1

print(d2['a'] > d3)