import pandas as pd

column = pd.Series[1,2,3]
print(column[pd.isna(column) | pd.to_numeric(column, errors='coerce')])