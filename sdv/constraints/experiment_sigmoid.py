from random import uniform

import matplotlib.pyplot as plt
import pandas as pd

from sdv.tabular import CopulaGAN
from tabular import Between

data = pd.DataFrame({
    'a': [uniform(-10, 10) for _ in range(1000)]
})

constraint = Between('a', -10.0, 10.0)
model = CopulaGAN(constraints=[constraint])
model.fit(data)
samples = model.sample()

sort = sorted(samples['a'])
print(sort[:10])
print(samples)

plt.hist(samples, bins=50)
plt.show()

# the larger the multiplier, the farther from the edges the distribution gets????
