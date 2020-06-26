# SDV Evaluation

After using SDV to model your database and generate a synthetic version of it you
might want to evaluate how similar the syntehtic data is to your real data.

SDV has an evaluation module with a simple function that allows you to compare
the syntehtic data to your real data using [SDMetrics](/sdv-dev/SDMetrics) and
generate a simple standardized score.

## Evaluating your synthetic data

After you have modeled your databased and generated samples out of the SDV models
you will be left with a dictionary that contains table names and dataframes.

For exmple, if we model and sample the demo dataset:

```python3
from sdv import SDV
from sdv.demo import load_demo

metadata, tables = load_demo(metadata=True)

sdv = SDV()
sdv.fit(metadata, tables)

samples = sdv.sample_all(10)
```

`samples` will contain a dictionary with three tables, just like the `tables` dict.


At this point, you can evaluate how similar the two sets of tables are by using the
`sdv.evaluation.evaluate` function as follows:

```
from sdv.evaluation import evaluate

score = evaluate(samples, tables, metadata)
```

The output will be a maximization score that will indicate how good the modeling was:
the higher the value, the more similar the sets of table are. Notice that in most cases
the value will be negative.

For further options, including visualizations and more detailed reports, please refer to
the [SDMetrics](/sdv-dev/SDMetrics) library.
