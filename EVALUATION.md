# SDV Evaluation

After using SDV to model your database and generate a synthetic version of it you
might want to evaluate how similar the synthetic data is to your real data.

SDV has an evaluation module with a simple function that allows you to compare
the synthetic data to your real data using [SDMetrics](https://github.com/sdv-dev/SDMetrics) and
generate a simple standardized score.

## Evaluating your synthetic data

After you have modeled your databased and generated samples out of the SDV models
you will be left with a dictionary that contains table names and dataframes.

For example, if we model and sample the demo dataset:

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
the [SDMetrics](https://github.com/sdv-dev/SDMetrics) library.


## SDV Benchmark

SDV also provides a simple functionality to evaluate the performance of SDV across a
collection of demo datasets or custom datasets hosted in a local folder.

In order to execute this evaluation you can execute the function `sdv.benchmark.run_benchmark`:

```python3
from sdv.benchmark import run_benchmark

scores = run_benchmark()
```

This function has the following arguments:

* `datasets`: List of dataset names, which can either be names of demo datasets or
  names of custom datasets stored in a local folder.
* `datasets_path`: Path where the custom datasets are stored. If not provided, the
  dataset names are interpreted as demo datasets.
* `distributed`: Whether to execute the benchmark using Dask. Defaults to True.
* `timeout`: Maximum time allowed for each dataset to be modeled, sampled and evaluated.
  Any dataset that takes longer to run will return a score of `None`.

For example, the following command will run the SDV benchmark on all the given demo datasets
using `dask` and a timeout of 60 seconds:

```python
scores = run_benchmark(
    datasets=['DCG_v1', 'trains_v1', 'UTube_v1'],
    distributed=True,
    timeout=60
)
```

And the result will be a DataFrame containing a table with the columns `dataset`, `score`:

| dataset | score |
|:-------:|:-----:|
| DCG_v1  | -14.49341665631863 |
| trains_v1  | -30.26840342069557 |
| UTube_v1  | -8.57618576332235 |

Additionally, if some dataset has raised an error or has reached the timeout, an `error`
column will be added indicating the details.

### Demo Datasets

The collection of datasets can be seen using the `sdv.demo.get_demo_demos`,
which returns a table with a description of the dataset properties:

```python3
from sdv.demo import get_available_demos

demos = get_available_demos()
```

The result is a table indicating the name of the dataset and a few properties, such as the
number of tables that compose the dataset and the total number of rows and columns:

| name                  |   tables |    rows |   columns |
|-----------------------|----------|---------|-----------|
| UTube_v1              |        2 |    2735 |        10 |
| SAP_v1                |        4 | 3841029 |        71 |
| NCAA_v1               |        9 |  202305 |       333 |
| airbnb-simplified     |        2 | 5751408 |        22 |
| Atherosclerosis_v1    |        4 |   12781 |       307 |
| rossmann              |        3 | 2035533 |        21 |
| walmart               |        4 |  544869 |        24 |
| AustralianFootball_v1 |        4 |  139179 |       193 |
| Pyrimidine_v1         |        2 |     296 |        38 |
| world_v1              |        3 |    5302 |        39 |
| Accidents_v1          |        3 | 1463093 |        87 |
| trains_v1             |        2 |      83 |        15 |
| legalActs_v1          |        5 | 1754397 |        50 |
| DCG_v1                |        2 |    8258 |         9 |
| imdb_ijs_v1           |        7 | 5647694 |        50 |
| SalesDB_v1            |        4 | 6735507 |        35 |
| MuskSmall_v1          |        2 |     568 |       173 |
| KRK_v1                |        1 |    1000 |         9 |
| Chess_v1              |        2 |    2052 |        57 |
| Telstra_v1            |        5 |  148021 |        23 |
| mutagenesis_v1        |        3 |   10324 |        26 |
| PremierLeague_v1      |        4 |   11308 |       250 |
| census                |        1 |   32561 |        15 |
| FNHK_v1               |        3 | 2113275 |        43 |
| imdb_MovieLens_v1     |        7 | 1249411 |        58 |
| financial_v1          |        8 | 1079680 |        84 |
| ftp_v1                |        2 |   96491 |        13 |
| Triazine_v1           |        2 |    1302 |        35 |
