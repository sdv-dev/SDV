<div align="center">
<br/>
<p align="center">
    <i>This repository is part of <a href="https://sdv.dev">The Synthetic Data Vault Project</a>, a project from <a href="https://datacebo.com">DataCebo</a>.</i>
</p>

[![Dev Status](https://img.shields.io/badge/Dev%20Status-5%20--%20Production%2fStable-green)](https://pypi.org/search/?c=Development+Status+%3A%3A+5+-+Production%2FStable)
[![PyPi Shield](https://img.shields.io/pypi/v/SDV.svg)](https://pypi.python.org/pypi/SDV)
[![Unit Tests](https://github.com/sdv-dev/SDV/actions/workflows/unit.yml/badge.svg?branch=master)](https://github.com/sdv-dev/SDV/actions/workflows/unit.yml?query=branch%3Amaster)
[![Integration Tests](https://github.com/sdv-dev/SDV/actions/workflows/integration.yml/badge.svg?branch=master)](https://github.com/sdv-dev/SDV/actions/workflows/integration.yml?query=branch%3Amaster)
[![Coverage Status](https://codecov.io/gh/sdv-dev/SDV/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/SDV)
[![Downloads](https://static.pepy.tech/personalized-badge/sdv?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/sdv)
[![Colab](https://img.shields.io/badge/Tutorials-Try%20now!-orange?logo=googlecolab)](https://docs.sdv.dev/sdv/demos)
[![Slack](https://img.shields.io/badge/Slack-Join%20now!-36C5F0?logo=slack)](https://bit.ly/sdv-slack-invite)

<div align="left">
<br/>
<p align="center">
<a href="https://github.com/sdv-dev/SDV">
<img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/SDV-logo.png"></img>
</a>
</p>
</div>

</div>

# Overview

The **Synthetic Data Vault** (SDV) is a Python library designed to be your one-stop shop for
creating tabular synthetic data. The SDV uses a variety of machine learning algorithms to learn
patterns from your real data and emulate them in synthetic data.

## Features
:brain: **Create synthetic data using machine learning.** The SDV offers multiple models, ranging
from classical statistical methods (GaussianCopula) to deep learning methods (CTGAN). Generate
data for single tables, multiple connected tables or sequential tables.

:bar_chart: **Evaluate and visualize data.** Compare the synthetic data to the real data against a
variety of measures. Diagnose problems and generate a quality report to get more insights.

:arrows_counterclockwise: **Preprocess, anonymize and define constraints.** Control data
processing to improve the quality of synthetic data, choose from different types of anonymization
and define business rules in the form of logical constraints.

| Important Links                               |                                                                                                     |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------|
| [![][Colab Logo] **Tutorials**][Tutorials]    | Get some hands-on experience with the SDV. Launch the tutorial notebooks and run the code yourself. |
| :book: **[Docs]**                             | Learn how to use the SDV library with user guides and API references.                               |
| :orange_book: **[Blog]**                      | Get more insights about using the SDV, deploying models and our synthetic data community.          |
| [![][Slack Logo] **Community**][Community]    | Join our Slack workspace for announcements and discussions.                                         |
| :computer: **[Website]**                      | Check out the SDV website for more information about the project.                                   |

[Website]: https://sdv.dev
[Blog]: https://datacebo.com/blog
[Docs]: https://bit.ly/sdv-docs
[Repository]: https://github.com/sdv-dev/SDV
[License]: https://github.com/sdv-dev/SDV/blob/master/LICENSE
[Development Status]: https://pypi.org/search/?c=Development+Status+%3A%3A+5+-+Production%2FStable
[Slack Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/slack.png
[Community]: https://bit.ly/sdv-slack-invite
[Colab Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/google_colab.png
[Tutorials]: https://docs.sdv.dev/sdv/demos

# Install
The SDV is publicly available under the [Business Source License](https://github.com/sdv-dev/SDV/blob/master/LICENSE).
Install SDV using pip or conda. We recommend using a virtual environment to avoid conflicts with
other software on your device.

```bash
pip install sdv
```

```bash
conda install -c pytorch -c conda-forge sdv
```

# Getting Started
Load a demo dataset to get started. This dataset is a single table describing guests staying at a
fictional hotel.

```python
from sdv.datasets.demo import download_demo

real_data, metadata = download_demo(
    modality='single_table',
    dataset_name='fake_hotel_guests')
```

![Single Table Metadata Example](https://github.com/sdv-dev/SDV/blob/master/docs/images/Single-Table-Metadata-Example.png)

The demo also includes **metadata**, a description of the dataset, including the data types in each
column and the primary key (`guest_email`).

## Synthesizing Data
Next, we can create an **SDV synthesizer**,  an object that you can use to create synthetic data.
It learns patterns from the real data and replicates them to generate synthetic data. Let's use
the `FAST_ML` preset synthesizer, which is optimized for performance.

```python
from sdv.lite import SingleTablePreset

synthesizer = SingleTablePreset(metadata, name='FAST_ML')
synthesizer.fit(data=real_data)
```

And now the synthesizer is ready to create synthetic data!

```python
synthetic_data = synthesizer.sample(num_rows=500)
```

The synthetic data will have the following properties:
- **Sensitive columns are fully anonymized.** The email, billing address and credit card number
columns contain new data so you don't expose the real values.
- **Other columns follow statistical patterns.** For example, the proportion of room types, the
distribution of check in dates and the correlations between room rate and room type are preserved.
- **Keys and other relationships are intact.** The primary key (guest email) is unique for each row.
If you have multiple tables, the connection between a primary and foreign keys makes sense.

## Evaluating Synthetic Data
The SDV library allows you to evaluate the synthetic data by comparing it to the real data. Get
started by generating a quality report.

```python
from sdv.evaluation.single_table import evaluate_quality

quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata)
```

```
Creating report: 100%|██████████| 4/4 [00:00<00:00, 19.30it/s]
Overall Quality Score: 89.12%
Properties:
Column Shapes: 90.27%
Column Pair Trends: 87.97%
```

This object computes an overall quality score on a scale of 0 to 100% (100 being the best) as well
as detailed breakdowns. For more insights, you can also visualize the synthetic vs. real data.

```python
from sdv.evaluation.single_table import get_column_plot

fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    column_name='amenities_fee',
    metadata=metadata
)
    
fig.show()
```

![Real vs. Synthetic Data](https://github.com/sdv-dev/SDV/blob/master/docs/images/Real-vs-Synthetic-Evaluation.png)

# What's Next?
Using the SDV library, you can synthesize single table, multi table and sequential data. You can
also customize the full synthetic data workflow, including preprocessing, anonymization and adding
constraints.

To learn more, visit the [SDV Demo page](https://docs.sdv.dev/sdv/demos).

# Credits
Thank you to our team of contributors who have built and maintained the SDV ecosystem over the
years!

[View Contributors](https://github.com/sdv-dev/SDV/graphs/contributors)

## Citation
If you use SDV for your research, please cite the following paper:

*Neha Patki, Roy Wedge, Kalyan Veeramachaneni*. [The Synthetic Data Vault](https://dai.lids.mit.edu/wp-content/uploads/2018/03/SDV.pdf). [IEEE DSAA 2016](https://ieeexplore.ieee.org/document/7796926).

```
@inproceedings{
    SDV,
    title={The Synthetic data vault},
    author={Patki, Neha and Wedge, Roy and Veeramachaneni, Kalyan},
    booktitle={IEEE International Conference on Data Science and Advanced Analytics (DSAA)},
    year={2016},
    pages={399-410},
    doi={10.1109/DSAA.2016.49},
    month={Oct}
}
```

---


<div align="center">
  <a href="https://datacebo.com"><picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/sdv-dev/SDV/blob/master/docs/images/datacebo-logo-dark-mode.png">
      <img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/datacebo-logo.png"></img>
  </picture></a>
</div>
<br/>
<br/>

[The Synthetic Data Vault Project](https://sdv.dev) was first created at MIT's [Data to AI Lab](
https://dai.lids.mit.edu/) in 2016. After 4 years of research and traction with enterprise, we
created [DataCebo](https://datacebo.com) in 2020 with the goal of growing the project.
Today, DataCebo is the proud developer of SDV, the largest ecosystem for
synthetic data generation & evaluation. It is home to multiple libraries that support synthetic
data, including:

* 🔄 Data discovery & transformation. Reverse the transforms to reproduce realistic data.
* 🧠 Multiple machine learning models -- ranging from Copulas to Deep Learning -- to create tabular,
  multi table and time series data.
* 📊 Measuring quality and privacy of synthetic data, and comparing different synthetic data
  generation models.

[Get started using the SDV package](https://bit.ly/sdv-docs) -- a fully
integrated solution and your one-stop shop for synthetic data. Or, use the standalone libraries
for specific needs.
