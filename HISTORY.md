# Release Notes

## 0.6.1 - 2020-12-31

SDMetrics version is updated to include the new Time Series metrics, which have also
been added to the API Reference and User Guides documentation. Additionally,
a few code has been refactored to reduce external dependencies and a few minor bugs
related to single table constraints have been fixed

## Issues resolved:

* Add timeseries metrics and user guides - [Issue #289](https://github.com/sdv-dev/SDV/issues/289) by @csala
* Add functions to generate regex ids - [Issue #288](https://github.com/sdv-dev/SDV/issues/288) by @csala
* Saving a fitted tabular model with UniqueCombinations constraint raises PicklingError -
  [Issue #286](https://github.com/sdv-dev/SDV/issues/288) by @csala
* Constraints: `handling_strategy='reject_sampling'` causes `'ZeroDivisionError: division by zero'` -
  [Issue #285](https://github.com/sdv-dev/SDV/issues/285) by @csala

## 0.6.0 - 2020-12-22

This release updates to the latest CTGAN, RDT and SDMetrics libraries to introduce a
new TVAE model, multiple new metrics for single table and multi table, and fixes
issues in the re-creation of tabular models from a metadata dict.

## Issues resolved:

* Upgrade to SDMetrics v0.1.0 and add `sdv.metrics` module - [Issue #281](https://github.com/sdv-dev/SDV/issues/281) by @csala
* Upgrade to CTGAN 0.3.0 and add TVAE model - [Issue #278](https://github.com/sdv-dev/SDV/issues/278) by @fealho
* Add `dtype_transformers` to `Table.from_dict` - [Issue #276](https://github.com/sdv-dev/SDV/issues/276) by @csala
* Fix Metadata `from_dict` behavior - [Issue #275](https://github.com/sdv-dev/SDV/issues/275) by @csala

## 0.5.0 - 2020-11-25

This version updates the dependencies and makes a few internal changes in order
to ensure that SDV works properly on Windows Systems, making this the first
release to be officially supported on Windows.

Apart from this, some more internal changes have been made to solve a few minor
issues from the older versions while also improving the processing speed when
processing relational datasets with the default parameters.

### API breaking changes

* The `distribution` argument of the `GaussianCopula` has been renamed to `field_distributions`.
* The `HMA1` and `SDV` classes now use the `categorical_fuzzy` transformer by default instead of
  the `one_hot_encoding` one.

### Issues resolved

* GaussianCopula: rename `distribution` argument to `field_distributions` - [Issue #237](https://github.com/sdv-dev/SDV/issues/237) by @csala
* GaussianCopula: Improve error message if an invalid distribution name is passed - [Issue #220](https://github.com/sdv-dev/SDV/issues/220) by csala
* Import urllib.request explicitly - [Issue #227](https://github.com/sdv-dev/SDV/issues/227) by @csala
* TypeError: cannot astype a datetimelike from [datetime64[ns]] to [int32] - [Issue #218](https://github.com/sdv-dev/SDV/issues/218) by @csala
* Change default categorical transformer to `categorical_fuzzy` in HMA1 - [Issue #214](https://github.com/sdv-dev/SDV/issues/214) by @csala
* Integer categoricals being sampled as strings instead of integer values - [Issue #194](https://github.com/sdv-dev/SDV/issues/194) by @csala

## 0.4.5 - 2020-10-17

In this version a new family of models for Synthetic Time Series Generation is introduced
under the `sdv.timeseries` sub-package. The new family of models now includes a new class
called `PAR`, which implements a *Probabilistic AutoRegressive* model.

This version also adds support for composite primary keys and regex based generation of id
fields in tabular models and drops Python 3.5 support.

### Issues resolved

* Drop python 3.5 support - [Issue #204](https://github.com/sdv-dev/SDV/issues/204) by @csala
* Support composite primary keys in tabular models - [Issue #207](https://github.com/sdv-dev/SDV/issues/207) by @csala
* Add the option to generate string `id` fields based on regex on tabular models - [Issue #208](https://github.com/sdv-dev/SDV/issues/208) by @csala
* Synthetic Time Series - [Issue #142](https://github.com/sdv-dev/SDV/issues/142) by @csala


## 0.4.4 - 2020-10-06

This version adds a new tabular model based on combining the CTGAN model with the reversible
transformation applied in the GaussianCopula model that converts random variables with
arbitrary distributions to new random variables with standard normal distribution.

The reversible transformation is handled by the GaussianCopulaTransformer recently added to RDT.

### Issues resolved

* Add CopulaGAN Model - [Issue #202](https://github.com/sdv-dev/SDV/issues/202) by @csala

## 0.4.3 - 2020-09-28

This release moves the models and algorithms related to generation of synthetic
relational data to a new `sdv.relational` subpackage (Issue #198)

As part of the change, also the old `sdv.models` have been removed and now
relational model is based on the recently introduced `sdv.tabular` models.

## 0.4.2 - 2020-09-19

In this release the `sdv.evaluation` module has been reworked to include 4 different
metrics and in all cases return a normalized score between 0 and 1.

Included metrics are:
- `cstest`
- `kstest`
- `logistic_detection`
- `svc_detection`

## 0.4.1 - 2020-09-07

This release fixes a couple of minor issues and introduces an important rework of the
User Guides section of the documentation.

### Issues fixed

* Error Message: "make sure the Graphviz executables are on your systems' PATH" - [Issue #182](https://github.com/sdv-dev/SDV/issues/182) by @csala
* Anonymization mappings leak - [Issue #187](https://github.com/sdv-dev/SDV/issues/187) by @csala

## 0.4.0 - 2020-08-08

In this release SDV gets new documentation, new tutorials, improvements to the Tabular API
and broader python and dependency support.

Complete list of changes:

* New Documentation site based on the `pydata-sphinx-theme`.
* New User Guides and Notebook tutorials.
* New Developer Guides section within the docs with details about the SDV architecture,
  the ecosystem libraries and how to extend and contribute to the project.
* Improved API for the Tabular models with focus on ease of use.
* Support for Python 3.8 and the newest versions of pandas, scipy and scikit-learn.
* New Slack Workspace for development discussions and community support.

## 0.3.6 - 2020-07-23

This release introduces a new concept of `Constraints`, which allow the user to define
special relationships between columns that will not be handled via modeling.

This is done via a new `sdv.constraints` subpackage which defines some well-known pre-defined
constraints, as well as a generic framework that allows the user to customize the constraints
to their needs as much as necessary.

### New Features

* Support for Constraints - [Issue #169](https://github.com/sdv-dev/SDV/issues/169) by @csala


## 0.3.5 - 2020-07-09

This release introduces a new subpackage `sdv.tabular` with models designed specifically
for single table modeling, while still providing all the usual conveniences from SDV, such
as:

* Seamless multi-type support
* Missing data handling
* PII anonymization

Currently implemented models are:

* GaussianCopula: Multivariate distributions modeled using copula functions. This is stronger
  version, with more marginal distributions and options, than the one used to model multi-table
  datasets.
* CTGAN: GAN-based data synthesizer that can generate synthetic tabular data with high fidelity.


## 0.3.4 - 2020-07-04

### New Features

* Support for Multiple Parents - [Issue #162](https://github.com/sdv-dev/SDV/issues/162) by @csala
* Sample by default the same number of rows as in the original table - [Issue #163](https://github.com/sdv-dev/SDV/issues/163) by @csala

### General Improvements

* Add benchmark - [Issue #165](https://github.com/sdv-dev/SDV/issues/165) by @csala

## 0.3.3 - 2020-06-26

### General Improvements

* Use SDMetrics for evaluation - [Issue #159](https://github.com/sdv-dev/SDV/issues/159) by @csala

## 0.3.2 - 2020-02-03

### General Improvements

* Improve metadata visualization - [Issue #151](https://github.com/sdv-dev/SDV/issues/151) by @csala @JDTheRipperPC

## 0.3.1 - 2020-01-22

### New Features

* Add Metadata Validation - [Issue #134](https://github.com/sdv-dev/SDV/issues/134) by @csala @JDTheRipperPC

* Add Metadata Visualization - [Issue #135](https://github.com/sdv-dev/SDV/issues/135) by @JDTheRipperPC

### General Improvements

* Add path to metadata JSON - [Issue #143](https://github.com/sdv-dev/SDV/issues/143) by @JDTheRipperPC

* Use new Copulas and RDT versions - [Issue #147](https://github.com/sdv-dev/SDV/issues/147) by @csala @JDTheRipperPC

## 0.3.0 - 2019-12-23

### New Features

* Create sdv.models subpackage - [Issue #141](https://github.com/sdv-dev/SDV/issues/141) by @JDTheRipperPC

## 0.2.2 - 2019-12-10

### New Features

* Adapt evaluation to the different data types - [Issue #128](https://github.com/sdv-dev/SDV/issues/128) by @csala @JDTheRipperPC

* Extend `load_demo` functionality to load other datasets - [Issue #136](https://github.com/sdv-dev/SDV/issues/136) by @JDTheRipperPC

## 0.2.1 - 2019-11-25

### New Features

* Methods to generate Metadata from DataFrames - [Issue #126](https://github.com/sdv-dev/SDV/issues/126) by @csala @JDTheRipperPC

## 0.2.0 - 2019-10-11

### New Features

* compatibility with rdt issue 72 - [Issue #120](https://github.com/sdv-dev/SDV/issues/120) by @csala @JDTheRipperPC

### General Improvements

* Error docstring sampler.__fill_text_columns - [Issue #144](https://github.com/sdv-dev/SDV/issues/114) by @JDTheRipperPC
* Reach 90% coverage - [Issue #112](https://github.com/sdv-dev/SDV/issues/112) by @JDTheRipperPC
* Review unittests - [Issue #111](https://github.com/sdv-dev/SDV/issues/111) by @JDTheRipperPC

### Bugs Fixed

* Time required for sample_all function? - [Issue #118](https://github.com/sdv-dev/SDV/issues/118) by @csala @JDTheRipperPC

## 0.1.2 - 2019-09-18

### New Features

* Add option to model the amount of child rows - Issue [93](https://github.com/sdv-dev/SDV/issues/93) by @ManuelAlvarezC

### General Improvements

* Add Evaluation Metrics - Issue [52](https://github.com/sdv-dev/SDV/issues/52) by @ManuelAlvarezC

* Ensure unicity on primary keys on different calls - Issue [63](https://github.com/sdv-dev/SDV/issues/63) by @ManuelAlvarezC

### Bugs fixed

* executing readme: 'not supported between instances of 'int' and 'NoneType' - Issue [104](https://github.com/sdv-dev/SDV/issues/104) by @csala

## 0.1.1 - Anonymization of data

* Add warnings when trying to model an unsupported dataset structure. [GH#73](https://github.com/sdv-dev/SDV/issues/73)
* Add option to anonymize data. [GH#51](https://github.com/sdv-dev/SDV/issues/51)
* Add support for modeling data with different distributions, when using `GaussianMultivariate` model. [GH#68](https://github.com/sdv-dev/SDV/issues/68)
* Add support for `VineCopulas` as a model. [GH#71](https://github.com/sdv-dev/SDV/issues/71)
* Improve `GaussianMultivariate` parameter sampling, avoiding warnings and unvalid parameters. [GH#58](https://github.com/sdv-dev/SDV/issues/58)
* Fix issue that caused that sampled categorical values sometimes got numerical values mixed. [GH#81](https://github.com/sdv-dev/SDV/issues/81)
* Improve the validation of extensions. [GH#69](https://github.com/sdv-dev/SDV/issues/69)
* Update examples. [GH#61](https://github.com/sdv-dev/SDV/issues/61)
* Replaced `Table` class with a `NamedTuple`. [GH#92](https://github.com/sdv-dev/SDV/issues/92)
* Fix inconsistent dependencies and add upper bound to dependencies. [GH#96](https://github.com/sdv-dev/SDV/issues/96)
* Fix error when merging extension in `Modeler.CPA` when running examples. [GH#86](https://github.com/sdv-dev/SDV/issues/86)

## 0.1.0 - First Release

* First release on PyPI.
