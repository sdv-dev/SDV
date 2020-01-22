# History

## 0.3.1 - 2019-01-22

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
