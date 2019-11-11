# History

## 0.2.0 - 2019-10-11

### New Features

* compatibility with rdt issue 72 - [Issue #120](https://github.com/HDI-Project/SDV/issues/120) by @csala @JDTheRipperPC

### General Improvements

* Error docstring sampler.__fill_text_columns - [Issue #144](https://github.com/HDI-Project/SDV/issues/114) by @JDTheRipperPC
* Reach 90% coverage - [Issue #112](https://github.com/HDI-Project/SDV/issues/112) by @JDTheRipperPC
* Review unittests - [Issue #111](https://github.com/HDI-Project/SDV/issues/111) by @JDTheRipperPC

### Bugs Fixed

* Time required for sample_all function? - [Issue #118](https://github.com/HDI-Project/SDV/issues/118) by @csala @JDTheRipperPC

## 0.1.2 - 2019-09-18

### New Features

* Add option to model the amount of child rows - Issue [93](https://github.com/HDI-Project/SDV/issues/93) by @ManuelAlvarezC

### General Improvements

* Add Evaluation Metrics - Issue [52](https://github.com/HDI-Project/SDV/issues/52) by @ManuelAlvarezC

* Ensure unicity on primary keys on different calls - Issue [63](https://github.com/HDI-Project/SDV/issues/63) by @ManuelAlvarezC

### Bugs fixed

* executing readme: 'not supported between instances of 'int' and 'NoneType' - Issue [104](https://github.com/HDI-Project/SDV/issues/104) by @csala

## 0.1.1 - Anonymization of data

* Add warnings when trying to model an unsupported dataset structure. [GH#73](https://github.com/HDI-Project/SDV/issues/73)
* Add option to anonymize data. [GH#51](https://github.com/HDI-Project/SDV/issues/51)
* Add support for modeling data with different distributions, when using `GaussianMultivariate` model. [GH#68](https://github.com/HDI-Project/SDV/issues/68)
* Add support for `VineCopulas` as a model. [GH#71](https://github.com/HDI-Project/SDV/issues/71)
* Improve `GaussianMultivariate` parameter sampling, avoiding warnings and unvalid parameters. [GH#58](https://github.com/HDI-Project/SDV/issues/58)
* Fix issue that caused that sampled categorical values sometimes got numerical values mixed. [GH#81](https://github.com/HDI-Project/SDV/issues/81)
* Improve the validation of extensions. [GH#69](https://github.com/HDI-Project/SDV/issues/69)
* Update examples. [GH#61](https://github.com/HDI-Project/SDV/issues/61)
* Replaced `Table` class with a `NamedTuple`. [GH#92](https://github.com/HDI-Project/SDV/issues/92)
* Fix inconsistent dependencies and add upper bound to dependencies. [GH#96](https://github.com/HDI-Project/SDV/issues/96)
* Fix error when merging extension in `Modeler.CPA` when running examples. [GH#86](https://github.com/HDI-Project/SDV/issues/86)

## 0.1.0 - First Release

* First release on PyPI.
