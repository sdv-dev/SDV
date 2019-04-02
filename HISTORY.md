History
=======

0.1.1 - Anonymization of data
-----------------------------

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

0.1.0 - First Release
---------------------

* First release on PyPI.
