# Release Notes

## 0.14.0 - 2022-03-21

This release updates the sampling API and splits the existing functionality into three methods - `sample`, `sample_conditions`,
and `sample_remaining_columns`. We also add support for sampling in batches, displaying a progress bar when sampling with more than one batch,
sampling deterministically, and writing the sampled results to an output file. Finally, we include fixes for sampling with conditions
and updates to the documentation.

### Bugs Fixed
* Fix write to file in sampling - Issue [#732](https://github.com/sdv-dev/SDV/issues/732) by @katxiao
* Conditional sampling doesn't work if the model has a CustomConstraint - Issue [#696](https://github.com/sdv-dev/SDV/issues/696) by @katxiao

### New Features
* Updates to GaussianCopula conditional sampling methods - Issue [#729](https://github.com/sdv-dev/SDV/issues/729) by @katxiao
* Update conditional sampling errors - Issue [#730](https://github.com/sdv-dev/SDV/issues/730) by @katxiao
* Enable Batch Sampling + Progress Bar - Issue [#693](https://github.com/sdv-dev/SDV/issues/693) by @katxiao
* Create sample_remaining_columns() method - Issue [#692](https://github.com/sdv-dev/SDV/issues/692) by @katxiao
* Create sample_conditions() method - Issue [#691](https://github.com/sdv-dev/SDV/issues/691) by @katxiao
* Improve sample() method - Issue [#690](https://github.com/sdv-dev/SDV/issues/690) by @katxiao
* Create Condition object - Issue [#689](https://github.com/sdv-dev/SDV/issues/689) by @katxiao
* Is it possible to generate data with new set of primary keys? - Issue [#686](https://github.com/sdv-dev/SDV/issues/686) by @katxiao
* No way to fix the random seed? - Issue [#157](https://github.com/sdv-dev/SDV/issues/157) by @katxiao
* Can you set a random state for the sdv.tabular.ctgan.CTGAN.sample method? - Issue [#515](https://github.com/sdv-dev/SDV/issues/515) by @katxiao
* generating different synthetic data while training the model multiple times. - Issue [#299](https://github.com/sdv-dev/SDV/issues/299) by @katxiao

### Documentation Changes
* Typo in the document documentation - Issue [#680](https://github.com/sdv-dev/SDV/issues/680) by @katxiao

## 0.13.1 - 2021-12-22

This release adds support for passing tabular constraints to the HMA1 model, and adds more explicit error handling for
metric evaluation. It also includes a fix for using categorical columns in the PAR model and documentation updates
for metadata and HMA1.

### Bugs Fixed

* Categorical column after sequence_index column - Issue [#314](https://github.com/sdv-dev/SDV/issues/314) by @fealho

### New Features

* Support passing tabular constraints to the HMA1 model - Issue [#296](https://github.com/sdv-dev/SDV/issues/296) by @katxiao
* Metric evaluation error handling metrics - Issue [#638](https://github.com/sdv-dev/SDV/issues/638) by @katxiao

### Documentation Changes

* Make true/false values lowercase in Metadata Schema specification - Issue [#664](https://github.com/sdv-dev/SDV/issues/664) by @katxiao
* Update docstrings for hma1 methods - Issue [#642](https://github.com/sdv-dev/SDV/issues/642) by @katxiao

## 0.13.0 - 2021-11-22

This release makes multiple improvements to different `Constraint` classes. The `Unique` constraint can now
handle columns with the name `index` and no longer crashes on subsets of the original data. The `Between`
constraint can now handle columns with nulls properly. The memory of all constraints was also improved.

Various other features and fixes were added. Conditional sampling no longer crashes when the `num_rows` argument
is not provided. Multiple localizations can now be used for PII fields. Scaffolding for integration tests was added
and the workflows now run `pip check`.

Additionally, this release adds support for Python 3.9!

### Bugs Fixed

* Gaussian Copula – Memory Issue in Release 0.10.0 - Issue [#459](https://github.com/sdv-dev/SDV/issues/459) by @xamm
* Applying Unique Constraint errors when calling model.fit() on a subset of data - Issue [#610](https://github.com/sdv-dev/SDV/issues/610) by @xamm
* Calling sampling with conditions and without num_rows crashes - Issue [#614](https://github.com/sdv-dev/SDV/issues/614) by @xamm
* Metadata.visualize with path parameter throws AttributeError - Issue [#634](https://github.com/sdv-dev/SDV/issues/634) by @xamm
* The Unique constraint crashes when the data contains a column called index - Issue [#616](https://github.com/sdv-dev/SDV/issues/616) by @xamm
* The Unique constraint cannot handle non-default index - Issue [#617](https://github.com/sdv-dev/SDV/issues/617) by @xamm
* ConstraintsNotMetError when applying Between constraint on datetime columns containing null values - Issue [#632](https://github.com/sdv-dev/SDV/issues/632) by @katxiao

### New Features

* Adds Multi localisations feature for PII fields defined in #308 - PR [#609](https://github.com/sdv-dev/SDV/pull/609) by @xamm

### Housekeeping Tasks

* Support latest version of Faker - Issue [#621](https://github.com/sdv-dev/SDV/issues/621) by @katxiao
* Add scaffolding for Metadata integration tests - Issue [#624](https://github.com/sdv-dev/SDV/issues/624) by @katxiao
* Add support for Python 3.9 - Issue [#631](https://github.com/sdv-dev/SDV/issues/631) by @amontanez24

### Internal Improvements

* Add pip check to CI workflows - Issue [#626](https://github.com/sdv-dev/SDV/issues/626) by @pvk-developer

### Documentation Changes

* Anonymizing PII in single table tutorials states address field as e-mail type - Issue [#604](https://github.com/sdv-dev/SDV/issues/604) by @xamm

Special thanks to @xamm, @katxiao, @pvk-developer and @amontanez24 for all the work that made this release possible!

## 0.12.1 - 2021-10-12

This release fixes bugs in constraints, metadata behavior, and SDV documentation. Specifically, we added
proper handling of data containing null values for constraints and timeseries data, and updated the
default metadata detection behavior.

### Bugs Fixed

* ValueError: The parameter loc has invalid values - Issue [#353](https://github.com/sdv-dev/SDV/issues/353) by @fealho
* Gaussian Copula is generating different data with metadata and without metadata - Issue [#576](https://github.com/sdv-dev/SDV/issues/576) by @katxiao
* Make pomegranate an optional dependency - Issue [#567](https://github.com/sdv-dev/SDV/issues/567) by @katxiao
* Small wording change for Question Issue Template - Issue [#571](https://github.com/sdv-dev/SDV/issues/571) by @katxiao
* ConstraintsNotMetError when using GreaterThan constraint with datetime - Issue [#590](https://github.com/sdv-dev/SDV/issues/590) by @katxiao
* GreaterThan constraint crashing with NaN values - Issue [#592](https://github.com/sdv-dev/SDV/issues/592) by @katxiao
* Null values in GreaterThan constraint raises error - Issue [#589](https://github.com/sdv-dev/SDV/issues/589) by @katxiao
* ColumnFormula raises ConstraintsNotMetError when checking NaN values - Issue [#593](https://github.com/sdv-dev/SDV/issues/593) by @katxiao
* GreaterThan constraint raises TypeError when using datetime - Issue [#596](https://github.com/sdv-dev/SDV/issues/596) by @katxiao
* Fix repository language - Issue [#464](https://github.com/sdv-dev/SDV/issues/464) by @fealho
* Update __init__.py - Issue [#578](https://github.com/sdv-dev/SDV/issues/578) by @dyuliu
* IndexingError: Unalignable boolean - Issue [#446](https://github.com/sdv-dev/SDV/issues/446) by @fealho

## 0.12.0 - 2021-08-17

This release focuses on improving and expanding upon the existing constraints. More specifically, the users can now
(1) specify multiple columns in `Positive` and `Negative` constraints, (2) use the new `Unique`constraint and
(3) use datetime data with the `Between` constraint. Additionaly, error messages have been added and updated
to provide more useful feedback to the user.

Besides the added features, several bugs regarding the `UniqueCombinations` and `ColumnFormula` constraints have been fixed,
and an error in the metadata.json for the `student_placements` dataset was corrected. The release also added documentation
for the `fit_columns_model` which affects the majority of the available constraints.

### New Features

* Change default fit_columns_model to False - Issue [#550](https://github.com/sdv-dev/SDV/issues/550) by @katxiao
* Support multi-column specification for positive and negative constraint - Issue [#545](https://github.com/sdv-dev/SDV/issues/545) by @sarahmish
* Raise error when multiple constraints can't be enforced - Issue [#541](https://github.com/sdv-dev/SDV/issues/541) by @amontanez24
* Create Unique Constraint - Issue [#532](https://github.com/sdv-dev/SDV/issues/532) by @amontanez24
* Passing invalid conditions when using constraints produces unreadable errors - Issue [#511](https://github.com/sdv-dev/SDV/issues/511) by @katxiao
* Improve error message for ColumnFormula constraint when constraint column used in formula - Issue [#508](https://github.com/sdv-dev/SDV/issues/508) by @katxiao
* Add datetime functionality to Between constraint - Issue [#504](https://github.com/sdv-dev/SDV/issues/504) by @katxiao

### Bugs Fixed

* UniqueCombinations constraint with handling_strategy = 'transform' yields synthetic data with nan values - Issue [#521](https://github.com/sdv-dev/SDV/issues/521) by @katxiao and @csala
* UniqueCombinations constraint outputting wrong data type - Issue [#510](https://github.com/sdv-dev/SDV/issues/510) by @katxiao and @csala
* UniqueCombinations constraint on only one column gets stuck in an infinite loop - Issue [#509](https://github.com/sdv-dev/SDV/issues/509) by @katxiao
* Conditioning on a non-constraint column using the ColumnFormula constraint - Issue [#507](https://github.com/sdv-dev/SDV/issues/507) by @katxiao
* Conditioning on the constraint column of the ColumnFormula constraint - Issue [#506](https://github.com/sdv-dev/SDV/issues/506) by @katxiao
* Update metadata.json for duration of student_placements dataset - Issue [#503](https://github.com/sdv-dev/SDV/issues/503) by @amontanez24
* Unit test for HMA1 when working with a single child row per parent row - Issue [#497](https://github.com/sdv-dev/SDV/issues/497) by @pvk-developer
* UniqueCombinations constraint for more than 2 columns - Issue [#494](https://github.com/sdv-dev/SDV/issues/494) by @katxiao and @csala

### Documentation Changes

* Add explanation of fit_columns_model to API docs - Issue [#517](https://github.com/sdv-dev/SDV/issues/517) by @katxiao

## 0.11.0 - 2021-07-12

This release primarily addresses bugs and feature requests related to using constraints for the single-table models.
Users can now enforce scalar comparison with the existing `GreaterThan` constraint and apply 5 new constraints: `OneHotEncoding`, `Positive`, `Negative`, `Between` and `Rounding`.
Additionally, the SDV will now auto-apply constraints for rounding numerical values, and for keeping the data within the observed bounds.
All related user guides are updated with the new functionality.

### New Features

* Add OneHotEncoding Constraint - Issue [#303](https://github.com/sdv-dev/SDV/issues/303) by @fealho
* GreaterThan Constraint should apply to scalars - Issue [#410](https://github.com/sdv-dev/SDV/issues/410) by @amontanez24
* Improve GreaterThan constraint - Issue [#368](https://github.com/sdv-dev/SDV/issues/368) by @amontanez24
* Add Non-negative and Positive constraints across multiple columns- Issue [#409](https://github.com/sdv-dev/SDV/issues/409) by @amontanez24
* Add Between values constraint - Issue [#367](https://github.com/sdv-dev/SDV/issues/367) by @fealho
* Ensure values fall within the specified range - Issue [#423](https://github.com/sdv-dev/SDV/issues/423) by @amontanez24
* Add Rounding constraint - Issue [#482](https://github.com/sdv-dev/SDV/issues/482) by @katxiao
* Add rounding and min/max arguments that are passed down to the NumericalTransformer - Issue [#491](https://github.com/sdv-dev/SDV/issues/491) by @amontanez24

### Bugs Fixed

* GreaterThan constraint between Date columns rasises TypeError - Issue [#421](https://github.com/sdv-dev/SDV/issues/421) by @amontanez24
* GreaterThan constraint's transform strategy fails on columns that are not float - Issue [#448](https://github.com/sdv-dev/SDV/issues/448) by @amontanez24
* AttributeError on UniqueCombinations constraint with non-strings - Issue [#196](https://github.com/sdv-dev/SDV/issues/196) by @katxiao
* Use reject sampling to sample missing columns for constraints - Issue [#435](https://github.com/sdv-dev/SDV/issues/435) by @amontanez24

### Documentation Changes

* Ensure privacy metrics are available in the API docs - Issue [#458](https://github.com/sdv-dev/SDV/issues/458) by @fealho
* Ensure forumla constraint is called ColumnFormula everywhere in the docs - Issue [#449](https://github.com/sdv-dev/SDV/issues/449) by @fealho

## 0.10.1 - 2021-06-10

This release changes the way we sample conditions to not only group by the conditions passed by the user, but also by the transformed conditions that result from them.

### Issues resolved

* Conditionally sampling on variable in constraint should have variety for other variables - Issue [#440](https://github.com/sdv-dev/SDV/issues/440) by @amontanez24

## 0.10.0 - 2021-05-21

This release improves the constraint functionality by allowing constraints and conditions
at the same time. Additional changes were made to update tutorials.

### Issues resolved

* Not able to use constraints and conditions in the same time - Issue [#379](https://github.com/sdv-dev/SDV/issues/379)
by @amontanez24
* Update benchmarking user guide for reading private datasets - Issue [#427](https://github.com/sdv-dev/SDV/issues/427)
by @katxiao

## 0.9.1 - 2021-04-29

This release broadens the constraint functionality by allowing for the `ColumnFormula`
constraint to take lambda functions and returned functions as an input for its formula.

It also improves conditional sampling by ensuring that any `id` fields generated by the
model remain unique throughout the sampled data.

The `CTGAN` model was improved by adjusting a default parameter to be more mathematically
correct.

Additional changes were made to improve tutorials as well as fix fragile tests.

### Issues resolved

* Tutorials test sometimes fails - Issue [#355](https://github.com/sdv-dev/SDV/issues/355)
by @fealho
* Duplicate IDs when using reject-sampling - Issue [#331](https://github.com/sdv-dev/SDV/issues/331)
by @amontanez24 and @csala
* discriminator_decay should be initialized at 1e-6 but it's 0 - Issue [#401](https://github.com/sdv-dev/SDV/issues/401) by @fealho and @YoucefZemmouri
* Tutorial typo - Issue [#380](https://github.com/sdv-dev/SDV/issues/380) by @fealho
* Request for sdv.constraint.ColumnFormula for a wider range of function - Issue [#373](https://github.com/sdv-dev/SDV/issues/373) by @amontanez24 and @JetfiRex

## 0.9.0 - 2021-03-31

This release brings new privacy metrics to the evaluate framework which help to determine
if the real data could be obtained or deduced from the synthetic samples.
Additionally, now there is a normalized score for the metrics, which stays between `0` and `1`.

There are improvements that reduce the usage of memory ram when sampling new data. Also there
is a new parameter to control the reject sampling crash, `graceful_reject_sampling`, which if
set to `True` and if it's not possible to generate all the requested rows, it will just issue a
warning and return whatever it was able to generate.

The `Metadata` object can now be visualized using different combinations of `names` and `details`,
which can be set to `True` or `False` in order to display only the table names with details or
without. There is also an improvement on the `validation`, which now will display all the errors
found at the end of the validation instead of only the first one.

This version also exposes all the hyperparameters of the models `CTGAN` and `TVAE` to allow a more
advanced usage. There is also a fix for the `TVAE` model on small datasets and it's performance
with `NaN` values has been improved. There is a fix for when using
`UniqueCombinationConstraint` with the `transform` strategy.

### Issues resolved

* Memory Usage Gaussian Copula Trained Model consuming high memory when generating synthetic data - Issue [#304](https://github.com/sdv-dev/SDV/issues/304) by @pvk-developer and @AnupamaGangadhar
* Add option to visualize metadata with only table names - Issue [#347](https://github.com/sdv-dev/SDV/issues/347) by @csala
* Add sample parameter to control reject sampling crash - Issue [#343](https://github.com/sdv-dev/SDV/issues/343) by @fealho
* Verbose metadata validation - Issue [#348](https://github.com/sdv-dev/SDV/issues/348) by @csala
* Missing the introduction of custom specification for hyperparameters in the TVAE model - Issue [#344](https://github.com/sdv-dev/SDV/issues/343) by @imkhoa99 and @pvk-developer

## 0.8.0 - 2021-02-24

This version adds conditional sampling for tabular models by combining a reject-sampling
strategy with the native conditional sampling capabilities from the gaussian copulas.

It also introduces several upgrades on the HMA1 algorithm that improve data quality and
robustness in the multi-table scenarios by making changes in how the parameters of the child
tables are aggregated on the parent tables, including a complete rework of how the correlation
matrices are modeled and rebuild after sampling.

### Issues resolved

* Fix probabilities contain NaN error - Issue [#326](https://github.com/sdv-dev/SDV/issues/326) by @csala
* Conditional Sampling for tabular models - Issue [#316](https://github.com/sdv-dev/SDV/issues/316) by @fealho and @csala
* HMA1: LinAlgError: SVD did not converge - Issue [#240](https://github.com/sdv-dev/SDV/issues/240) by @csala

## 0.7.0 - 2021-01-27

This release introduces a few changes in the HMA1 relational algorithm to decrease modeling
and sampling times, while also ensuring that correlations are properly kept across tables
and also adding support for some relational schemas that were not supported before.

A few changes in constraints and tabular models also ensure that situations that produced
errors before now work without errors.

### Issues resolved

* Fix unique key generation - Issue [#306](https://github.com/sdv-dev/SDV/issues/306) by @fealho
* Ensure tables that contain nothing but ids can be modeled - Issue [#302](https://github.com/sdv-dev/SDV/issues/302) by @csala
* Metadata visualization improvements - Issue [#301](https://github.com/sdv-dev/SDV/issues/301) by @csala
* Multi-parent re-model and re-sample issue - Issue [#298](https://github.com/sdv-dev/SDV/issues/298) by @csala
* Support datetimes in GreaterThan constraint - Issue [#266](https://github.com/sdv-dev/SDV/issues/266) by @rollervan
* Support for multiple foreign keys in one table - Issue [#185](https://github.com/sdv-dev/SDV/issues/185) by @csala

## 0.6.1 - 2020-12-31

SDMetrics version is updated to include the new Time Series metrics, which have also
been added to the API Reference and User Guides documentation. Additionally,
a few code has been refactored to reduce external dependencies and a few minor bugs
related to single table constraints have been fixed

### Issues resolved

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

### Issues resolved

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
