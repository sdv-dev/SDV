# Release Notes

## v1.31.0 - 2025-12-15

### New Features

* Add bucket and credentials parameter to download_demo - Issue [#2747](https://github.com/sdv-dev/SDV/issues/2747) by @amontanez24

## v1.30.0 - 2025-12-05

### New Features

* When using PyTorch, enable GPU usage for MacOS - Issue [#2672](https://github.com/sdv-dev/SDV/issues/2672) by @R-Palazzo

## v1.29.1 - 2025-11-21

### Bugs Fixed

* Warning raised multiple times when metadata does not have the 'datetime_format' key - Issue [#2739](https://github.com/sdv-dev/SDV/issues/2739) by @fealho
* Adding constraints to multi-table synthesizer in multiple steps causes fit to crash - Issue [#2736](https://github.com/sdv-dev/SDV/issues/2736) by @frances-h

## v1.29.0 - 2025-11-14

### New Features

* Prevent users from accidentally overriding the source or README files - Issue [#2686](https://github.com/sdv-dev/SDV/issues/2686) by @fealho
* Ensure that the source/readme files can only be downloaded in `txt` format - Issue [#2685](https://github.com/sdv-dev/SDV/issues/2685) by @fealho
* If a source or readme does not exist for a dataset, show a warning - Issue [#2684](https://github.com/sdv-dev/SDV/issues/2684) by @fealho
* Add get_source and get_readme functions for demo datasets - Issue [#2663](https://github.com/sdv-dev/SDV/issues/2663) by @fealho
* Update the download_demo  and get_available_demos functions - Issue [#2661](https://github.com/sdv-dev/SDV/issues/2661) by @fealho

### Bugs Fixed

* PARSynthesizer crashes if context columns are listed in a different order than the data - Issue [#2719](https://github.com/sdv-dev/SDV/issues/2719) by @sarahmish
* HMASynthesizer - Cap displayed column count in PerformanceAlert Message - Issue [#2716](https://github.com/sdv-dev/SDV/issues/2716) by @fealho
* HMA sampling raises a pandas FutureWarning: Downcasting object dtype arrays - Issue [#2711](https://github.com/sdv-dev/SDV/issues/2711) by @frances-h
* download_demo errors if the dataset_name in the metainfo does not match the dataset name in the folder - Issue [#2691](https://github.com/sdv-dev/SDV/issues/2691) by @fealho
* download_demo should ignore non-csv files in data.zip - Issue [#2690](https://github.com/sdv-dev/SDV/issues/2690) by @fealho
* Improve error message when no csv file are found for a dataset when using download_demo - Issue [#2689](https://github.com/sdv-dev/SDV/issues/2689) by @fealho
* `download_demo` may fail for some `data.zip` files - Issue [#2688](https://github.com/sdv-dev/SDV/issues/2688) by @fealho
* `download_demo` failing when `output_folder_path` is set - Issue [#2687](https://github.com/sdv-dev/SDV/issues/2687) by @fealho

### Internal

* HMASynthesizer - model child tables with norm distribution - Issue [#2665](https://github.com/sdv-dev/SDV/issues/2665) by @rwedge

### Maintenance

* Remove support for Python 3.8 - Issue [#2681](https://github.com/sdv-dev/SDV/issues/2681) by @fealho

## v1.28.0 - 2025-10-17

### New Features

* Unable to validate just 1 table of a multi-table schema - Issue [#2678](https://github.com/sdv-dev/SDV/issues/2678) by @frances-h
* Allow users to validate the DayZ parameters - Issue [#2667](https://github.com/sdv-dev/SDV/issues/2667) by @frances-h
* Allow users to estimate parameters for DayZSynthesizer - Issue [#2666](https://github.com/sdv-dev/SDV/issues/2666) by @R-Palazzo

### Bugs Fixed

* Minimum tests failing - OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed - Issue [#2725](https://github.com/sdv-dev/SDV/issues/2725) by @amontanez24
* [DayZ Parameters] `'missing_values_proportion'` must be zero for any key columns - Issue [#2708](https://github.com/sdv-dev/SDV/issues/2708) by @frances-h
* [DayZ Parameters] Validation results in unexpected errors for some edge cases - Issue [#2703](https://github.com/sdv-dev/SDV/issues/2703) by @fealho
* [DayZ Parameters] `create_parameters` should fall back to default parameters if parameters cannot be detected - Issue [#2702](https://github.com/sdv-dev/SDV/issues/2702) by @fealho
* [DayZ Parameters] DayZ parameter validation does not validate DAYZ_SPEC_VERSION - Issue [#2701](https://github.com/sdv-dev/SDV/issues/2701) by @R-Palazzo
* [DayZParameters] `KeyError` when creating parameters with empty data and metadata - Issue [#2700](https://github.com/sdv-dev/SDV/issues/2700) by @fealho
* Unable to load the DayZSynthesizer after saving it - Issue [#2698](https://github.com/sdv-dev/SDV/issues/2698) by @R-Palazzo
* `DayZSynthesizer.create_parameters` errors in Colab with numeric columns - Issue [#2683](https://github.com/sdv-dev/SDV/issues/2683) by @frances-h
* PARSynthesizer: `FutureWarnings` in `groupby.apply` and `Series.__getitem__` from pandas - Issue [#2682](https://github.com/sdv-dev/SDV/issues/2682) by @R-Palazzo

## v1.27.0 - 2025-09-15

### New Features

* Create a specific warning type for the purposes of refitting a synthesizer - Issue [#2662](https://github.com/sdv-dev/SDV/issues/2662) by @frances-h
* [OneHotEncoding constraint] Allow me to specify whether to keep the one-hot columns or collapse them into one categorical column - Issue [#2650](https://github.com/sdv-dev/SDV/issues/2650) by @fealho

### Bugs Fixed

* "numerical_distributions" in HMASynthesizer get ignored - Issue [#2648](https://github.com/sdv-dev/SDV/issues/2648) by @fealho

### Internal

* Add helper method for transforming conditions - Issue [#2660](https://github.com/sdv-dev/SDV/issues/2660) by @rwedge
* [OneHotEncoding Constraint] For higher quality, ensure the model creates floating point numbers - Issue [#2649](https://github.com/sdv-dev/SDV/issues/2649) by @fealho

## v1.26.0 - 2025-08-18

### New Features

* Allow me to add a custom constraint to PARSynthesizer - Issue [#2641](https://github.com/sdv-dev/SDV/issues/2641) by @fealho

### Bugs Fixed

* Cannot apply multiple constraints to PARSynthesizer (one with all context cols, and one with all non-context cols) - Issue [#2630](https://github.com/sdv-dev/SDV/issues/2630) by @fealho
* In PARSynthesizer, I cannot apply a context column that is sdtype `id` (or another PII type) - Issue [#2466](https://github.com/sdv-dev/SDV/issues/2466) by @fealho

## v1.25.0 - 2025-08-04

### New Features

* Streamline the loading of a synthesizer into a single function - Issue [#2619](https://github.com/sdv-dev/SDV/issues/2619) by @R-Palazzo
* Add `InterTableCondition` condition class - Issue [#2611](https://github.com/sdv-dev/SDV/issues/2611) by @frances-h
* Add `DataFrameCondition` condition class - Issue [#2610](https://github.com/sdv-dev/SDV/issues/2610) by @gsheni
* Add ability to specify old transformer behavior for ID columns - Issue [#2450](https://github.com/sdv-dev/SDV/issues/2450) by @fealho

### Bugs Fixed

* Sample not backwards compatible with older synthesizers - Issue [#2635](https://github.com/sdv-dev/SDV/issues/2635) by @rwedge
* Add custom error message when using a multi table synthesizer with empty data - Issue [#2633](https://github.com/sdv-dev/SDV/issues/2633) by @fealho
* Rename `InterTableCondition` to `MultiTableCondition` - Issue [#2622](https://github.com/sdv-dev/SDV/issues/2622) by @frances-h
* Error when sampling using HMA after changing the default distributions and grandparent - parent - child table relationship - Issue [#2606](https://github.com/sdv-dev/SDV/issues/2606) by @rwedge
* auto_assign_transformers crashes if the datetime column is represented as Timestamps - Issue [#2594](https://github.com/sdv-dev/SDV/issues/2594) by @gsheni
* Metadata validation should error if metadata is empty - Issue [#2553](https://github.com/sdv-dev/SDV/issues/2553) by @fealho
* PARSynthesizer crashes during sample if there was an all-null column in the input data - Issue [#2473](https://github.com/sdv-dev/SDV/issues/2473) by @fealho
* PARSynthesizer can create different sequences with the same sequence key (uniqueness is not enforced) - Issue [#2467](https://github.com/sdv-dev/SDV/issues/2467) by @fealho
* In PARSynthesizer, I cannot apply a context column that is sdtype `id` (or another PII type) - Issue [#2466](https://github.com/sdv-dev/SDV/issues/2466) by @fealho
* Improve error message when trying to conditionally sample before fitting - Issue [#2366](https://github.com/sdv-dev/SDV/issues/2366) by @fealho

### Internal

* `DatetimeFormatter`: When `ValueError` occurs, the `pd.to_datetime` can fail due to format miss-match - Issue [#2541](https://github.com/sdv-dev/SDV/issues/2541) by @pvk-developer

## v1.24.1 - 2025-07-14

### Bugs Fixed

* Unable to add overlapping single-table constraints to a multi-table schema - Issue [#2601](https://github.com/sdv-dev/SDV/issues/2601) by @frances-h
* Resolve DeprecationWarning (pd.api.types.is_categorical_dtype is deprecated) - Issue [#2505](https://github.com/sdv-dev/SDV/issues/2505) by @gsheni

### Internal

* Refactor _create_config method - Issue [#2593](https://github.com/sdv-dev/SDV/issues/2593) by @fealho

### Maintenance

* Add workflow to release SDV on PyPI - Issue [#2603](https://github.com/sdv-dev/SDV/issues/2603) by @gsheni

## v1.24.0 - 2025-06-30

### New Features

* Add support for condtionally sampling column relationships and contextually anonymized columns - Issue [#2582](https://github.com/sdv-dev/SDV/issues/2582) by @frances-h
* Add support for conditional sampling null values - Issue [#2581](https://github.com/sdv-dev/SDV/issues/2581) by @frances-h

### Bugs Fixed

* Unable to sample synthetic data when using timezone extraction - Issue [#2589](https://github.com/sdv-dev/SDV/issues/2589) by @pvk-developer

### Internal

* Ignore Timezone Information in Datetime Constraints (Short-Term Fix) - Issue [#2595](https://github.com/sdv-dev/SDV/issues/2595) by @pvk-developer
* Add workflow to check that issues tied to PRs have a milestone - Issue [#2585](https://github.com/sdv-dev/SDV/issues/2585) by @amontanez24

## v1.23.0 - 2025-06-16

### New Features

* Update the wording used to get the version of SDV Community - Issue [#2568](https://github.com/sdv-dev/SDV/issues/2568) by @rwedge
* If I don't have `torch` installed, I should still be able to use SDV features that don't require torch - Issue [#2551](https://github.com/sdv-dev/SDV/issues/2551) by @rwedge

### Bugs Fixed

* I should not be able to load Metadata if it contains unrecognized keys - Issue [#2548](https://github.com/sdv-dev/SDV/issues/2548) by @R-Palazzo

### Internal

* Check pyproject for pre-release dependencies - Issue [#2564](https://github.com/sdv-dev/SDV/issues/2564) by @rwedge
* `DataProcessor` should assign `'id'` sdtype to ID columns instead of `'text'` - Issue [#2424](https://github.com/sdv-dev/SDV/issues/2424) by @R-Palazzo

### Maintenance

* Update python set up step in workflows to use latest python version - Issue [#2281](https://github.com/sdv-dev/SDV/issues/2281) by @frances-h

## v1.22.1 - 2025-06-03

### Bugs Fixed

* Original metadata object passed to ProgrammableConstraint - Issue [#2565](https://github.com/sdv-dev/SDV/issues/2565) by @frances-h

## v1.22.0 - 2025-06-02

### New Features

* Add violin option to get_column_plot function docstrings - Issue [#2554](https://github.com/sdv-dev/SDV/issues/2554) by @amontanez24
* Allow `fit` to be an optional method for `ProgrammableConstraint` and `SingleTableProgrammableConstraint` - Issue [#2525](https://github.com/sdv-dev/SDV/issues/2525) by @pvk-developer
* Add `ProgrammableConstraint` and `ProgrammableSingleTableConstraint` - Issue [#2513](https://github.com/sdv-dev/SDV/issues/2513) by @frances-h
* Enable single-table constraint reject sampling with multi-table synthesizers - Issue [#2512](https://github.com/sdv-dev/SDV/issues/2512) by @R-Palazzo
* Consolidate names of CAG/data patterns to `constraints` - Issue [#2492](https://github.com/sdv-dev/SDV/issues/2492) by @R-Palazzo
* Add 'version' parameter to SingleTableSynthesizer.get_metadata - Issue [#2484](https://github.com/sdv-dev/SDV/issues/2484) by @pvk-developer
* Add synthesizer.validate_cag method - Issue [#2471](https://github.com/sdv-dev/SDV/issues/2471) by @gsheni
* Add CAG validation to synthesizer.validate - Issue [#2470](https://github.com/sdv-dev/SDV/issues/2470) by @R-Palazzo
* Deprecate `ScalarInequality` and `ScalarRange` constraints - Issue [#2433](https://github.com/sdv-dev/SDV/issues/2433)
* Add CAG support to single table synthesizers - Issue [#2389](https://github.com/sdv-dev/SDV/issues/2389) by @fealho
* Add `OneHotEncoding` CAG pattern - Issue [#2387](https://github.com/sdv-dev/SDV/issues/2387) by @fealho
* Add the `FixedIncrements` CAG pattern - Issue [#2386](https://github.com/sdv-dev/SDV/issues/2386) by @gsheni
* Add `Range` CAG pattern - Issue [#2385](https://github.com/sdv-dev/SDV/issues/2385) by @fealho
* Add `Inequality` CAG pattern - Issue [#2384](https://github.com/sdv-dev/SDV/issues/2384) by @fealho
* Add `FixedCombinations` CAG pattern + add CAG base class to public - Issue [#2383](https://github.com/sdv-dev/SDV/issues/2383) by @frances-h

### Bugs Fixed

* Using old style constraints should raise a `FutureWarning` - Issue [#2561](https://github.com/sdv-dev/SDV/issues/2561) by @frances-h
* `get_constraints` for multi-table does not return single-table constraints - Issue [#2559](https://github.com/sdv-dev/SDV/issues/2559) by @R-Palazzo
* Formatted columns dropped by CAG constraints may invalidate constraint - Issue [#2550](https://github.com/sdv-dev/SDV/issues/2550) by @R-Palazzo
* SDV cannot be used on a readonly filesystem - Issue [#2543](https://github.com/sdv-dev/SDV/issues/2543) by @pvk-developer
* Incorrect formatting when applying `Inequality` constraint - Issue [#2524](https://github.com/sdv-dev/SDV/issues/2524)
* `ValueError` if conditionally sampling on a column dropped by constraints - Issue [#2519](https://github.com/sdv-dev/SDV/issues/2519) by @frances-h
* Constraint hits IntCastingNanError when reverse transforming int column with nan values - Issue [#2514](https://github.com/sdv-dev/SDV/issues/2514) by @frances-h
* Inequality CAG does not respect datetime format - Issue [#2495](https://github.com/sdv-dev/SDV/issues/2495)
* `auto_assign_transformers` errors after adding CAG pattern - Issue [#2490](https://github.com/sdv-dev/SDV/issues/2490) by @R-Palazzo
* Evaluate and improve CAG pattern testing coverage - Issue [#2489](https://github.com/sdv-dev/SDV/issues/2489) by @fealho
* Inequality CAG errors out if data contains NaN values - Issue [#2488](https://github.com/sdv-dev/SDV/issues/2488) by @R-Palazzo
* Add multi-table CAG support - Issue [#2487](https://github.com/sdv-dev/SDV/issues/2487) by @frances-h
* PARSynthesizer is not aware of the sdtypes produced after pre-processing - Issue [#2482](https://github.com/sdv-dev/SDV/issues/2482) by @fealho
* Make single table CAGs backwards compatible - Issue [#2446](https://github.com/sdv-dev/SDV/issues/2446) by @fealho

## v1.21.0 - 2025-05-16

### New Features

* Add an API for copying the Metadata - Issue [#2530](https://github.com/sdv-dev/SDV/issues/2530) by @amontanez24
* Add an API for removing a table from the Metadata - Issue [#2527](https://github.com/sdv-dev/SDV/issues/2527) by @amontanez24
* Add an API for removing a column from the Metadata - Issue [#2526](https://github.com/sdv-dev/SDV/issues/2526) by @amontanez24
* Allow SDV to be used on a readonly filesystem - Issue [#2517](https://github.com/sdv-dev/SDV/issues/2517) by @pvk-developer
* Allow me to put in additional options when reading multiple CSV files from `CSVHandler` - Issue [#2478](https://github.com/sdv-dev/SDV/issues/2478) by @pvk-developer

### Internal

* Dtypes benchmark should include missing values for any dtypes that support it - Issue [#2494](https://github.com/sdv-dev/SDV/issues/2494) by @rwedge

## v1.20.1 - 2025-05-01

### Bugs Fixed

* Show a warning if I'm trying to refit/sample from a synthesizer but the metadata has changed - Issue [#2463](https://github.com/sdv-dev/SDV/issues/2463) by @pvk-developer
* Metadata auto-detection should not be creating a schema where a foreign key column is reused - Issue [#2454](https://github.com/sdv-dev/SDV/issues/2454) by @pvk-developer
* Metadata validation does not catch the case where a foreign key is reused - Issue [#2453](https://github.com/sdv-dev/SDV/issues/2453) by @pvk-developer

### Maintenance

* Remove dtypes github action workflow - Issue [#2475](https://github.com/sdv-dev/SDV/issues/2475) by @gsheni
* Use IndexGenerator instead of IDGenerator from RDT - Issue [#2432](https://github.com/sdv-dev/SDV/issues/2432) by @amontanez24

## v1.20.0 - 2025-04-14

### New Features

* When auto-detecting metadata, add a parameter to control the foreign key detection algorithm - Issue [#2456](https://github.com/sdv-dev/SDV/issues/2456) by @amontanez24
* Provide a more descriptive error message when Regex is is not supported - Issue [#2434](https://github.com/sdv-dev/SDV/issues/2434) by @R-Palazzo
* Update transformer assignment for `id` columns - Issue [#2416](https://github.com/sdv-dev/SDV/issues/2416) by @frances-h
* When in doubt, metadata auto-detection should mark columns as sdtype `categorical` - Issue [#2413](https://github.com/sdv-dev/SDV/issues/2413) by @lajohn4747
* Metadata auto-detection should find `id` columns that are not primary/foreign keys - Issue [#2412](https://github.com/sdv-dev/SDV/issues/2412) by @amontanez24

### Bugs Fixed

* Metadata visualization doesn't indicate which columns are sequence key or sequence index - Issue [#2411](https://github.com/sdv-dev/SDV/issues/2411) by @lajohn4747

### Internal

* Store metadata as `Metadata` for `BaseSynthesizer` - Issue [#2445](https://github.com/sdv-dev/SDV/issues/2445) by @fealho

## v1.19.0 - 2025-03-12

### New Features

* Allow re-writes to metadata JSON files - Issue [#2392](https://github.com/sdv-dev/SDV/issues/2392) by @lajohn4747

### Bugs Fixed

* GaussianCopula is not reporting the correct distribution name in the case of a fallback - Issue [#2394](https://github.com/sdv-dev/SDV/issues/2394) by @fealho

### Internal

* Only Notify Slack on dtype Support Additions or Removals - Issue [#2406](https://github.com/sdv-dev/SDV/issues/2406) by @pvk-developer

### Maintenance

* Support Python 3.13 - Issue [#2270](https://github.com/sdv-dev/SDV/issues/2270) by @rwedge

## v1.18.0 - 2025-02-14

### New Features

* When detecting metadata from dataframes, allow me the option to turn on/off sdtype and relationship detection - Issue [#2341](https://github.com/sdv-dev/SDV/issues/2341) by @fealho
* Surface more detailed error info when detecting metadata from dataframes - Issue [#2327](https://github.com/sdv-dev/SDV/issues/2327) by @R-Palazzo

### Bugs Fixed

* Conditional sampling error when using a datetime column as a context column with PAR Synthesizer - Issue [#2187](https://github.com/sdv-dev/SDV/issues/2187) by @pvk-developer
* PARSynthesizer is synthesizing integers for the `sequence_key` column when source data is text - Issue [#1880](https://github.com/sdv-dev/SDV/issues/1880) by @fealho

### Maintenance

* Update our upload-artifact github action version - Issue [#2370](https://github.com/sdv-dev/SDV/issues/2370) by @amontanez24

## v1.17.4 - 2025-01-20

### New Features

* Update the warning that's displayed when using HMA on complex schemas - Issue [#2277](https://github.com/sdv-dev/SDV/issues/2277) by @R-Palazzo

### Bugs Fixed

* Release Notes generator is creating new notes incorrectly - Issue [#2348](https://github.com/sdv-dev/SDV/issues/2348) by @amontanez24
* Support the ability to pass in `None` for both `get_column_plot` and `get_column_pair_plot` - Issue [#2343](https://github.com/sdv-dev/SDV/issues/2343) by @R-Palazzo
* Metadata `anonymize` doesn't produce the right `METADATA_SPEC_VERSION` - Issue [#2304](https://github.com/sdv-dev/SDV/issues/2304) by @R-Palazzo
* GaussianCopula `get_learned_distributions` crashes if nothing was learned - Issue [#2297](https://github.com/sdv-dev/SDV/issues/2297) by @R-Palazzo
* Sampling with HMA Synthesizer generates many `SingleTableMetadata` deprecation warnings - Issue [#2290](https://github.com/sdv-dev/SDV/issues/2290) by @R-Palazzo

### Maintenance

* Include stack trace when sampling errors are surfaced - Issue [#2326](https://github.com/sdv-dev/SDV/issues/2326) by @amontanez24
* Combine  `static_code_analysis.yml` with `release_notes.yml` - Issue [#2305](https://github.com/sdv-dev/SDV/issues/2305) by @R-Palazzo

## v1.17.3 - 2024-12-17

### Maintenance

* Pandas FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated... - Issue [#2286](https://github.com/sdv-dev/SDV/issues/2286) by @fealho

## v1.17.2 - 2024-11-18

### New Features

* Update integer boundaries for ID columns - Issue [#2285](https://github.com/sdv-dev/SDV/issues/2285) by @R-Palazzo

### Bugs Fixed

* When using GaussianCopula, show a warning if a column in numerical_distributions cannot be applied - Issue [#2296](https://github.com/sdv-dev/SDV/issues/2296) by @pvk-developer 
* Incorrect column name ordering for Multi-Table Synthesizer - Issue [#2280](https://github.com/sdv-dev/SDV/issues/2280) by @R-Palazzo
* Inequality constraint cannot be applied to compare datetime to date - Issue [#2275](https://github.com/sdv-dev/SDV/issues/2275) by @pvk-developer
* PARSynthesizer is not learning rounding scheme for numerical columns - Issue [#2274](https://github.com/sdv-dev/SDV/issues/2274) by @frances-h
* Unable to turn off rounding scheme for a column (need a warning) - Issue [#2266](https://github.com/sdv-dev/SDV/issues/2266) by @fealho

### Maintenance

* Refactor `_fit` Method of `GaussianCopulaSynthesizer` for Modularity - Issue [#2267](https://github.com/sdv-dev/SDV/issues/2267) by @pvk-developer
* Add integration tests to code coverage report - Issue [#2263](https://github.com/sdv-dev/SDV/issues/2263) by @amontanez24
* Add support for numpy 2.0.0 - Issue [#2078](https://github.com/sdv-dev/SDV/issues/2078) by @R-Palazzo

## v1.17.1 - 2024-10-10

### Bugs Fixed

* Extraneous metadata warning is printed when customizing HMA Synthesizer - Issue [#2250](https://github.com/sdv-dev/SDV/issues/2250) by @pvk-developer

### Internal

* `ValueError` raised when adding columns to a new `Metadata` instance - Issue [#2252](https://github.com/sdv-dev/SDV/issues/2252) by @pvk-developer
* Enhance Benchmark Report Generation - Issue [#2235](https://github.com/sdv-dev/SDV/issues/2235) by @pvk-developer

### Maintenance

* Use PyDrive2 instead of PyDrive - Issue [#2238](https://github.com/sdv-dev/SDV/issues/2238) by @pvk-developer

## v1.17.0 - 2024-10-02

This release consolidates the `SingleTableMetadata` and `MultiTableMetadata` classes into one new class simply called `Metadata`. The old classes are now deprecated. The new class has the same structure as the `MultiTableMetadata` class, except it will work with single table synthesizers.

### New Features

* Add `metadata.validate_table` method for single table usage - Issue [#2215](https://github.com/sdv-dev/SDV/issues/2215) by @R-Palazzo
* Improve usage of `detect_from_dataframes` function - Issue [#2214](https://github.com/sdv-dev/SDV/issues/2214) by @amontanez24
* For single-table use cases, make it frictionless to update Metadata - Issue [#2213](https://github.com/sdv-dev/SDV/issues/2213) by @R-Palazzo
* Add a warning if you're loading a SingleTableMetadata object - Issue [#2210](https://github.com/sdv-dev/SDV/issues/2210) by @R-Palazzo
* Move all tests over to use Metadata instead of SingleTableMetadata and MultiTableMetadata - Issue [#2151](https://github.com/sdv-dev/SDV/issues/2151) by @lajohn4747
* Update demos to use new metadata - Issue [#2131](https://github.com/sdv-dev/SDV/issues/2131) by @lajohn4747
* Enable evaluation methods to work with new metadata - Issue [#2130](https://github.com/sdv-dev/SDV/issues/2130) by @pvk-developer
* Enable multi table synthesizers to use new Metadata - Issue [#2129](https://github.com/sdv-dev/SDV/issues/2129) by @lajohn4747
* Enable single table synthesizers to use new Metadata - Issue [#2128](https://github.com/sdv-dev/SDV/issues/2128) by @lajohn4747
* Create unified Metadata class - Issue [#2104](https://github.com/sdv-dev/SDV/issues/2104) by @lajohn4747
* Make "table" default table name - Issue [#2245](https://github.com/sdv-dev/SDV/issues/2245) by @fealho

## v1.16.2 - 2024-09-25

### New Features

* Supported data types benchmark - Issue [#2200](https://github.com/sdv-dev/SDV/issues/2200) by @pvk-developer

### Bugs Fixed

* The `_validate_circular_relationships` method may fail to detect circular relationships - Issue [#2205](https://github.com/sdv-dev/SDV/issues/2205) by @fealho

## v1.16.1 - 2024-08-27

### Internal

* [dtypes] `FixedIncrements` Fails with New Numerical Data Types - Issue [#2157](https://github.com/sdv-dev/SDV/issues/2157) by @R-Palazzo

## v1.16.0 - 2024-08-22

This release enables the `HMASynthesizer` and other utility functions to work with null foreign key values! It also adds an `anonymization` method to the metadata classes. Additionally, it patches a bug that lets SDV work with more Pandas data types.

### New Features

* Add metadata anonymization to public SDV - Issue [#2137](https://github.com/sdv-dev/SDV/issues/2137) by @R-Palazzo
* Switch drop_missing_values in in drop_unknown_references to support null foreign keys by default - Issue [#2076](https://github.com/sdv-dev/SDV/issues/2076) by @R-Palazzo
* Support nullable foreign keys in HMA - Issue [#2063](https://github.com/sdv-dev/SDV/issues/2063) by @rwedge
* Remove input error from base synthesizer class once nullable foreign keys are supported - Issue [#2057](https://github.com/sdv-dev/SDV/issues/2057) by @rwedge
* Support null foreign keys in get_random_subset - Issue [#2056](https://github.com/sdv-dev/SDV/issues/2056) by @R-Palazzo
* Warn the user if they are trying to save an unfit synthesizer - Issue [#1961](https://github.com/sdv-dev/SDV/issues/1961) by @fealho

### Bugs Fixed

* Using FixedCombinations constraint with an integer constraint column causes sampling to fail - Issue [#2183](https://github.com/sdv-dev/SDV/issues/2183) by @R-Palazzo
* Metadata Detection Fails with new Data Type - Issue [#2182](https://github.com/sdv-dev/SDV/issues/2182) by @R-Palazzo
* Unable visualize just the real data (or just the synthetic data) in a multi-table setting - Issue [#2160](https://github.com/sdv-dev/SDV/issues/2160) by @R-Palazzo
* [dtypes] Numerical Formatter Fails to Learn Format of New Data Types - Issue [#2156](https://github.com/sdv-dev/SDV/issues/2156) by @R-Palazzo
* Primary keys may not be unique for variable length regexes - Issue [#2116](https://github.com/sdv-dev/SDV/issues/2116) by @amontanez24
* Confusing warning when using GANs that suggests that CUDA isn't being used - Issue [#2052](https://github.com/sdv-dev/SDV/issues/2052) by @fealho
* PAR DiagnosticReport not 1.0 with float categorical columns - Issue [#1910](https://github.com/sdv-dev/SDV/issues/1910) by @lajohn4747
* In `PARSynthesizer` I cannot pass in datetime context (`InvalidDataError` during fitting) - Issue [#1485](https://github.com/sdv-dev/SDV/issues/1485) by @lajohn4747

### Internal

* Enabling sdv logging causes tests to fail locally - Issue [#2162](https://github.com/sdv-dev/SDV/issues/2162) by @amontanez24
* Separate primary key detection functionality - Issue [#2101](https://github.com/sdv-dev/SDV/issues/2101) by @amontanez24

### Maintenance

* [dtypes] Update the NumericalFormatter to use the `learn_rounding_digits` from RDT - Issue [#2164](https://github.com/sdv-dev/SDV/issues/2164) by @R-Palazzo
* Mock every usage of `is_faker_function` to speed up the unit tests - Issue [#2163](https://github.com/sdv-dev/SDV/issues/2163) by @R-Palazzo
* Review docs-related dev dependencies - Issue [#2148](https://github.com/sdv-dev/SDV/issues/2148) by @rwedge
* Cap boto and botocore - Issue [#2123](https://github.com/sdv-dev/SDV/issues/2123) by @lajohn4747

## v1.15.0 - 2024-07-11

This release adds a new utils function called `get_random_sequence_subset`, that allows users to get a subset of sequential data.

### New Features

* Add utils to the Top Level Package. - Issue [#2119](https://github.com/sdv-dev/SDV/issues/2119) by @pvk-developer
* Add a utility function `get_random_sequence_subset` - Issue [#2085](https://github.com/sdv-dev/SDV/issues/2085) by @amontanez24

### Bugs Fixed

* Context column cannot be a sequence key: Need better error message for this case - Issue [#2097](https://github.com/sdv-dev/SDV/issues/2097) by @gsheni
* Primary key and sequential key cannot be the same - Issue [#2096](https://github.com/sdv-dev/SDV/issues/2096) by @lajohn4747
* Error when applying `FixedCombinations` constraint on a child table with multiple parents in `HMASynthesizer` - Issue [#2087](https://github.com/sdv-dev/SDV/issues/2087) by @pvk-developer
* PARSynthesizer errors during `fit` if sequence_index is numerical sdtype - Issue [#2079](https://github.com/sdv-dev/SDV/issues/2079) by @lajohn4747
* Cap numpy to less than 2.0.0 until SDV supports - Issue [#2075](https://github.com/sdv-dev/SDV/issues/2075) by @gsheni
* Rename the `file_name` parameter to `filepath` parameter in ExcelHandler - Issue [#2065](https://github.com/sdv-dev/SDV/issues/2065) by @lajohn4747
* HMA sampling crashes when unknown sdtype detected for numerical column - Issue [#2064](https://github.com/sdv-dev/SDV/issues/2064) by @lajohn4747
* HMA Synthesizer's `scale` parameter doesn't work for small values - Issue [#2045](https://github.com/sdv-dev/SDV/issues/2045) by @lajohn4747
* PAR DiagnosticReport not 1.0 with float categorical columns - Issue [#1910](https://github.com/sdv-dev/SDV/issues/1910) by @lajohn4747
* If a parent has 0/1 children, HMASynthesizer may create constant data - Issue [#1895](https://github.com/sdv-dev/SDV/issues/1895) by @gsheni

### Internal

* Add timeouts to requests in release notes script - Issue [#2067](https://github.com/sdv-dev/SDV/issues/2067) by @gsheni
* Investigate HMA case where parent is missing num_rows column - Issue [#1703](https://github.com/sdv-dev/SDV/issues/1703) by @gsheni

### Maintenance

* Release notes should not include PRs - Issue [#2074](https://github.com/sdv-dev/SDV/issues/2074) by @amontanez24
* Switch to using ruff for Python linting and code formatting - Issue [#1803](https://github.com/sdv-dev/SDV/issues/1803) by @gsheni

## v1.14.0 - 2024-06-13

This release provides a number of new features. A big one is that it adds the ability to fit the `HMASynthesizer` on disconnected schemas! It also enables the `PARSynthesizer` to work with constraints in certain conditions. More specifically, the `PARSynthesizer` can now handle constraints as long as the columns involved in the constraints are either exclusively all context columns or exclusively all non-context columns.

Additionally, a `verbose` parameter was added to the `TVAESynthesizer` to get a more detailed progress bar. Also, a bug was corrected that renamed the `file_path` parameter in the `ExcelHandler.read()` method to `filepath` as specified in the official [SDV docs](https://docs.sdv.dev/sdv/multi-table-data/data-preparation/loading-data/excel#read).

### Internal

* Add workflow to generate release notes - Issue [#2050](https://github.com/sdv-dev/SDV/issues/2050) by @amontanez24

### Bugs Fixed

* PARSynthesizer: Duplicate sequence index values when `sequence_length` is higher than real data - Issue [#2031](https://github.com/sdv-dev/SDV/issues/2031) by @lajohn4747
* PARSynthesizer model won't fit if sequence_index is missing - Issue [#1972](https://github.com/sdv-dev/SDV/issues/1972) by @lajohn4747
* `DataProcessor` never gets assigned a `table_name`. - Issue [#1964](https://github.com/sdv-dev/SDV/issues/1964) by @fealho

### New Features

* Rename `file_path` to `filepath` parameter in ExcelHandler - Issue [#2055](https://github.com/sdv-dev/SDV/issues/2055) by @amontanez24
* Enable the ability to run multi table synthesizers on disjointed table schemas - Issue [#2047](https://github.com/sdv-dev/SDV/issues/2047) by @lajohn4747
* Add header to log.csv file - Issue [#2046](https://github.com/sdv-dev/SDV/issues/2046) by @lajohn4747
* If no filepath is provided, do not create a file during `sample` - Issue [#2042](https://github.com/sdv-dev/SDV/issues/2042) by @lajohn4747
* Add verbosity to `TVAESynthesizer` - Issue [#1990](https://github.com/sdv-dev/SDV/issues/1990) by @fealho
* Allow constraints in PARSynthesizer (for all context cols, or all non-context columns) - Issue [#1936](https://github.com/sdv-dev/SDV/issues/1936) by @lajohn4747
* Improve error message when sampling on a non-CPU device - Issue [#1819](https://github.com/sdv-dev/SDV/issues/1819) by @fealho
* Better data validation message for `auto_assign_transformers` - Issue [#1509](https://github.com/sdv-dev/SDV/issues/1509) by @lajohn4747

### Miscellaneous

* Do not enforce min/max on sequence index column - Issue [#2043](https://github.com/sdv-dev/SDV/pull/2043)
* Include validation check for single table auto_assign_transformers - Issue [#2021](https://github.com/sdv-dev/SDV/pull/2021)
* Add the dummy context column to metadata and not to extra_context_column - Issue [#2019](https://github.com/sdv-dev/SDV/pull/2019)

# 1.13.1 - 2024-05-16

This release fixes the `ModuleNotFoundError` error that was causing the 1.13.0 release to fail.

## 1.13.0 - 2024-05-15

This release adds a utility function called `get_random_subset` that helps users get a subset of their multi-table data so that modeling can be done quicker. Given a dictionary of table names mapped to DataFrames, metadata, a main table and a desired number of rows to use for the main table, it will subsample the data in a way that maintains referential integrity.

This release also adds two new local file handlers: the `CSVHandler` and the `ExcelHandler`. This enables users to easily load from and save synthetic data to these files types. These handlers return data and metadata in the multi-table format, so we also added the function `get_table_metadata` to get a `SingleTableMetadata` object from a `MultiTableMetadata` object.

Finally, this release fixes some bugs that prevented synthesizers from working with data that had numerical column names.

### New Features

* Add `get_random_subset` poc utility function - Issue [#1877](https://github.com/sdv-dev/SDV/issues/1877) by @R-Palazzo
* Add usage logging - Issue [#1903](https://github.com/sdv-dev/SDV/issues/1903) by @pvk-developer
* Move function `drop_unknown_references` from `poc` to be directly under `utils` - Issue [#1947](https://github.com/sdv-dev/SDV/issues/1947) by @R-Palazzo
* Add CSVHandler - Issue [#1949](https://github.com/sdv-dev/SDV/issues/1949) by @pvk-developer
* Add ExcelHandler - Issue [#1950](https://github.com/sdv-dev/SDV/issues/1950) by @pvk-developer
* Add get_table_metadata function - Issue [#1951](https://github.com/sdv-dev/SDV/issues/1951) by @R-Palazzo
* Save usage log file as a csv - Issue [#1974](https://github.com/sdv-dev/SDV/issues/1974) by @frances-h
* Split out metadata creation from data import in the local files handlers - Issue [#1975](https://github.com/sdv-dev/SDV/issues/1975) by @pvk-developer
* Improve error message when trying to sample before fitting (single table) - Issue [#1978](https://github.com/sdv-dev/SDV/issues/1978) by @R-Palazzo

### Bugs Fixed

* Metadata detection crashes when the column names are integers (`AttributeError: 'int' object has no attribute 'lower'`) - Issue [#1933](https://github.com/sdv-dev/SDV/issues/1933) by @lajohn4747
* Synthesizers crash when column names are integers (`TypeError: unsupported operand`) - Issue [#1935](https://github.com/sdv-dev/SDV/issues/1935) by @lajohn4747
* Switch parameter order in drop_unknown_references - Issue [#1944](https://github.com/sdv-dev/SDV/issues/1944) by @R-Palazzo
* Unexpected NaN values in sequence_index when dataframe isn't reset - Issue [#1973](https://github.com/sdv-dev/SDV/issues/1973) by @fealho
* Fix pandas DtypeWarning in download_demo - Issue [#1980](https://github.com/sdv-dev/SDV/issues/1980) by @fealho

### Maintenance

* Only run unit and integration tests on oldest and latest python versions for macos - Issue [#1948](https://github.com/sdv-dev/SDV/issues/1948) by @frances-h

### Internal

* Update code to remove `FutureWarning` related to 'enforce_uniqueness' parameter - Issue [#1995](https://github.com/sdv-dev/SDV/issues/1995) by @pvk-developer

## 1.12.1 - 2024-04-19

This release makes a number of changes to how id columns are generated. By default, id columns with a regex will now have their values scrambled in the output. Id columns without a regex that are numeric will be created randomly. If they're not numeric, they will have a random suffix.

Additionally, improvements were made to the visibility of the `get_loss_values_plot`.

### New Features

* Create unique id for each synthesizer - Issue [#1902](https://github.com/sdv-dev/SDV/issues/1902) by @pvk-developer
* Generator Discriminator Loss Chart Color Change - Issue [#1916](https://github.com/sdv-dev/SDV/issues/1916) by @lajohn4747
* If using regex to generate values, scramble them - Issue [#1921](https://github.com/sdv-dev/SDV/issues/1921) by @amontanez24
* When generating ids without a regex, create them randomly - Issue [#1922](https://github.com/sdv-dev/SDV/issues/1922) by @frances-h

### Maintenance

* Cleanup automated PR workflows - Issue [#1926](https://github.com/sdv-dev/SDV/issues/1926) by @R-Palazzo

### Internal
* Add add-on modules to sys.modules  - Issue [#1924](https://github.com/sdv-dev/SDV/issues/1924) by @amontanez24

## 1.12.0 - 2024-04-16

This release adds support for Python 3.12! It also adds a number of feature improvements. It adds a `simplify_schema` utility function to the `sdv.utils.poc` module which simplifies multi-table schemas so they can be run using `HMASynthesizer`. Multi-table data dictionaries can now be saved directly to CSVs using the `sdv.datasets.local.save_csvs` utility function. Additionally, generator-discriminator loss values can now be plotted directly from CTGAN using the `get_loss_values_plot` method. This release also adds error messages when trying to load an SDV synthesizer on an older version of the SDV, or when trying to re-fit a synthesizer from an older version of the SDV.

This release also fixes a number of bugs. Metadata auto-detection now validates that all primary keys are unique, and the metadata correctly validates sdtypes in a column relationship. Bugs in the `HMASynthesizer` that would cause the diagnostic score to not be equal to 1.0 for cardinality and data validity were fixed. Finally, errors in constraints now correctly raise a `ConstraintsNotMetError` instead of an `InvalidData` error.

### New Features

* sdv helper function for generating generator-discriminator loss charts - Issue [#1828](https://github.com/sdv-dev/SDV/issues/1828) by @lajohn4747
* Add utility function to simplify multi-table schemas - Issue [#1832](https://github.com/sdv-dev/SDV/issues/1832) by @R-Palazzo
* Show an error if I accidentally load an SDV synthesizer on an older version of SDV - Issue [#1837](https://github.com/sdv-dev/SDV/issues/1837) by @pvk-developer
* Show an error when attempting to re-train a synthesizer that was created on a previous SDV version - Issue [#1838](https://github.com/sdv-dev/SDV/issues/1838) by @pvk-developer
* Add warning when user tries to train a model using datetime values without a datetime_format set - Issue [#1847](https://github.com/sdv-dev/SDV/issues/1847) by @pvk-developer
* Add a function to save my multi-table data as CSVs - Issue [#1849](https://github.com/sdv-dev/SDV/issues/1849) by @R-Palazzo
* Deprecate `SingleTablePreset` (including `FastML` Preset) - Issue [#1855](https://github.com/sdv-dev/SDV/issues/1855) by @lajohn4747
* Missing error message if the user forgets to add a `sequence_key` when using PARSynthesizer - Issue [#1883](https://github.com/sdv-dev/SDV/issues/1883) by @frances-h

### Internal

* Add dependency checker - Issue [#1818](https://github.com/sdv-dev/SDV/issues/1818) by @frances-h

### Bugs Fixed

* Metadata isn't validating sdtypes in a column relationship (public SDV only) - Issue [#1781](https://github.com/sdv-dev/SDV/issues/1781) by @R-Palazzo
* Contextual Anonymization transformers shouldn't be used for primary keys - Issue [#1807](https://github.com/sdv-dev/SDV/issues/1807) by @fealho
* HMASynthesizer diagnostic score is not 1.0 when using `'truncnorm'` distribution - Issue [#1831](https://github.com/sdv-dev/SDV/issues/1831) by @frances-h
* InvalidDataError: The provided data does not match the metadata (although it matches) - Issue [#1833](https://github.com/sdv-dev/SDV/issues/1833) by @pvk-developer
* HMA likelihood match should respect cardinality - Issue [#1834](https://github.com/sdv-dev/SDV/issues/1834) by @fealho
* When inappropriately applying ScalarRange constraint, InvalidDataError is being returned instead of ConstraintsNotMetError - Issue [#1842](https://github.com/sdv-dev/SDV/issues/1842) by @pvk-developer
* When inappropriately applying a CustomConstraint, an InvalidDataError is being returned instead of ConstraintsNotMetError - Issue [#1856](https://github.com/sdv-dev/SDV/issues/1856) by @pvk-developer
* Error in Setting `IDGenerator` for Primary Key columns - Issue [#1862](https://github.com/sdv-dev/SDV/issues/1862) by @lajohn4747
* Metadata auto-detection should ensure primary keys are unique (special sdtypes are not exempt from this rule!) - Issue [#1871](https://github.com/sdv-dev/SDV/issues/1871) by @R-Palazzo

### Maintenance

* Support Python 3.12 - Issue [#1704](https://github.com/sdv-dev/SDV/issues/1704) by @fealho
* Add dependency checker - Issue [#1818](https://github.com/sdv-dev/SDV/issues/1818) by @frances-h
* Add bandit workflow - Issue [#1881](https://github.com/sdv-dev/SDV/issues/1881) by @amontanez24

## 1.11.0 - 2024-03-21

This release adds the `poc` utility submodule to help users more easily create a proof-of-concept with multi-table datasets. The `poc` submodule includes the `drop_unknown_references` utility function to automatically drop unknown references in a multi-table dataset. Additionally, multiple columns in the metadata can now be updated at once using the `update_columns` and `update_columns_metadata` methods. The SDV now also warns users when a synthesizer is loaded that was fitted on a different version of the SDV.

### New Features

* Make the `get_parameters` function consistent between synthesizers - Issue [#1756](https://github.com/sdv-dev/SDV/issues/1756) by @fealho
* Reinstate `get_table_parameters` for the multi-table synthesizers - Issue [#1757](https://github.com/sdv-dev/SDV/issues/1757) by @fealho
* Validate condition and provide user-friendly messages for NaN/missing values (currently unsupported) - Issue [#1758](https://github.com/sdv-dev/SDV/issues/1758) by @fealho
* Improved error message if a column is already present in a relationship - Issue [#1770](https://github.com/sdv-dev/SDV/issues/1770) by @R-Palazzo
* Better error messaging for nullable foreign keys  - Issue [#1780](https://github.com/sdv-dev/SDV/issues/1780) by @fealho
* Add a utility to drop unknown references (and enforce referential integrity) - Issue [#1792](https://github.com/sdv-dev/SDV/issues/1792) by @R-Palazzo
* Add `update_columns` and `update_columns_metadata` methods to metadata - Issue [#1804](https://github.com/sdv-dev/SDV/issues/1804) by @R-Palazzo
* Add `get_column_names` method to metadata - Issue [#1805](https://github.com/sdv-dev/SDV/issues/1805) by @frances-h
* Show original error message when plugin fails to load - Issue [#1816](https://github.com/sdv-dev/SDV/issues/1816) by @rwedge
* Show warning when loading a synthesizer on a previously-saved SDV version - Issue [#1836](https://github.com/sdv-dev/SDV/issues/1836) by @pvk-developer
* Add verbosity to `drop_unknown_references` - Issue [#1845](https://github.com/sdv-dev/SDV/issues/1845) by @R-Palazzo
* Create a `poc` module for utilities that help with proof-of-concept  - Issue [#1846](https://github.com/sdv-dev/SDV/issues/1846) by @pvk-developer

### Internal

* Cleanup `utils` module: Make internal functions private - Issue [#1793](https://github.com/sdv-dev/SDV/issues/1793) by @R-Palazzo
* Transition from using setup.py to pyroject.toml to specify project metadata - Issue [#1801](https://github.com/sdv-dev/SDV/issues/1801) by @R-Palazzo
* Remove bumpversion and use bump-my-version - Issue [#1802](https://github.com/sdv-dev/SDV/issues/1802) by @R-Palazzo

### Maintenance

* Transition from using setup.py to pyroject.toml to specify project metadata - Issue [#1801](https://github.com/sdv-dev/SDV/issues/1801) by @R-Palazzo
* Remove bumpversion and use bump-my-version - Issue [#1802](https://github.com/sdv-dev/SDV/issues/1802) by @R-Palazzo
* Add support for RDT 1.10.0 - Issue [#1850](https://github.com/sdv-dev/SDV/issues/1850) by @amontanez24

### Bugs Fixed

* `INFO` logs mention table name as `None` for single table data - Issue [#1814](https://github.com/sdv-dev/SDV/issues/1814) by @lajohn4747
* Fix drop_unknown_references for null foreign keys - Issue [#1820](https://github.com/sdv-dev/SDV/issues/1820) by @R-Palazzo

## 1.10.0 - 2024-02-15

This release adds multiple improvements to handling premium transformers and column relationships, including using premium transformers even if the PII flag is set to true. Additionally, the SDV now warns users to save the metadata after auto-detection has been used. Semantic sdtype detection has also been improved to tokenize column names to prevent unexpected substring matches.

This release also fixes a few warning bugs and fixes an issue that would cause `metadata.to_dict` to fail for metadata loaded from older versions of the SDV. A few synthesizer bugs were also resolved. The quality of the sequence_index for the `PARSynthesizer` has been improved, and an issue that would cause `CTGANSynthesizer`, `TVAESynthesizer`, and `CopulaGANSynthesizer` to crash if all columns were to be generated from scratch has been fixed.

### Bugs Fixed

* HMASynthesizer sometimes creates null values (out-of-bounds parameters synthesized) - Issue [#1691](https://github.com/sdv-dev/SDV/issues/1691) by @fealho
* Unable to conditionally sample some rows when using a `ScalarRange` constraint - Issue [#1737](https://github.com/sdv-dev/SDV/issues/1737) by @fealho
* Metadata.to_dict fails on metadata instances pre 1.9.0 - Issue [#1739](https://github.com/sdv-dev/SDV/issues/1739) by @amontanez24
* Metadata auto-detection should not assign a primary key if there are NaN values in it - Issue [#1740](https://github.com/sdv-dev/SDV/issues/1740) by @R-Palazzo
* '<Synthesizer>' object has no attribute '_model' - Issue [#1741](https://github.com/sdv-dev/SDV/issues/1741) by @fealho
* Column relationship warning should be raised during synthesizer initialization only - Issue [#1750](https://github.com/sdv-dev/SDV/issues/1750) by @R-Palazzo
* Improve quality of `sequence_index`: Move the start dates into the context model - Issue [#1760](https://github.com/sdv-dev/SDV/issues/1760) by @frances-h
* Add-ons warning is raised twice for multi table synthesizers.  - Issue [#1768](https://github.com/sdv-dev/SDV/issues/1768) by @R-Palazzo

### New Features

* Metadata auto-detection should tokenize words before determining PII - Issue [#1725](https://github.com/sdv-dev/SDV/issues/1725) by @fealho
* Provide a friendlier error if data is stored as dtype `'category'` (CTGAN, TVAE) - Issue [#1735](https://github.com/sdv-dev/SDV/issues/1735) by @frances-h
* Allow the ability to easily remove primary keys - Issue [#1742](https://github.com/sdv-dev/SDV/issues/1742) by @frances-h
* Constraint should not be set on columns inside a gps relationship - Issue [#1748](https://github.com/sdv-dev/SDV/issues/1748) by @R-Palazzo
* Set the default transformer for GPS column relationship - Issue [#1749](https://github.com/sdv-dev/SDV/issues/1749) by @R-Palazzo
* Add a `version` module to align with SDV Enterprise - Issue [#1761](https://github.com/sdv-dev/SDV/issues/1761) by @R-Palazzo
* Warn users to save their metadata file after auto-detecting/updating it - Issue [#1762](https://github.com/sdv-dev/SDV/issues/1762) by @R-Palazzo
* Set the GPSNoiser as default transformer for GPS column relationship  - Issue [#1767](https://github.com/sdv-dev/SDV/issues/1767) by @R-Palazzo
* Update transformer assignment logic for handling pii - Issue [#1775](https://github.com/sdv-dev/SDV/issues/1775) by @R-Palazzo

## 1.9.0 - 2024-01-11

This release makes a number of improvements. It introduces a new concept to the metadata known as column relationships! Column relationships can be used to define when certain groups of columns in a table should be treated as a special concept (eg. address). You can add a column relationship by using the new `add_column_relationship` method. The metadata detection was also improved by allowing semantic sdtypes (eg. 'email', 'phone_number') to be detected as primary keys.

This release also patches some bugs. An issue messing up the likelihood matching in the `HMASynthesizer` was resolved. The `CTGANSynthesizer` no longer fails when using the `FixedCombinations` constraint. The `Inequality` constraint was also patched to handle datetimes better.

### Deprecations

* The `set_address_columns` method is deprecated in favor of `add_column_relationship`.

### New Features

* Improve error messages for composite keys - Issue [#1684](https://github.com/sdv-dev/SDV/issues/1684) by @frances-h
* Add column relationship validation to single table metadata - Issue [#1698](https://github.com/sdv-dev/SDV/issues/1698) by @frances-h
* Add add_column_relationship method to single table metadata - Issue [#1699](https://github.com/sdv-dev/SDV/issues/1699) by @frances-h
* Make synthesizers work with column_relationships - Issue [#1700](https://github.com/sdv-dev/SDV/issues/1700) by @R-Palazzo
* Metadata auto-detection should find primary keys of semantic sdtypes - Issue [#1724](https://github.com/sdv-dev/SDV/issues/1724) by @fealho

### Bugs Fixed

* InvalidDataError for Inequality constraint (even though data is valid) - Issue [#1692](https://github.com/sdv-dev/SDV/issues/1692) by @fealho
* `BaseIndependentSampler` crashes because it tries to cast id columns - Issue [#1712](https://github.com/sdv-dev/SDV/issues/1712) by @pvk-developer
* KeyError in `CTGANSynthesizer` when applying `FixedCombinations` constraint - Issue [#1717](https://github.com/sdv-dev/SDV/issues/1717) by @pvk-developer
* Fix _get_likelihoods not generating likelihood values - Issue [#1720](https://github.com/sdv-dev/SDV/pull/1720) by @frances-h

## 1.8.0 - 2023-12-05

This release adds support for the new Diagnostic Report from SDMetrics. This report calculates scores for three basic but important properties of your data: data validity, data structure and in the multi table case, relationship validity. Data validity checks that the columns of your data are valid (eg. correct range or values). Data structure makes sure the synthetic data has the correct columns. Relationship validity checks to make sure key references are correct and the cardinality is within ranges seen in the real data.

Additionally, a few bugs were fixed and functionality was improved around synthesizers. It is now possible to access the loss values for the `TVAESynthesizer` and `CTGANSynthesizer` by using the `get_loss_values` method. The `get_parameters` method is now more detailed and returns all the parameters used to make a synthesizer. The metadata is now capable of detecting some common pii sdtypes. Finally, a bug that made every parent row generated by the `HMASynthesizer` have at least one child row was patched. This should improve cardinality.

### Maintenance

* Address `SettingWithCopyWarning` (HMASynthesizer) - Issue [#1557](https://github.com/sdv-dev/SDV/issues/1557) by @pvk-developer
* Bump SDMetrics version - Issue [#1702](https://github.com/sdv-dev/SDV/issues/1702) by @amontanez24

### New Features

* Allow me to access loss values for GAN-based synthesizers  - Issue [#1671](https://github.com/sdv-dev/SDV/issues/1671) by @frances-h
* Create a unified `get_parameters` method for all multi-table synthesizers - Issue [#1674](https://github.com/sdv-dev/SDV/issues/1674) by @frances-h
* Set credentials key as variables - Issue [#1680](https://github.com/sdv-dev/SDV/issues/1680) by @R-Palazzo
* Identifying PII Sdtypes in Metadata - Issue [#1683](https://github.com/sdv-dev/SDV/issues/1683) by @R-Palazzo
* Make SDV compatible with the latest SDMetrics - Issue [#1687](https://github.com/sdv-dev/SDV/issues/1687) by @fealho
* SingleTablePreset uses FrequencyEncoder - Issue [#1695](https://github.com/sdv-dev/SDV/issues/1695) by @fealho

### Bugs Fixed

* HMASynthesizer creates too much synthetic data (always creates a child for every parent row) - Issue [#1673](https://github.com/sdv-dev/SDV/issues/1673) by @frances-h

## 1.7.0 - 2023-11-16

This release adds an alert to the `CTGANSynthesizer` during preprocessing. The alert informs the user if the fitting of the synthesizer is likely to be slow on their schema. Additionally, it is now possible to enforce that sampled datetime values stay within the range of the fitted data!

This release also makes internal changes to support address data in SDV Enterprise.

### New Features

* Add set_address_columns method - Issue [#1593](https://github.com/sdv-dev/SDV/issues/1593) by @R-Palazzo
* Update_transformers should raise error on address columns - Issue [#1594](https://github.com/sdv-dev/SDV/issues/1594) by @R-Palazzo
* add_constraints should raise error on address columns - Issue [#1595](https://github.com/sdv-dev/SDV/issues/1595) by @R-Palazzo
* Print alert if CTGANSynthesizer is likely to be slow - Issue [#1658](https://github.com/sdv-dev/SDV/issues/1658) by @fealho
* Set enforce_min_max_values to True for datetime transformers - Issue [#1676](https://github.com/sdv-dev/SDV/issues/1676) by @R-Palazzo

### Bugs Fixed

* Unable to visualize metadata (`Error: bad label format` and `CalledProcessError`) - Issue [#1625](https://github.com/sdv-dev/SDV/issues/1625) by @fealho
* Can't set address columns after fitting - Issue [#1661](https://github.com/sdv-dev/SDV/issues/1661) by @R-Palazzo

## 1.6.0 - 2023-11-07

This release improves user messaging in multiple ways. The most notable is that users will now see an alert if the `HMASynthesizer` is likely to be slow for their data's schema. Additionally, the logger messaging for constraints and the error messaging when setting distributions on non-parametric models was made more detailed.

The visualization plots in the `sdv.evaluation` sub-package all got a new parameter called `plot_type`, allowing the users to specify the plot type to use if the one being inferred is not useful. The `sdv.datasets.local.load_csvs` method now has a parameter called `read_csv_parameters`, that allow users to specify how the csvs should be read during loading. The same change was also made to the `sdv.metadata.multi_table.detect_table_from_csv`, `sdv.metadata.multi_table.detect_from_csvs`  and `sdv.metadata.single_table.detect_from_csv` methods.

Multiple bugs were resolved including one that caused new categories to be created during the sample step of `CTGANSynthesizer`.

### New Features

* Improve debug messages when a constraint falls back to reject sampling approach - Issue [#1478](https://github.com/sdv-dev/SDV/issues/1478) by @amontanez24
* Constraints should work with timezone-aware datetime columns - Issue [#1576](https://github.com/sdv-dev/SDV/issues/1576) by @fealho
* Better error message when trying to get distributions from non-parametric models - PR [#1633](https://github.com/sdv-dev/SDV/pull/1633) by @frances-h
* Add options to read CSV files - Issue [#1644](https://github.com/sdv-dev/SDV/issues/1644) by @lajohn4747
* Print alert if HMASynthesizer is likely to be slow - Issue [#1646](https://github.com/sdv-dev/SDV/issues/1646) by @lajohn4747
* Make SDV compatible with SDMetrics 0.12.1 - Issue [#1650](https://github.com/sdv-dev/SDV/issues/1650) by @pvk-developer
* Make function to estimate number of columns CTGAN produces - Issue [#1657](https://github.com/sdv-dev/SDV/issues/1657) by @fealho

### Bugs Fixed

* In get_available_demos, the num_tables column should be an int - Issue [#1420](https://github.com/sdv-dev/SDV/issues/1420) by @lajohn4747
* AttributeError when using specific locale strings (es_AR, fr_BE) - Issue [#1439](https://github.com/sdv-dev/SDV/issues/1439) by @lajohn4747
* Confusing error when passing in an empty dataframe (with constraints) - Issue [#1455](https://github.com/sdv-dev/SDV/issues/1455) by @lajohn4747
* HMASynthesizer: Better error message for learned distributions (misleading fit error) - Issue [#1579](https://github.com/sdv-dev/SDV/issues/1579) by @fealho
* Fix tests in SDV after update in RDT 1.7.1 - Issue [#1638](https://github.com/sdv-dev/SDV/issues/1638) by @lajohn4747
* CTGAN sometimes creates new categories (int data) - Issue [#1647](https://github.com/sdv-dev/SDV/issues/1647) by @pvk-developer
* CTGAN sometimes creates new categories (object data) - Issue [#1648](https://github.com/sdv-dev/SDV/issues/1648) by @pvk-developer
* Better error message if I provide an incompatible sdtype/locale combo - Issue [#1653](https://github.com/sdv-dev/SDV/issues/1653) by @pvk-developer

## 1.5.0 - 2023-10-13

Several improvements and bug fixes were made in this release. Most notably, the metadata detection was substantially improved. Support for the 'unknown' sdtype was added, providing more flexibility in data representation. The software now attempts to intelligently detect primary keys and identify parent-child relationships in the metadata, streamlining the metadata creation process.

Additionally, issues related to conditional sampling with negative float values, the inability to update transformers for columns created by constraints, and compatibility with numpy version 1.25 and higher were addressed. The default branch was also switched from 'master' to 'main' for better development practices. Various bugs and errors, including those involving HMA and datetime format detection, were also resolved.

### New Features

* Improve metadata detection - Issue [#1515](https://github.com/sdv-dev/SDV/issues/1515) by @R-Palazzo
* Support 'unknown' sdtype - Issue [#1516](https://github.com/sdv-dev/SDV/issues/1516) by @R-Palazzo
* Detect primary keys in metadata - Issue [#1521](https://github.com/sdv-dev/SDV/issues/1521) by @frances-h
* Detect relationships in MultiTableMetadata - Issue [#1522](https://github.com/sdv-dev/SDV/issues/1522) by @frances-h
* Make function to estimate number of columns HMA produces. - Issue [#1572](https://github.com/sdv-dev/SDV/issues/1572) by @fealho
* Add wrapper for get_cardinalty_plot - Issue [#1573](https://github.com/sdv-dev/SDV/issues/1573) by @frances-h
* [Metadata detection] Add a cardinality cap when choosing between categorical vs. numerical  - Issue [#1584](https://github.com/sdv-dev/SDV/issues/1584) by @pvk-developer
* [Metadata Detection] Only make primary/foreign keys sdtype `id` (leave others as `unknown`) - Issue [#1598](https://github.com/sdv-dev/SDV/issues/1598) by @amontanez24
* Check and supply a more descriptive error when trying to use `'gaussian_kde'` with HMA - Issue [#1604](https://github.com/sdv-dev/SDV/issues/1604) by @frances-h

### Bugs Fixed

* Conditional sampling with negative float values doesn't work - Issue [#1161](https://github.com/sdv-dev/SDV/issues/1161) by @fealho
* Cannot update transformers for columns that get created by constraints (`KeyError`) - Issue [#1454](https://github.com/sdv-dev/SDV/issues/1454) by @frances-h
* HMA produces KeyError for a schema with 3+ levels of depth - Issue [#1558](https://github.com/sdv-dev/SDV/issues/1558) by @fealho
* Columns consisting of only Nones are being detected as datetime - Issue [#1589](https://github.com/sdv-dev/SDV/issues/1589) by @pvk-developer
* HMASynthesizer throws an error when sampling multi table models with three levels of depths - Issue [#1600](https://github.com/sdv-dev/SDV/issues/1600) by @amontanez24
* `ValueError: Invalid distribution specification` when setting numerical_distributions on child table (HMA) - Issue [#1605](https://github.com/sdv-dev/SDV/issues/1605) by @fealho
* Bug: updating transformers in DataProcessor resets warning filters - Issue [#1618](https://github.com/sdv-dev/SDV/issues/1618) by @rwedge

### Maintenance

* Investigate how to get numpy >1.25 to pass - Issue [#1501](https://github.com/sdv-dev/SDV/issues/1501) by @rwedge
* Switch default branch from master to main - Issue [#1550](https://github.com/sdv-dev/SDV/issues/1550) by @amontanez24

## 1.4.0 - 2023-08-23

This release makes multiple improvements to the metadata. Both the single and multi table metadata classes now have a `validate_data` method. This method runs checks to validate the data against the current specifications in the metadata. The `SingleTableMetadata.visualize` is also improved. The sequence index is now shown in the same section as the sequence key. It also now shows all key and index information (eg. sequence key, primary key, sequence index) in one section.

The `CTGANSynthesizer` has been made more efficient in the following ways:
1. Boolean columns are now being skipped during `preprocess` like categorial columns are.
2. It is possible to apply other transformations to categorical columns and have `CTGAN` skip the one-hot encoding step.

Additional changes include that the columns labeled with the sdtype `id` will now go through the `IDGenerator` transformer by default and constraint transformations that were being overwritten during sampling will now be respected.

### New Features

* Add validate_data method to Metadata - Issue [#1518](https://github.com/sdv-dev/SDV/issues/1518) by @fealho
* Use IDGenerator for ID columns - Issue [#1519](https://github.com/sdv-dev/SDV/issues/1519) by @frances-h
* Metadata visualization for sequential data: Only create 2 sections - Issue [#1543](https://github.com/sdv-dev/SDV/issues/1543) by @frances-h

### Bugs Fixed

* Inefficient CTGAN modeling when adding categorical transformers - Issue [#1450](https://github.com/sdv-dev/SDV/issues/1450) by @fealho
* CTGANSynthesizer is assigning LabelEncoder to boolean columns (instead of None) - Issue [#1530](https://github.com/sdv-dev/SDV/issues/1530) by @fealho
* Metadata visualization for sequential data: Missing sequence index - Issue [#1542](https://github.com/sdv-dev/SDV/issues/1542) by @frances-h
* Constraint outputs are being overwritten in DataProcessor.reverse_transform - Issue [#1551](https://github.com/sdv-dev/SDV/issues/1551) by @amontanez24

## 1.3.0 - 2023-08-14

This release adds two new methods to the `MultiTableMetadata`: `detect_from_csvs` and `detect_From_dataframes`. These methods allow you to detect metadata for a whole dataset at once by either loading them from a folder or a dictionary mapping table names to the `pandas.DataFrames`. The `SingleTableMetadata` can now be visualized! Additionally, there is now a `summarized` option in the `show_table_details` parameter of the `visualize` methods. This will print each sdtype in the table and the number of columns that have that sdtype.

Additionally, this release patches a bug that prevented custom constraints from working on columns that were primary or alternate keys. It also adds support for Python 3.11!

### New Features

* Align default transformers between SDV and RDT - Issue [#1484](https://github.com/sdv-dev/SDV/issues/1484) by @R-Palazzo
* Add visualize method to SingleTableMetadata - Issue [#1517](https://github.com/sdv-dev/SDV/issues/1517) by @pvk-developer
* Add detect_from_csvs and detect_from_dataframes methods to MultiTableMetadata - Issue [#1520](https://github.com/sdv-dev/SDV/issues/1520) by @R-Palazzo
* Allow empty tables to be fitted using fit_processed_data - Issue [#1524](https://github.com/sdv-dev/SDV/issues/1524) by @fealho
* Summarized metadata visualization - Issue [#1525](https://github.com/sdv-dev/SDV/issues/1525) by @pvk-developer

### Bugs Fixed

* Cannot use custom constraint transforms for certain columns (inconsistent ordering in forward vs. reverse) - Issue [#1476](https://github.com/sdv-dev/SDV/issues/1476) by @fealho
* Cannot create custom constraint with primary key - Issue [#1514](https://github.com/sdv-dev/SDV/issues/1514) by @amontanez24

### Maintenance

* Add support for Python 3.11 - Issue [#1459](https://github.com/sdv-dev/SDV/issues/1459) by @fealho

## 1.2.1 - 2023-07-13

This release fixes a bug that caused the `Inequality` constraint and others to fail if there were None values in a datetime column.

### Bugs Fixed

* Inequality fails with None and datetime - Issue [#1471](https://github.com/sdv-dev/SDV/issues/1471) by @pvk-developer

### Maintenance

* Drop support for Python 3.7 - Issue [#1487](https://github.com/sdv-dev/SDV/issues/1487) by @pvk-developer

### Internal

* Make HMA use hierarchical sampling mixin - Issue [#1428](https://github.com/sdv-dev/SDV/issues/1428) by @frances-h
* Move progress bar out of base multi table synthesizer - Issue [#1486](https://github.com/sdv-dev/SDV/issues/1486) by @R-Palazzo

## 1.2.0 - 2023-06-07

This release adds a parameter called `verbose` to the `HMASynthesizer`. Setting it to True will show progress bars during the fitting steps. Additionally, performance optimizations were made to the modeling and initialization of the `HMASynthesizer`.

Multiple changes were made to enhance constraints. The `Range` constraint was improved to be able to generate more accurate data when null values are provided. Constraints are also now validated against the data when running `validate()` on any synthesizer.

Finally, some warnings were resolved.

### New Features

* Report fitting progress for the HMASynthesizer - Issue [#1440](https://github.com/sdv-dev/SDV/issues/1440) by @pvk-developer

### Bugs Fixed

* Range constraint does not produce cases of missing values & may create violative data - Issue [#1393](https://github.com/sdv-dev/SDV/issues/1393) by @R-Palazzo
* Synthesizers don't validate constraints during validate() - Issue [#1402](https://github.com/sdv-dev/SDV/issues/1402) by @pvk-developer
* Confusing error during metadata validation - Issue [#1417](https://github.com/sdv-dev/SDV/issues/1417) by @frances-h
* SettingWithCopyWarning when conditional sampling - [#1436](https://github.com/sdv-dev/SDV/issues/1436) by @pvk-developer
* HMASynthesizer is modeling child tables - Issue [#1442](https://github.com/sdv-dev/SDV/issues/1442) by @pvk-developer
* ValueError when sampling PII columns - Issue [#1445](https://github.com/sdv-dev/SDV/issues/1445) by @pvk-developer

### Internal

* Add BaseHierarchicalSampler Mixin - Issue [#1394](https://github.com/sdv-dev/SDV/issues/1394) by @frances-h
* Add BaseIndependentSampler Mixin - Issue [#1395](https://github.com/sdv-dev/SDV/issues/1395) by @frances-h
* Synthesizers created twice during HMA init - Issue [#1418](https://github.com/sdv-dev/SDV/issues/1418) by @frances-h
* Get rid of unnecessary methods for single table sampling - Issue [#1430](https://github.com/sdv-dev/SDV/issues/1430) by @amontanez24
* Detect all addons from top level __init__ - PR [#1453](https://github.com/sdv-dev/SDV/pull/1453) by @frances-h

### Maintenance

* Upgrade to torch 2.0 - Issue [#1365](https://github.com/sdv-dev/SDV/issues/1365) by @fealho
* During fit, there is a FutureWarning (due to RDT 1.5.0) - Issue [#1456](https://github.com/sdv-dev/SDV/issues/1456) by @amontanez24

## 1.1.0 - 2023-05-10

This release adds a new initialization parameter to synthesizers called `locales` that allows users to set the locales to use for all columns that have a locale based `sdtype` (eg. `address` or `phone_number`). Additionally, it adds support for Pandas 2.0!

Multiple enhancements were made to improve the performance of data and metadata validation in synthesizers. The `Inequality` constraint was improved to be able to generate more scenarios of data concerning the presence of NaNs. Finally, many warnings have been resolved.

### New Features

* Add add-on detection for new constraints - Issue [#1397](https://github.com/sdv-dev/SDV/issues/1397) by @frances-h
* Add add-on detection for multi and single table synthesizers - Issue [#1385](https://github.com/sdv-dev/SDV/issues/1385) by @frances-h
* Setting a locale for all my anonymized (PII) columns - Issue [#1371](https://github.com/sdv-dev/SDV/issues/1371) by @frances-h

### Bugs Fixed

* Skip checking for Faker function if its a default sdtype - PR [#1410](https://github.com/sdv-dev/SDV/pull/1410) by @pvk-developer
* Inequality constraint does not produce all possibilities of missing values - Issue [#1392](https://github.com/sdv-dev/SDV/issues/1392) by @R-Palazzo
* Deprecated locale warning - Issue [#1400](https://github.com/sdv-dev/SDV/issues/1400) by @frances-h

### Maintenance

* Use cached Faker instance to discover if an sdtype is a Faker function. - Issue [#1405](https://github.com/sdv-dev/SDV/issues/1405) by @pvk-developer
* Upgrade to pandas 2.0 - Issue [#1366](https://github.com/sdv-dev/SDV/issues/1366) by @pvk-developer

### Internal

* Refactor Multi Table Modeling - Issue [#1387](https://github.com/sdv-dev/SDV/issues/1387) by @pvk-developer
* PytestConfigWarning: Unknown config option: collect_ignore - Issue [#1376](https://github.com/sdv-dev/SDV/issues/1376) by @amontanez24
* Pandas FutureWarning: Could not cast to int64 - Issue [#1357](https://github.com/sdv-dev/SDV/issues/1357) by @R-Palazzo
* RuntimeWarning: invalid value encountered in cast. - Issue [#1369](https://github.com/sdv-dev/SDV/issues/1369) by @amontanez24

## 1.0.1 - 2023-04-20

This release improves the `load_custom_constraint_classes` method by removing the `table_name` parameter and just loading the constraint
for all tables instead. It also improves some error messages as well as removes some of the warnings that have been surfacing.

Support for sdtypes is enhanced by resolving a bug that was incorrecttly specifying Faker functions for some of them. Support for datetime formats has also been improved. Finally, the `path` argument in some `save` and `load` methods was changed to `filepath` for consistency.

### New Features

* Method load_custom_constraint_classes does not need table_name parameter - Issue [#1354](https://github.com/sdv-dev/SDV/issues/1354) by @R-Palazzo
* Improve error message for invalid primary keys - Issue [#1341](https://github.com/sdv-dev/SDV/issues/1341) by @R-Palazzo
* Add functionality to find version add-on - Issue [#1309](https://github.com/sdv-dev/SDV/issues/1309) by @frances-h

### Bugs Fixed

* Certain sdtypes cause Faker to raise error - Issue [#1346](https://github.com/sdv-dev/SDV/issues/1346) by @frances-h
* Change path to filepath for load and save methods - Issue [#1352](https://github.com/sdv-dev/SDV/issues/1352) by @fealho
* Some datetime formats cause InvalidDataError, even if the datetime matches the format - Issue [#1136](https://github.com/sdv-dev/SDV/issues/1136) by @amontanez24

### Internal

* Inequality constraint raises RuntimeWarning (invalid value encountered in log) - Issue [#1275](https://github.com/sdv-dev/SDV/issues/1275) by @frances-h
* Pandas FutureWarning: Default dtype for Empty Series will be 'object' - Issue [#1355](https://github.com/sdv-dev/SDV/issues/1355) by @R-Palazzo
* Pandas FutureWarning: Length 1 tuple will be returned - Issue [#1356](https://github.com/sdv-dev/SDV/issues/1356) by @R-Palazzo

## 1.0.0 - 2023-03-28

This is a major release that introduces a new API to the `SDV` aimed at streamlining the process of synthetic data generation! To achieve this, this release includes the addition of several large features.

### Metadata

Some of the most notable additions are the new `SingleTableMetadata` and `MultiTableMetadata` classes. These classes enable a number of features that make it easier to synthesize your data correctly such as:

* Automatic data detection - Calling `metadata.detect_from_dataframe()` or `metadata.detect_from_csv()` will populate the metadata autonomously with values it thinks represent the data.
* Easy updating - Once an instance of the metadata is created, values can be easily updated using a number of methods defined in the API. For more info, view the [docs](https://docs.sdv.dev/sdv/single-table-data/data-preparation/single-table-metadata-api).
* Metadata validation - Calling `metadata.validate()` will return a report of any invalid definitions in the metadata specification.
* Upgrading - Users with the previous metadata format can easily update to the new specification using the `upgrade_metadata()` method.
* Saving and loading - The metadata itself can easily be saved to a json file and loaded back up later.

### Class and Module Names

Another major change is the renaming of our core modeling classes and modules. The name changes are meant to highlight the difference between the underlying machine learning models, and the objects responsible for the end-to-end workflow of generating synthetic data. The main name changes are as follows:
* `tabular` -> `single_table`
* `relational` -> `multi_table`
* `timeseries` -> `sequential`
* `BaseTabularModel` -> `BaseSingleTableSynthesizer`
* `GaussianCopula` -> `GaussianCopulaSynthesizer`
* `CTGAN` -> `CTGANSynthesizer`
* `TVAE` -> `TVAESynthesizer`
* `CopulaGan` -> `CopulaGANSynthesizer`
* `PAR` -> `PARSynthesizer`
* `HMA1` -> `HMASynthesizer`

In `SDV` 1.0, synthesizers are classes that take in metadata and handle data preprocessing, model training and model sampling. This is similar to the previous `BaseTabularModel` in `SDV` <1.0.

### Synthetic Data Workflow

`Synthesizers` in `SDV` 1.0 define a clear workflow for generating synthetic data.
1. Synthesizers are initialized with a metadata class.
2. They can then be used to transform the data and apply constraints using the `synthesizer.preprocess()` method. This step also validates that the data matches the provided metadata to avoid errors in fitting or sampling.
3. The processed data can then be fed into the underlying machine learning model using `synthesizer.fit_processed_data()`. (Alternatively, data can be preprocessed and fit to the model using `synthesizer.fit()`.)
4. Data can then be sampled using `synthesizer.sample()`.

Each synthesizer class also provides a series of methods to help users customize the transformations their data goes through. Read more about that [here](https://docs.sdv.dev/sdv/single-table-data/modeling/synthetic-data-workflow/transform-and-anonymize).

Notice that the preprocessing and model fitting steps can now be separated. This can be helpful if preprocessing is time consuming or if the data has been processed externally.

### Other Highly Requested Features

Another major addition is control over randomization. In `SDV` <1.0, users could set a seed to control the randomization for only some columns. In `SDV` 1.0, randomization is controlled for all columns. Every new call to sample generates new data, but the synthesizer's seed can be reset to the original state using `synthesizer.reset_randomization()`, enabling reproducibility.

`SDV 1.0` adds accessibility and transparency into the transformers used for preprocessing and underlying machine learning models.
* Using the `synthesizer.get_transformers()` method, you can access the transformers used to preprocess each column and view their properties. This can be useful for debugging and accessing privacy information like mappings used to mask data.
* Distribution parameters learned by copula models can be accessed using the `synthesizer.get_learned_distributions()` method.

PII handling is improved by the following features:
* Primary keys can be set to natural sdtypes (eg. SSN, email, name). Previously they could only be numerical or text.
* The `PseudoAnonymizedFaker` can be used to provide consistent mapping to PII columns. As mentioned before, the mapping itself can be accessed by viewing the transformers for the column using `synthesizer.get_transformers()`.
* A bug causing PII columns to slow down modeling is patched.

Finally, the synthetic data can now be easily evaluated using the `evaluate_quality()` and `run_diagnostic()` methods. The data can be compared visually to the actual data using the `get_column_plot()` and `get_column_pair_plot()` methods. For more info on how to visualize or interpret the synthetic data evaluation, read the docs [here](https://docs.sdv.dev/sdv/single-table-data/evaluation).

### Issues Resolved

#### New Features

* Change auto_assign_transformers to handle id types - Issue [#1325](https://github.com/sdv-dev/SDV/issues/1325) by @pvk-developer
* Change 'text' sdtype to 'id' - Issue [#1324](https://github.com/sdv-dev/SDV/issues/1324) by @frances-h
* In `upgrade_metadata`, return the object instead of writing it to a JSON file - Issue [#1319](https://github.com/sdv-dev/SDV/issues/1319) by @frances-h
* In `upgrade_metadata` index primary keys should be converted to `text` - Issue [#1318](https://github.com/sdv-dev/SDV/issues/1318) by @amontanez24
* Add `load_from_dict` to SingleTableMetadata and MultiTableMetadata - Issue [#1314](https://github.com/sdv-dev/SDV/issues/1314) by @amontanez24
* Throw a `SynthesizerInputError` if `FixedCombinations` constraint is applied to a column that is not `boolean` or `categorical` - Issue [#1306](https://github.com/sdv-dev/SDV/issues/1306) by @frances-h
* Missing `save` and `load` methods for `HMASynthesizer` - Issue [#1262](https://github.com/sdv-dev/SDV/issues/1262) by @amontanez24
* Better input validation when creating single and multi table synthesizers - Issue [#1242](https://github.com/sdv-dev/SDV/issues/1242) by @fealho
* Better input validation on `HMASynthesizer.sample` - Issue [#1241](https://github.com/sdv-dev/SDV/issues/1241) by @R-Palazzo
* Validate that relationship must be between a `primary key` and `foreign key` - Issue [#1236](https://github.com/sdv-dev/SDV/issues/1236) by @fealho
* Improve `update_column` validation for `pii` attribute - Issue [#1226](https://github.com/sdv-dev/SDV/issues/1226) by @pvk-developer
* Order the output of `get_transformers()` based on the metadata - Issue [#1222](https://github.com/sdv-dev/SDV/issues/1222) by @pvk-developer
* Log if any `numerical_distributions` will not be applied - Issue [#1212](https://github.com/sdv-dev/SDV/issues/1212) by @fealho
* Improve error handling for `GaussianCopulaSynthesizer`: `numerical_distributions` - Issue [#1211](https://github.com/sdv-dev/SDV/issues/1211) by @fealho
* Improve error handling when validating `constraints` - Issue [#1210](https://github.com/sdv-dev/SDV/issues/1210) by @fealho
* Add `fake_companies` demo - Issue [#1209](https://github.com/sdv-dev/SDV/issues/1209) by @amontanez24
* Allow me to create a custom constraint class and use it in the same file - Issue [#1205](https://github.com/sdv-dev/SDV/issues/1205) by @amontanez24
* Sampling should reset after retraining the model - Issue [#1201](https://github.com/sdv-dev/SDV/issues/1201) by @pvk-developer
* Change function name `HMASynthesizer.update_table_parameters` --> `set_table_parameters` - Issue [#1200](https://github.com/sdv-dev/SDV/issues/1200) by @pvk-developer
* Add `get_info` method to synthesizers - Issue [#1199](https://github.com/sdv-dev/SDV/issues/1199) by @fealho
* Add evaluation methods to synthesizer - Issue [#1190](https://github.com/sdv-dev/SDV/issues/1190) by @fealho
* Update `evaluate.py` to work with the new `metadata` - Issue [#1186](https://github.com/sdv-dev/SDV/issues/1186) by @fealho
* Remove old code - Issue [#1181](https://github.com/sdv-dev/SDV/issues/1181) by @pvk-developer
* Drop support for python 3.6 and add support for 3.10 - Issue [#1176](https://github.com/sdv-dev/SDV/issues/1176) by @fealho
* Add constraint methods to MultiTableSynthesizers - Issue [#1171](https://github.com/sdv-dev/SDV/issues/1171) by @fealho
* Update custom constraint workflow - Issue [#1169](https://github.com/sdv-dev/SDV/issues/1169) by @pvk-developer
* Add get_constraints method to synthesizers - Issue [#1168](https://github.com/sdv-dev/SDV/issues/1168) by @pvk-developer
* Migrate adding and validating constraints to BaseSynthesizer - Issue [#1163](https://github.com/sdv-dev/SDV/issues/1163) by @pvk-developer
* Change metadata `"SCHEMA_VERSION"` --> `"METADATA_SPEC_VERSION"` - Issue [#1139](https://github.com/sdv-dev/SDV/issues/1139) by @amontanez24
* Add ability to reset random sampling - Issue [#1130](https://github.com/sdv-dev/SDV/issues/1130) by @pvk-developer
* Add get_available_demos - Issue [#1129](https://github.com/sdv-dev/SDV/issues/1129) by @fealho
* Add demo loading functionality - Issue [#1128](https://github.com/sdv-dev/SDV/issues/1128) by @fealho
* Use logging instead of printing in detect methods - Issue [#1107](https://github.com/sdv-dev/SDV/issues/1107) by @fealho
* Add save and load methods to synthesizers - Issue [#1106](https://github.com/sdv-dev/SDV/issues/1106) by @pvk-developer
* Add sampling methods to PARSynthesizer - Issue [#1083](https://github.com/sdv-dev/SDV/issues/1083) by @amontanez24
* Add transformer methods to PARSynthesizer - Issue [#1082](https://github.com/sdv-dev/SDV/issues/1082) by @fealho
* Add validate to PARSynthesizer - Issue [#1081](https://github.com/sdv-dev/SDV/issues/1081) by @amontanez24
* Add preprocess and fit methods to PARSynthesizer - Issue [#1080](https://github.com/sdv-dev/SDV/issues/1080) by @amontanez24
* Create SingleTablePreset - Issue [#1079](https://github.com/sdv-dev/SDV/issues/1079) by @amontanez24
* Add sample method to multi-table synthesizers - Issue [#1078](https://github.com/sdv-dev/SDV/issues/1078) by @pvk-developer
* Add get_learned_distributions method to synthesizers - Issue [#1075](https://github.com/sdv-dev/SDV/issues/1075) by @pvk-developer
* Add preprocess and fit methods to multi-table synthesizers - Issue [#1074](https://github.com/sdv-dev/SDV/issues/1074) by @pvk-developer
* Add transformer related methods to BaseMultiTableSynthesizer - Issue [#1072](https://github.com/sdv-dev/SDV/issues/1072) by @fealho
* Add validate method to `BaseMultiTableSynthesizer` - Issue [#1071](https://github.com/sdv-dev/SDV/issues/1071) by @pvk-developer
* Create BaseMultiTableSynthesizer and HMASynthesizer classes - Issue [#1070](https://github.com/sdv-dev/SDV/issues/1070) by @pvk-developer
* Create PARSynthesizer - Issue [#1055](https://github.com/sdv-dev/SDV/issues/1055) by @amontanez24
* Raise an error if an invalid sdtype is provided to the metadata - Issue [#1042](https://github.com/sdv-dev/SDV/issues/1042) by @amontanez24
* Only allow datetime and numerical sdtypes to be set as the sequence index - Issue [#1030](https://github.com/sdv-dev/SDV/issues/1030) by @amontanez24
* Change set_alternate_keys to add_alternate_keys and add error handling - Issue [#1029](https://github.com/sdv-dev/SDV/issues/1029) by @amontanez24
* Create `MultiTableMetadata.add_table` method - Issue [#1024](https://github.com/sdv-dev/SDV/issues/1024) by @amontanez24
* Add update_transformers to synthesizers - Issue [#1021](https://github.com/sdv-dev/SDV/issues/1021) by @fealho
* Add assign_transformers and get_transformers methods to synthesizers - Issue [#1020](https://github.com/sdv-dev/SDV/issues/1020) by @pvk-developer
* Add fit and fit_processed_data methods to synthesizers - Issue [#1019](https://github.com/sdv-dev/SDV/issues/1019) by @pvk-developer
* Add preprocess method to synthesizers - Issue [#1018](https://github.com/sdv-dev/SDV/issues/1018) by @pvk-developer
* Add sampling to synthesizer classes - Issue [#1015](https://github.com/sdv-dev/SDV/issues/1015) by @pvk-developer
* Add validate method to synthesizer - Issue [#1014](https://github.com/sdv-dev/SDV/issues/1014) by @fealho
* Create GaussianCopula, CTGAN, TVAE and CopulaGAN synthesizer classes - Issue [#1013](https://github.com/sdv-dev/SDV/issues/1013) by @pvk-developer
* Create BaseSynthesizer class - Issue [#1012](https://github.com/sdv-dev/SDV/issues/1012) by @pvk-developer
* Add constraint conversion to upgrade_metadata - Issue [#1005](https://github.com/sdv-dev/SDV/issues/1005) by @amontanez24
* Add method to generate keys to DataProcessor - Issue [#994](https://github.com/sdv-dev/SDV/issues/994) by @pvk-developer
* Create formatter - Issue [#970](https://github.com/sdv-dev/SDV/issues/970) by @fealho
* Create a utility to load multiple CSV files at once - Issue [#969](https://github.com/sdv-dev/SDV/issues/969) by @amontanez24
* Create a utility to convert old --> new metadata format - Issue [#966](https://github.com/sdv-dev/SDV/issues/966) by @amontanez24
* Add validation check that `primary_key`, `alternate_keys` and `sequence_key` cannot be sdtype categorical - Issue [#963](https://github.com/sdv-dev/SDV/issues/963) by @fealho
* Add anonymization to DataProcessor - Issue [#950](https://github.com/sdv-dev/SDV/issues/950) by @pvk-developer
* Add utility methods to DataProcessor - Issue [#948](https://github.com/sdv-dev/SDV/issues/948) by @fealho
* Add fit, transform and reverse_transform to DataProcessor - Issue [#947](https://github.com/sdv-dev/SDV/issues/947) by @amontanez24
* Create DataProcessor class - Issue [#946](https://github.com/sdv-dev/SDV/issues/946) by @amontanez24
* Add add_constraint method to MultiTableMetadata - Issue [#895](https://github.com/sdv-dev/SDV/issues/895) by @amontanez24
* Add key related methods to MultiTableMetadata - Issue [#894](https://github.com/sdv-dev/SDV/issues/894) by @fealho
* Add update_column and add_column methods to MultiTableMetadata - Issue [#893](https://github.com/sdv-dev/SDV/issues/893) by @amontanez24
* Add detect methods to MultiTableMetadata - Issue [#892](https://github.com/sdv-dev/SDV/issues/892) by @amontanez24
* Add load_from_json and save_to_json methods to the MultiTableMetadata - Issue [#891](https://github.com/sdv-dev/SDV/issues/891) by @fealho
* Add add_relationship method to MultiTableMetadata - Issue [#890](https://github.com/sdv-dev/SDV/issues/890) by @pvk-developer
* Add validate method to MultiTableMetadata - Issue [#888](https://github.com/sdv-dev/SDV/issues/888) by @pvk-developer
* Add visualize method to MultiTableMetadata class - Issue [#884](https://github.com/sdv-dev/SDV/issues/884) by @amontanez24
* Create MultiTableMetadata class - Issue [#883](https://github.com/sdv-dev/SDV/issues/883) by @pvk-developer
* Add add_constraint method to SingleTableMetadata - Issue [#881](https://github.com/sdv-dev/SDV/issues/881) by @amontanez24
* Add key related methods to SingleTableMetadata - Issue [#880](https://github.com/sdv-dev/SDV/issues/880) by @fealho
* Add validate method to SingleTableMetadata - Issue [#879](https://github.com/sdv-dev/SDV/issues/879) by @fealho
* Add _validate_inputs class method to each constraint - Issue [#878](https://github.com/sdv-dev/SDV/issues/878) by @fealho
* Add update_column and add_column methods to SingleTableMetadata - Issue [#877](https://github.com/sdv-dev/SDV/issues/877) by @pvk-developer
* Add detect methods to SingleTableMetadata - Issue [#876](https://github.com/sdv-dev/SDV/issues/876) by @pvk-developer
* Add load_from_json and save_to_json methods to SingleTableMetadata - Issue [#874](https://github.com/sdv-dev/SDV/issues/874) by @pvk-developer
* Create SingleTableMetadata class - Issue [#873](https://github.com/sdv-dev/SDV/issues/873) by @pvk-developer

#### Bugs Fixed

* In `upgrade_metadata`, PII values are being converted to generic categorical columns - Issue [#1317](https://github.com/sdv-dev/SDV/issues/1317) by @frances-h
* `PARSynthesizer` is missing `save` and `load` methods - Issue [#1289](https://github.com/sdv-dev/SDV/issues/1289) by @amontanez24
* Confusing warning when updating transformers - Issue [#1272](https://github.com/sdv-dev/SDV/issues/1272) by @frances-h
* When adding constraints, `auto_assign_transformers` is showing columns that should no longer exist - Issue [#1260](https://github.com/sdv-dev/SDV/issues/1260) by @pvk-developer
* Cannot fit twice if I modify transformers: `ValueError: There are non-numerical values in your data.` - Issue [#1259](https://github.com/sdv-dev/SDV/issues/1259) by @frances-h
* Cannot fit twice if I add constraints: `ValueError: There are non-numerical values in your data.` - Issue [#1258](https://github.com/sdv-dev/SDV/issues/1258) by @frances-h
* `HMASynthesizer` errors out when fitting a dataset that has a table which holds primary key and foreign keys only - Issue [#1257](https://github.com/sdv-dev/SDV/issues/1257) by @pvk-developer
* Change ValueErrors to InvalidMetadataErrors - Issue [#1251](https://github.com/sdv-dev/SDV/issues/1251) by @frances-h
* Multi-table should show foreign key transformers as None - Issue [#1249](https://github.com/sdv-dev/SDV/issues/1249) by @frances-h
* Cannot use `HMASynthesizer.fit_processed_data` more than once (`KeyError`) - Issue [#1240](https://github.com/sdv-dev/SDV/issues/1240) by @frances-h
* Function `get_available_demos` crashes if a dataset's `num-tables` or `size-MB` cannot be found - Issue [#1215](https://github.com/sdv-dev/SDV/issues/1215) by @amontanez24
* Cannot supply a natural key to `HMASynthesizer` (where `sdtype` is custom): Error in `sample` - Issue [#1214](https://github.com/sdv-dev/SDV/issues/1214) by @pvk-developer
* Unable to sample when using a `PseudoAnonymizedFaker` - Issue [#1207](https://github.com/sdv-dev/SDV/issues/1207) by @pvk-developer
* Incorrect `sdtype` specified in demo dataset `student_placements_pii` - Issue [#1206](https://github.com/sdv-dev/SDV/issues/1206) by @amontanez24
* Auto assigned transformers for datetime columns don't have the right parameters - Issue [#1204](https://github.com/sdv-dev/SDV/issues/1204) by @pvk-developer
* Cannot apply `Inequality` constraint on demo dataset's datetime columns - Issue [#1203](https://github.com/sdv-dev/SDV/issues/1203) by @pvk-developer
* pii should not be required to auto-assign faker transformers - Issue [#1194](https://github.com/sdv-dev/SDV/issues/1194) by @pvk-developer
* Misc. bug fixes for SDV 1.0.0 - Issue [#1193](https://github.com/sdv-dev/SDV/issues/1193) by @pvk-developer
* Small bug fixes in demo module - Issue [#1192](https://github.com/sdv-dev/SDV/issues/1192) by @pvk-developer
* Foreign Keys are added as Alternate Keys when upgrading - Issue [#1143](https://github.com/sdv-dev/SDV/issues/1143) by @pvk-developer
* Alternate keys not unique when assigned to a semantic type - Issue [#1111](https://github.com/sdv-dev/SDV/issues/1111) by @pvk-developer
* Synthesizer errors if column is semantic type and pii is False - Issue [#1110](https://github.com/sdv-dev/SDV/issues/1110) by @fealho
* Sampled values not unique if primary key is numerical - Issue [#1109](https://github.com/sdv-dev/SDV/issues/1109) by @pvk-developer
* Validate not called during synthesizer creation - Issue [#1105](https://github.com/sdv-dev/SDV/issues/1105) by @pvk-developer
* SingleTableSynthesizer fit doesn't update rounding - Issue [#1104](https://github.com/sdv-dev/SDV/issues/1104) by @amontanez24
* Method `auto_assign_tranformers` always sets `enforce_min_max_values=True` - Issue [#1095](https://github.com/sdv-dev/SDV/issues/1095) by @fealho
* Sampled context columns in PAR must be in the same order - Issue [#1052](https://github.com/sdv-dev/SDV/issues/1052) by @amontanez24
* Incorrect schema version printing during detect_table_from_dataframe - Issue [#1038](https://github.com/sdv-dev/SDV/issues/1038) by @amontanez24
* Same relationship can be added twice to MultiTableMetadata - Issue [#1031](https://github.com/sdv-dev/SDV/issues/1031) by @amontanez24
* Miscellaneous metadata bugs - Issue [#1026](https://github.com/sdv-dev/SDV/issues/1026) by @amontanez24

#### Maintenance

* SDV Package Maintenance Updates - Issue [#1140](https://github.com/sdv-dev/SDV/issues/1140) by @amontanez24

#### Internal

* Add integration tests for 'Synthesize Sequences' demo - Issue [#1295](https://github.com/sdv-dev/SDV/issues/1295) by @pvk-developer
* Add integration tests for 'Adding Constraints' demo - Issue [#1280](https://github.com/sdv-dev/SDV/issues/1280) by @pvk-developer
* Add integration tests to the 'Use Your Own Data' demo - Issue [#1278](https://github.com/sdv-dev/SDV/issues/1278) by @frances-h
* Add integration tests for 'Synthesize Multi Tables' demo - Issue [#1277](https://github.com/sdv-dev/SDV/issues/1277) by @pvk-developer
* Add integration tests for 'Synthesize a Table' demo - Issue [#1276](https://github.com/sdv-dev/SDV/issues/1276) by @frances-h
* Update `get_available_demos` tests - Issue [#1247](https://github.com/sdv-dev/SDV/issues/1247) by @fealho
* Make private attributes public in the metadata - Issue [#1245](https://github.com/sdv-dev/SDV/issues/1245) by @fealho

## 0.18.0 - 2023-01-24

This release adds suppport for Python 3.10 and drops support for 3.6.

### Maintenance

* Drop support for python 3.6 - Issue [#1177](https://github.com/sdv-dev/SDV/issues/1177) by @amontanez24
* Support for python 3.10 - Issue [#939](https://github.com/sdv-dev/SDV/issues/939) by @amontanez24
* Support Python >=3.10,<4 - Issue [#1000](https://github.com/sdv-dev/SDV/issues/1000) by @amontanez24

## 0.17.2 - 2022-12-08

This release fixes a bug in the demo module related to loading the demo data with constraints. It also adds a name to the demo datasets. Finally, it bumps the version of `SDMetrics` used.

### Maintenance

* Upgrade SDMetrics requirement to 0.8.0 - Issue [#1125](https://github.com/sdv-dev/SDV/issues/1125) by @katxiao

### New Features

* Provide a name for the default demo datasets - Issue [#1124](https://github.com/sdv-dev/SDV/issues/1124) by @amontanez24

### Bugs Fixed

* Cannot load_tabular_demo with metadata - Issue [#1123](https://github.com/sdv-dev/SDV/issues/1123) by @amontanez24

## 0.17.1 - 2022-09-29

This release bumps the dependency requirements to use the latest version of `SDMetrics`.

### Maintenance

* Patch release: Bump required version for SDMetrics - Issue [#1010](https://github.com/sdv-dev/SDV/issues/1010) by @katxiao

## 0.17.0 - 2022-09-09

This release updates the code to use RDT version 1.2.0 and greater, so that those new features are now available in SDV. This changes the transformers that are available in SDV models to be those that are in RDT version 1.2.0. As a result, some arguments for initializing models have changed.

Additionally, this release fixes bugs related to loading models with custom constraints. It also fixes a bug that added `NaNs` to the index of sampled data when using `sample_remaining_columns`.

### Bugs Fixed

* Incorrect rounding in Custom Constraint example - Issue [#941](https://github.com/sdv-dev/SDV/issues/941) by @amontanez24
* Can't save the model if use the custom constraint - Issue [#928](https://github.com/sdv-dev/SDV/issues/928) by @pvk-developer
* User Guide code fixes - Issue [#983](https://github.com/sdv-dev/SDV/issues/983) by @amontanez24
* Index contains NaNs when using sample_remaining_columns - Issue [#985](https://github.com/sdv-dev/SDV/issues/985) by @amontanez24
* Cannot sample after loading a model with custom constraints: TypeError - Issue [#984](https://github.com/sdv-dev/SDV/issues/984) by @pvk-developer
* Set HyperTransformer config manually, based on Metadata if given - Issue [#982](https://github.com/sdv-dev/SDV/issues/982) by @pvk-developer

### New Features

* Change default metrics for evaluate - Issue [#949](https://github.com/sdv-dev/SDV/issues/949) by @fealho

### Maintenance

* Update the RDT version to 1.0 - Issue [#897](https://github.com/sdv-dev/SDV/issues/897) by @pvk-developer

## 0.16.0 - 2022-07-21

This release brings user friendly improvements and bug fixes on the `SDV` constraints, to help
users generate their synthetic data easily.

Some predefined constraints have been renamed and redefined to be more user friendly & consistent.
The custom constraint API has also been updated for usability. The SDV now automatically determines
the best `handling_strategy` to use for each constraint, attempting `transform` by default and
falling back to `reject_sampling` otherwise. The `handling_strategy` parameters are no longer
included in the API.

Finally, this version of `SDV` also unifies the parameters for all sampling related methods for
all models (including TabularPreset).

### Changes to Constraints

* `GreatherThan` constraint is now separated in two new constraints: `Inequality`, which is
  intended to be used between two columns, and `ScalarInequality`, which is intended to be used
  between a column and a scalar.

* `Between` constraint is now separated in two new constraints: `Range`, which is intended to
  be used between three columns, and `ScalarRange`, which is intended to be used between a column
  and low and high scalar values.

* `FixedIncrements` a new constraint that makes the data increment by a certain value.
* New `create_custom_constraint` function available to create custom constraints.

### Removed Constraints
* `Rounding` Rounding is automatically being handled by the ``rdt.HyperTransformer``.
* `ColumnFormula` the `create_custom_constraint` takes place over this one and allows more
  advanced usage for the end users.

### New Features

* Improve error message for invalid constraints - Issue [#801](https://github.com/sdv-dev/SDV/issues/801) by @fealho
* Numerical Instability in Constrained GaussianCopula - Issue [#806](https://github.com/sdv-dev/SDV/issues/806) by @fealho
* Unify sampling params for reject sampling - Issue [#809](https://github.com/sdv-dev/SDV/issues/809) by @amontanez24
* Split `GreaterThan` constraint into `Inequality` and `ScalarInequality` - Issue [#814](https://github.com/sdv-dev/SDV/issues/814) by @fealho
* Split `Between` constraint into `Range` and `ScalarRange` - Issue [#815](https://github.com/sdv-dev/SDV/issues/815) @pvk-developer
* Change `columns` to `column_names` in `OneHotEncoding` and `Unique` constraints - Issue [#816](https://github.com/sdv-dev/SDV/issues/816) by @amontanez24
* Update columns parameter in `Positive` and `Negative` constraint - Issue [#817](https://github.com/sdv-dev/SDV/issues/817) by @fealho
* Create `FixedIncrements` constraint - Issue [#818](https://github.com/sdv-dev/SDV/issues/818) by @amontanez24
* Improve datetime handling in `ScalarInequality` and `ScalarRange` constraints - Issue [#819](https://github.com/sdv-dev/SDV/issues/819) by @pvk-developer
* Support strict boundaries even when transform strategy is used - Issue [#820](https://github.com/sdv-dev/SDV/issues/820) by @fealho
* Add `create_custom_constraint` factory method - Issue [#836](https://github.com/sdv-dev/SDV/issues/836) by @fealho

### Internal Improvements
* Remove `handling_strategy` parameter - Issue [#833](https://github.com/sdv-dev/SDV/issues/833) by @amontanez24
* Remove `fit_columns_model` parameter - Issue [#834](https://github.com/sdv-dev/SDV/issues/834) by @pvk-developer
* Remove the `ColumnFormula` constraint - Issue [#837](https://github.com/sdv-dev/SDV/issues/837) by @amontanez24
* Move `table_data.copy` to base class of constraints - Issue [#845](https://github.com/sdv-dev/SDV/issues/845) by @fealho

### Bugs Fixed
* Numerical Instability in Constrained GaussianCopula - Issue [#801](https://github.com/sdv-dev/SDV/issues/801) by @tlranda and @fealho
* Fix error message for `FixedIncrements` - Issue [#865](https://github.com/sdv-dev/SDV/issues/865) by @pvk-developer
* Fix constraints with conditional sampling - Issue [#866](https://github.com/sdv-dev/SDV/issues/866) by @amontanez24
* Fix error message in `ScalarInequality` - Issue [#868](https://github.com/sdv-dev/SDV/issues/868) by @pvk-developer
* Cannot use `max_tries_per_batch` on sample: `TypeError: sample() got an unexpected keyword argument 'max_tries_per_batch'` - Issue [#885](https://github.com/sdv-dev/SDV/issues/885) by @amontanez24
* Conditional sampling + batch size: `ValueError: Length of values (1) does not match length of index (5)` - Issue [#886](https://github.com/sdv-dev/SDV/issues/886) by @amontanez24
* `TabularPreset` doesn't support new sampling parameters - Issue [#887](https://github.com/sdv-dev/SDV/issues/887) by @fealho
* Conditional Sampling: `batch_size` is being set to `None` by default? - Issue [#889](https://github.com/sdv-dev/SDV/issues/889) by @amontanez24
* Conditional sampling using GaussianCopula inefficient when categories are noised - Issue [#910](https://github.com/sdv-dev/SDV/issues/910) by @amontanez24

### Documentation Changes
* Show the `API` for `TabularPreset` models - Issue [#854](https://github.com/sdv-dev/SDV/issues/854) by @katxiao
* Update handling constraints doc - Pull Request [#856](https://github.com/sdv-dev/SDV/issues/856) by @amontanez24
* Update custom costraints documentation - Pull Request [#857](https://github.com/sdv-dev/SDV/issues/857) by @pvk-developer

## 0.15.0 - 2022-05-25

This release improves the speed of the `GaussianCopula` model by removing logic that previously searched for the appropriate distribution to
use. It also fixes a bug that was happening when conditional sampling was used with the `TabularPreset`.

The rest of the release focuses on making changes to improve constraints including changing the `UniqueCombinations` constraint to `FixedCombinations`,
making the `Unique` constraint work with missing values and erroring when null values are seen in the `OneHotEncoding` constraint.

### New Features
* Silence warnings coming from univariate fit in copulas - Issue [#769](https://github.com/sdv-dev/SDV/issues/769) by @pvk-developer
* Remove parameters related to distribution search and change default - Issue [#767](https://github.com/sdv-dev/SDV/issues/767) by @fealho
* Update the UniqueCombinations constraint - Issue [#793](https://github.com/sdv-dev/SDV/issues/793) by @fealho
* Make Unique constraint works with nans - Issue [#797](https://github.com/sdv-dev/SDV/issues/797) by @fealho
* Error out if nans in OneHotEncoding - Issue [#800](https://github.com/sdv-dev/SDV/issues/800) by @amontanez24

### Bugs Fixed
* Unable to sample conditionally in Tabular_Preset model - Issue [#796](https://github.com/sdv-dev/SDV/issues/796) by @katxiao

### Documentation Changes
* Support GPU computing and progress track? - Issue [#478](https://github.com/sdv-dev/SDV/issues/478) by @fealho

## 0.14.1 - 2022-05-03

This release adds a `TabularPreset`, available in the `sdv.lite` module, which allows users to easily optimize a tabular model for speed.
In this release, we also include bug fixes for sampling with conditions, an unresolved warning, and setting field distributions. Finally,
we include documentation updates for sampling and the new `TabularPreset`.

### Bugs Fixed
* Sampling with conditions={column: 0.0} for float columns doesn't work - Issue [#525](https://github.com/sdv-dev/SDV/issues/525) by @shlomihod and @tssbas
* resolved FutureWarning with Pandas replaced append by concat - Issue [#759](https://github.com/sdv-dev/SDV/issues/759) by @Deathn0t
* Field distributions bug in CopulaGAN - Issue [#747](https://github.com/sdv-dev/SDV/issues/747) by @katxiao
* Field distributions bug in GaussianCopula - Issue [#746](https://github.com/sdv-dev/SDV/issues/746) by @katxiao

### New Features
* Set default transformer to categorical_fuzzy - Issue [#768](https://github.com/sdv-dev/SDV/issues/768) by @amontanez24
* Model nulls normally when tabular preset has constraints - Issue [#764](https://github.com/sdv-dev/SDV/issues/764) by @katxiao
* Don't modify my metadata object - Issue [#754](https://github.com/sdv-dev/SDV/issues/754) by @amontanez24
* Presets should be able to handle constraints - Issue [#753](https://github.com/sdv-dev/SDV/issues/753) by @katxiao
* Change preset optimize_for --> name - Issue [#749](https://github.com/sdv-dev/SDV/issues/749) by @katxiao
* Create a speed optimized Preset - Issue [#716](https://github.com/sdv-dev/SDV/issues/716) by @katxiao

### Documentation Changes
* Add tabular preset docs - Issue [#777](https://github.com/sdv-dev/SDV/issues/777) by @katxiao
* sdv.sampling module is missing from the API - Issue [#740](https://github.com/sdv-dev/SDV/issues/740) by @katxiao

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
