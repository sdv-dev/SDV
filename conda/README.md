## Instructions

These are instructions to deploy the latest version of **SDV** to [conda](https://docs.conda.io/en/latest/).
It should be done after every new release.

## Update the recipe
Prior to making the release on PyPI, you should update the meta.yaml to reflect any changes in the dependencies.
Note that you do not need to edit the version number as that is managed by bumpversion.

## Make the PyPI release
Follow the standard release instructions to make a PyPI release. Then, return here to make the conda release.

## Build a package
As part of the PyPI release, you will have updated the stable branch. You should now check out the stable 
branch and build the conda package.

```bash
git checkout stable
cd conda
conda build -c sdv-dev -c pytorch -c conda-forge .
```

## Upload to Anaconda
Finally, you can upload the resulting package to Anaconda.

```bash
anaconda login
anaconda upload -u sdv-dev <PATH_TO_PACKAGE>
```