# Release workflow

The process of releasing a new version involves several steps:

1. [Install SDV from source](#install-sdv-from-source)

2. [Linting and tests](#linting-and-tests)

3. [Make a release candidate](#make-a-release-candidate)

4. [Milestone](#milestone)

5. [Update HISTORY](#update-history)

6. [Check the release](#check-the-release)

7. [Update stable branch and bump version](#update-stable-branch-and-bump-version)

8. [Create the Release on GitHub](#create-the-release-on-github)

9. [Close milestone and create new milestone](#close-milestone-and-create-new-milestone)
    
10. [Release on Conda-Forge](#release-on-conda-forge)

## Install SDV from source

Clone the project and install the development requirements before start the release process. Alternatively, with your virtualenv activated.

```bash
git clone https://github.com/sdv-dev/SDV.git
cd SDV
git checkout main
make install-develop
make install-readme
```

## Linting and tests

Execute the tests and linting. The tests must end with no errors:

```bash
make test && make lint
```

And you will see something like this:

```
Coverage XML written to file ./integration_cov.xml
================ 425 passed, 568 warnings in 381.21s (0:06:21) =================
...
invoke lint
No broken requirements found.
All checks passed!
198 files already formatted
```

The execution has finished with no errors, 0 test skipped and 7820 warnings.

## Make a release candidate

1. On the SDV GitHub page, navigate to the [Actions][actions] tab.
2. Select the `Release` action.
3. Run it on the main branch. Make sure `Release candidate` is checked and `Test PyPI` is not.
4. Check on [PyPI][sdv-pypi] to assure the release candidate was successfully uploaded.
  - You should see X.Y.ZdevN PRE-RELEASE

[actions]: https://github.com/sdv-dev/sdv/actions
[sdv-pypi]: https://pypi.org/project/sdv/#history

## Milestone

It's important to check that the GitHub and milestone issues are up to date with the release.

You neet to check that:

- The milestone for the current release exists.
- All the issues closed since the latest release are associated to the milestone. If they are not, associate them
- All the issues associated to the milestone are closed. If there are open issues but the milestone needs to
  be released anyway, move them to the next milestone.
- All the issues in the milestone are assigned to at least one person.
- All the pull requests closed since the latest release are associated to an issue. If necessary, create issues
  and assign them to the milestone. Also assign the person who opened the issue to them.

## Update HISTORY
Run the [Release Prep](https://github.com/sdv-dev/SDV/actions/workflows/prepare_release.yml) workflow. This workflow will create a pull request with updates to HISTORY.md

Make sure HISTORY.md is updated with the issues of the milestone:

```
# History

## X.Y.Z (YYYY-MM-DD)

### New Features

* <ISSUE TITLE> - [Issue #<issue>](https://github.com/sdv-dev/SDV/issues/<issue>) by @resolver

### General Improvements

* <ISSUE TITLE> - [Issue #<issue>](https://github.com/sdv-dev/SDV/issues/<issue>) by @resolver

### Bug Fixed

* <ISSUE TITLE> - [Issue #<issue>](https://github.com/sdv-dev/SDV/issues/<issue>) by @resolver
```

The issue list per milestone can be found [here][milestones].

[milestones]: https://github.com/sdv-dev/SDV/milestones

Put the pull request up for review and get 2 approvals to merge into `main`.

## Check the release
Once HISTORY.md has been updated on `main`, check if the release can be made:

```bash
make check-release
```

## Update stable branch and bump version
The `stable` branch needs to be updated with the changes from `main` and the version needs to be bumped.
Depending on the type of release, run one of the following:

* `make release`: This will release the version that has already been bumped (patch, minor, or major). By default, this is typically a patch release. Use this when the changes are bugfixes or enhancements that do not modify the existing user API. Changes that modify the user API to add new features but that do not modify the usage of the previous features can also be released as a patch.
* `make release-minor`: This will bump and release the next minor version. Use this if the changes modify the existing user API in any way, even if it is backwards compatible. Minor backwards incompatible changes can also be released as minor versions while the library is still in beta state. After the major version v1.0.0 has been released, minor version can only be used to add backwards compatible API changes.
* `make release-major`: This will bump and release the next major version. Use this if the changes modify the user API in a backwards incompatible way after the major version v1.0.0 has been released.

Running one of these will **push commits directly** to `main`.
At the end, you should see the 3 commits on `main` (from oldest to newest):
- `make release-tag: Merge branch 'main' into stable`
- `Bump version: X.Y.Z.devN → X.Y.Z`
- `Bump version: X.Y.Z -> X.Y.A.dev0`

## Create the Release on GitHub

After the update to HISTORY.md is merged into `main` and the version is bumped, it is time to [create the release GitHub](https://github.com/sdv-dev/SDV/releases/new).
- Create a new tag with the version number with a v prefix (e.g. v0.3.1)
- The target should be the `stable` branch
- Release title is the same as the tag (e.g. v0.3.1)
- This is not a pre-release (`Set as a pre-release` should be unchecked)

Click `Publish release`, which will kickoff the release workflow and automatically upload the package to [public PyPI](https://pypi.org/project/sdv/).

## Close milestone and create new milestone

Finaly, **close the milestone** and, if it does not exist, **create the next milestone**.

## Release on conda-forge

After the release is published on [public PyPI](https://pypi.org/project/sdv/), Anacanoda will automatically open a [PR on conda-forge](https://github.com/conda-forge/sdv-feedstock/pulls). Make sure the dependencies match and then merge the PR for the anaconda release to be published.
