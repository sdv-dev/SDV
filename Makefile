.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


# CLEAN TARGETS

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-docs
clean-docs: ## remove previously built docs
	rm -rf docs/api/ docs/api_reference/api/ docs/tutorials docs/build docs/_build

.PHONY: clean-coverage
clean-coverage: ## remove coverage artifacts
	rm -f .coverage
	rm -f .coverage.*
	rm -fr htmlcov/

.PHONY: clean-test
clean-test: ## remove test artifacts
	rm -fr .tox/
	rm -fr .pytest_cache

.PHONY: clean
clean: clean-build clean-pyc clean-test clean-coverage clean-docs ## remove all build, test, coverage, docs and Python artifacts


# INSTALL TARGETS

.PHONY: install
install: clean-build clean-pyc ## install the package to the active Python's site-packages
	pip install .

.PHONY: install-test
install-test: clean-build clean-pyc ## install the package and test dependencies
	pip install .[test]

.PHONY: install-develop
install-develop: clean-build clean-pyc ## install the package in editable mode and dependencies for development
	pip install -e .[dev]


# LINT TARGETS

.PHONY: lint-sdv
lint-sdv: ## check style with flake8 and isort
	ruff check sdv/
	ruff format --check --diff sdv/

.PHONY: lint-tests
lint-tests: ## check style with flake8 and isort
	ruff check tests/
	ruff format --check --diff tests/

.PHONY: check-dependencies
check-dependencies: ## test if there are any broken dependencies
	pip check

.PHONY: lint
lint:
	invoke lint

.PHONY: fix-lint
fix-lint:
	invoke fix-lint


# TEST TARGETS

.PHONY: test-unit
test-unit: ## run tests quickly with the default Python
	invoke unit

.PHONY: test-integration
test-integration: ## run tests quickly with the default Python
	invoke integration

.PHONY: test-readme
test-readme: ## run the readme snippets
	invoke readme

.PHONY: test
test: test-unit test-integration test-readme ## test everything that needs test dependencies

.PHONY: test-all
test-all: ## run tests on every Python version with tox
	tox -r

.PHONY: coverage
coverage: ## check code coverage quickly with the default Python
	coverage run --source sdv -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html


# DOCS TARGETS

.PHONY: docs
docs: clean-docs ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs html

.PHONY: view-docs
view-docs: ## view the docs in a browser
	$(BROWSER) docs/_build/html/index.html

.PHONY: serve-docs
serve-docs: ## compile the docs watching for changes
	watchmedo shell-command -W -R -D -p '*.rst;*.md' -c '$(MAKE) -C docs html' docs


# RELEASE TARGETS

.PHONY: dist
dist: clean ## builds source and wheel package
	python -m build --wheel --sdist
	ls -l dist

.PHONY: publish-confirm
publish-confirm:
	@echo "WARNING: This will irreversibly upload a new version to PyPI!"
	@echo -n "Please type 'confirm' to proceed: " \
		&& read answer \
		&& [ "$${answer}" = "confirm" ]

.PHONY: publish-test
publish-test: dist publish-confirm ## package and upload a release on TestPyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: publish
publish: dist publish-confirm ## package and upload a release
	twine upload dist/*

.PHONY: bumpversion-release
bumpversion-release: ## Merge main to stable and bumpversion release
	git checkout stable || git checkout -b stable
	git merge --no-ff main -m"make release-tag: Merge branch 'main' into stable"
	bump-my-version bump release
	git push --tags origin stable

.PHONY: bumpversion-release-test
bumpversion-release-test: ## Merge main to stable and bumpversion release
	git checkout stable || git checkout -b stable
	git merge --no-ff main -m"make release-tag: Merge branch 'main' into stable"
	bump-my-version bump release --no-tag
	@echo git push --tags origin stable

.PHONY: bumpversion-patch
bumpversion-patch: ## Merge stable to main and bumpversion patch
	git checkout main
	git merge stable
	bump-my-version bump --no-tag patch
	git push

.PHONY: bumpversion-candidate
bumpversion-candidate: ## Bump the version to the next candidate
	bump-my-version bump candidate --no-tag

.PHONY: bumpversion-minor
bumpversion-minor: ## Bump the version the next minor skipping the release
	bump-my-version bump --no-tag minor

.PHONY: bumpversion-major
bumpversion-major: ## Bump the version the next major skipping the release
	bump-my-version bump --no-tag major

.PHONY: bumpversion-revert
bumpversion-revert: ## Undo a previous bumpversion-release
	git checkout main
	git branch -D stable

CLEAN_DIR := $(shell git status --short | grep -v ??)
CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null)
CHANGELOG_LINES := $(shell git diff HEAD..origin/stable HISTORY.md 2>&1 | wc -l)

.PHONY: check-clean
check-clean: ## Check if the directory has uncommitted changes
ifneq ($(CLEAN_DIR),)
	$(error There are uncommitted changes)
endif

.PHONY: check-main
check-main: ## Check if we are in main branch
ifneq ($(CURRENT_BRANCH),main)
	$(error Please make the release from main branch\n)
endif

.PHONY: check-history
check-history: ## Check if HISTORY.md has been modified
ifeq ($(CHANGELOG_LINES),0)
	$(error Please insert the release notes in HISTORY.md before releasing)
endif

.PHONY: git-push
git-push: ## Simply push the repository to github
	git push

.PHONY: check-release
check-release: check-clean check-main check-history ## Check if the release can be made
	@echo "A new release can be made"

.PHONY: release
release: check-release bumpversion-release publish bumpversion-patch

.PHONY: release-test
release-test: check-release bumpversion-release-test publish-test bumpversion-revert

.PHONY: release-candidate
release-candidate: check-main publish bumpversion-candidate git-push

.PHONY: release-candidate-test
release-candidate-test: check-clean check-main publish-test

.PHONY: release-minor
release-minor: check-release bumpversion-minor release

.PHONY: release-major
release-major: check-release bumpversion-major release

# Dependency targets

.PHONY: check-deps
check-deps:
	$(eval allow_list='cloudpickle=|graphviz=|numpy=|pandas=|tqdm=|copulas=|ctgan=|deepecho=|rdt=|sdmetrics=|platformdirs=|pyyaml=')
	pip freeze | grep -v "SDV.git" | grep -E $(allow_list) | sort > $(OUTPUT_FILEPATH)

.PHONY: upgradepip
upgradepip:
	python -m pip install --upgrade pip

.PHONY: upgradebuild
upgradebuild:
	python -m pip install --upgrade build

.PHONY: upgradesetuptools
upgradesetuptools:
	python -m pip install --upgrade setuptools

.PHONY: package
package: upgradepip upgradebuild upgradesetuptools
	python -m build ; \
	$(eval VERSION=$(shell python -c 'import setuptools; setuptools.setup()' --version))
	tar -zxvf "dist/sdv-${VERSION}.tar.gz"
	mv "sdv-${VERSION}" unpacked_sdist
