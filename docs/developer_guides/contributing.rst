.. _contributing:

Contributing to SDV
===================

The Synthetic Data Vault project is a collection of several open source libraries with purposes
that range from simple reversible transformations to the most complex state-of-the-art Generative
Deep Learning models.

Building and maintaining all these libraries is a community driven effort which relies on the work
of developers from all around the world that contribute their knowledge, code and feedback to make
the project grow and improve over time.

Are you ready to join this community? If so, please feel welcome and keep reading!

Types of contributions
----------------------

There are several ways to contribute to a project like **SDV**, and they do not always involve
coding.

If you want to contribute but do not know where to start, consider one of the following options:

Reporting Issues
~~~~~~~~~~~~~~~~

If there is something that you would like to see changed in the project, or that you just want
to ask, please create an issue at the `GitHub issues page`_.

If you do so, please:

* Explain in detail what you are requesting.
* Keep the scope as narrow as possible, to make it easier to implement or respond.
* Remember that this is a volunteer-driven project and that the maintainers will attend every
  request as soon as possible, but that in some cases this might take some time.

Write Documentation
~~~~~~~~~~~~~~~~~~~

SDV could always use more documentation, whether as part of the official SDV
docs, in docstrings, or even on the web in blog posts, articles, and such, so feel free to
contribute any changes that you deem necessary, from fixing a simple typo, to writing whole
new pages of documentation.

Contribute code
~~~~~~~~~~~~~~~

Obviously, the main element in the SDV library is the code.

If you are willing to contribute to it, please head for the next sections for detailed guidelines
about how to do so.


Get Started!
------------

Ready to contribute? Here's how to set up `SDV` for local development.

1. Fork the `SDV` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/SDV.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed,
   this is how you set up your fork for local development::

    $ mkvirtualenv SDV
    $ cd SDV/
    $ make install-develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Try to use the naming scheme of prefixing your branch with ``gh-X`` where X is
   the associated issue, such as ``gh-3-fix-foo-bug``. And if you are not
   developing on your own fork, further prefix the branch with your GitHub
   username, like ``githubusername/gh-3-fix-foo-bug``.

   Now you can make your changes locally.

5. While hacking your changes, make sure to cover all your developments with the required
   unit tests, and that none of the old tests fail as a consequence of your changes.
   For this, make sure to run the tests suite and check the code coverage::

    $ make lint       # Check code styling
    $ make test       # Run the tests
    $ make coverage   # Get the coverage report

6. When you're done making changes, check that your changes pass all the styling checks and
   tests, including other Python supported versions, using::

    $ make test-all

7. Make also sure to include the necessary documentation in the code as docstrings following
   the `Google docstrings style`_.
   If you want to view how your documentation will look like when it is published, you can
   generate and view the docs with this command::

    $ make view-docs

8. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

9. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. It resolves an open GitHub Issue and contains its reference in the title or
   the comment. If there is no associated issue, feel free to create one.
2. Whenever possible, it resolves only **one** issue. If your PR resolves more than
   one issue, try to split it in more than one pull request.
3. The pull request should include unit tests that cover all the changed code
4. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the documentation in an appropriate place.
5. The pull request should work for all the supported Python versions. Check the `Travis Build
   Status page`_ and make sure that all the checks pass.

Unit Testing Guidelines
-----------------------

All the Unit Tests should comply with the following requirements:

1. Unit Tests should be based only in unittest and pytest modules.

2. The tests that cover a module called ``sdv/path/to/a_module.py``
   should be implemented in a separated module called
   ``tests/sdv/path/to/test_a_module.py``.
   Note that the module name has the ``test_`` prefix and is located in a path similar
   to the one of the tested module, just inside the ``tests`` folder.

3. Each method of the tested module should have at least one associated test method, and
   each test method should cover only **one** use case or scenario.

4. Test case methods should start with the ``test_`` prefix and have descriptive names
   that indicate which scenario they cover.
   Names such as ``test_some_methed_input_none``, ``test_some_method_value_error`` or
   ``test_some_method_timeout`` are right, but names like ``test_some_method_1``,
   ``some_method`` or ``test_error`` are not.

5. Each test should validate only what the code of the method being tested does, and not
   cover the behavior of any third party package or tool being used, which is assumed to
   work properly as far as it is being passed the right values.

6. Any third party tool that may have any kind of random behavior, such as some Machine
   Learning models, databases or Web APIs, will be mocked using the ``mock`` library, and
   the only thing that will be tested is that our code passes the right values to them.

7. Unit tests should not use anything from outside the test and the code being tested. This
   includes not reading or writing to any file system or database, which will be properly
   mocked.

Tips
----

To run a subset of tests::

    $ python -m pytest tests.test_sdv
    $ python -m pytest -k 'foo'

Release Workflow
----------------

The process of releasing a new version involves several steps combining both ``git`` and
``bumpversion`` which, briefly:

1. Merge what is in ``master`` branch into ``stable`` branch.
2. Update the version in ``setup.cfg``, ``sdv/__init__.py`` and
   ``HISTORY.md`` files.
3. Create a new git tag pointing at the corresponding commit in ``stable`` branch.
4. Merge the new commit from ``stable`` into ``master``.
5. Update the version in ``setup.cfg`` and ``sdv/__init__.py``
   to open the next development iteration.

.. note:: Before starting the process, make sure that ``HISTORY.md`` has been updated with a new
          entry that explains the changes that will be included in the new version.
          Normally this is just a list of the Pull Requests that have been merged to master
          since the last release.

Once this is done, run of the following commands:

1. If you are releasing a patch version::

    make release

2. If you are releasing a minor version::

    make release-minor

3. If you are releasing a major version::

    make release-major

Release Candidates
~~~~~~~~~~~~~~~~~~

Sometimes it is necessary or convenient to upload a release candidate to PyPi as a pre-release,
in order to make some of the new features available for testing on other projects before they
are included in an actual full-blown release.

In order to perform such an action, you can execute::

    make release-candidate

This will perform the following actions:

1. Build and upload the current version to PyPi as a pre-release, with the format ``X.Y.Z.devN``

2. Bump the current version to the next release candidate, ``X.Y.Z.dev(N+1)``

After this is done, the new pre-release can be installed by including the ``dev`` section in the
dependency specification, either in ``setup.py``::

    install_requires = [
        ...
        'sdv>=X.Y.Z.dev',
        ...
    ]

or in command line::

    pip install 'sdv>=X.Y.Z.dev'


.. _GitHub issues page: https://github.com/sdv-dev/SDV/issues
.. _Travis Build Status page: https://travis-ci.org/sdv-dev/SDV/pull_requests
.. _Google docstrings style: https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments
