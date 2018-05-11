#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `sdv` package."""

import unittest
from click.testing import CliRunner

from sdv import sdv
from sdv import cli


class TestSdv(unittest.TestCase):
    """Tests for `sdv` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        pass

    def tearDown(self):
        """Tear down test fixtures, if any."""
        pass

    def test_000_something(self):
        """Test something."""
        self.assertTrue(True)

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'sdv.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
