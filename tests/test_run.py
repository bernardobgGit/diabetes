"""
Tests for the Kedro project pipelines.
"""

from pathlib import Path

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project


class TestKedroPipelines:
    def test_pipelines_are_registered(self):
        bootstrap_project(Path.cwd())

        registered = set(pipelines)

        assert {
            "data_engineering",
            "modelling",
            "refit",
            "inference",
            "__default__",
        } <= registered
