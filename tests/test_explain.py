"""Tests for explain()/dry_run() — annotated input preview, no subprocess."""

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest
import xarray as xr

from pyradtran.core import Simulation
from pyradtran.params import PROV_LITERAL, Var


class TestDryRun:
    def test_dry_run_returns_annotated_text(self, minimal_config):
        sim = Simulation(minimal_config)
        text = sim.dry_run(
            datetime(2022, 7, 1, 12), 78.0, 15.0,
            resolved_params={"albedo": (0.85, PROV_LITERAL)},
        )
        assert "albedo 0.85" in text
        assert "# params-literal" in text
        assert "# config" in text

    def test_dry_run_spawns_no_subprocess(self, minimal_config):
        sim = Simulation(minimal_config)
        with patch("subprocess.run") as mock_run:
            sim.dry_run(datetime(2022, 7, 1, 12), 78.0, 15.0)
        mock_run.assert_not_called()


class TestAccessorExplain:
    def test_explain_first_point_default(self, minimal_config, simple_input_dataset):
        text = simple_input_dataset.pyradtran.explain(config=minimal_config)
        assert "rte_solver disort" in text
        assert "# config" in text

    def test_explain_resolves_var_for_point(self, minimal_config, simple_input_dataset):
        ds = simple_input_dataset.copy()
        ds["alb"] = (["time"], [0.11, 0.22, 0.33])
        text = ds.pyradtran.explain(
            config=minimal_config, params={"albedo": Var("alb")}
        )
        assert "albedo 0.11" in text
        assert "# dataset-var" in text

    def test_explain_with_point_selector(self, minimal_config, simple_input_dataset):
        ds = simple_input_dataset.copy()
        ds["alb"] = (["time"], [0.11, 0.22, 0.33])
        text = ds.pyradtran.explain(
            config=minimal_config,
            params={"albedo": Var("alb")},
            point={"time": pd.Timestamp("2022-07-01 12:30")},
        )
        assert "albedo 0.22" in text
