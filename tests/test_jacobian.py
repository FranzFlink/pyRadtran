"""Jacobian mode: finite-difference kernels via paired runs."""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from pyradtran.exceptions import ValidationError
from pyradtran.params import Var


@pytest.fixture
def ds_in():
    return xr.Dataset(
        data_vars={
            "latitude": (["time"], [78.0, 78.1]),
            "longitude": (["time"], [15.0, 15.1]),
        },
        coords={"time": [0, 1]},
    )


class TestJacobian:
    def test_linear_model_gives_exact_kernel(self, minimal_config, ds_in):
        """Mock forward model: eup = 100 * albedo -> dU/dalbedo = 100."""

        def fake_run(self, **kwargs):
            alb = kwargs["params"]["albedo"]
            return xr.Dataset(
                {"eup": (("time",), [100.0 * alb, 100.0 * alb])},
                coords={"time": [0, 1]},
            )

        from pyradtran.interface import PyRadtranAccessor

        with patch.object(PyRadtranAccessor, "run", fake_run):
            jac = ds_in.pyradtran.jacobian(
                "albedo", 0.01, params={"albedo": 0.5}, config=minimal_config
            )
        assert jac["eup"].values == pytest.approx([100.0, 100.0])
        assert jac.attrs["jacobian_param"] == "albedo"
        assert jac.attrs["jacobian_delta"] == 0.01

    def test_base_value_from_config_field(self, minimal_config, ds_in):
        """albedo base value falls back to config albedo_value (0.1)."""
        seen = []

        def fake_run(self, **kwargs):
            seen.append(kwargs["params"]["albedo"])
            return xr.Dataset({"eup": (("time",), [1.0, 1.0])}, coords={"time": [0, 1]})

        from pyradtran.interface import PyRadtranAccessor

        with patch.object(PyRadtranAccessor, "run", fake_run):
            ds_in.pyradtran.jacobian("albedo", 0.01, config=minimal_config)
        assert seen == [pytest.approx(0.1), pytest.approx(0.11)]

    def test_var_param_rejected(self, minimal_config, ds_in):
        with pytest.raises(ValidationError):
            ds_in.pyradtran.jacobian(
                "albedo",
                0.01,
                params={"albedo": Var("alb")},
                config=minimal_config,
            )

    def test_unresolvable_base_rejected(self, minimal_config, ds_in):
        minimal_config.simulation_defaults.albedo_value = None
        with pytest.raises(ValidationError):
            ds_in.pyradtran.jacobian("albedo", 0.01, config=minimal_config)

    def test_status_not_differentiated(self, minimal_config, ds_in):
        """The status flag must survive as a flag (worst of both runs),
        not as a finite difference."""

        calls = {"n": 0}

        def fake_run(self, **kwargs):
            calls["n"] += 1
            status = [0, 1] if calls["n"] == 1 else [0, 0]
            alb = kwargs["params"]["albedo"]
            ds = xr.Dataset(
                {
                    "eup": (("time",), [100.0 * alb, 100.0 * alb]),
                    "status": (("time",), status),
                },
                coords={"time": [0, 1]},
            )
            return ds

        from pyradtran.interface import PyRadtranAccessor

        with patch.object(PyRadtranAccessor, "run", fake_run):
            jac = ds_in.pyradtran.jacobian(
                "albedo", 0.01, params={"albedo": 0.5}, config=minimal_config
            )
        # worst status per point across base+perturbed runs
        assert list(jac["status"].values) == [0, 1]
        assert jac["eup"].values == pytest.approx([100.0, 100.0])
