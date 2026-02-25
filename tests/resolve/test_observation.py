"""Tests for aim_resolve.resolve.observation — Observation class."""

import numpy as np
import pandas as pd
import pytest

from aim_resolve.resolve.observation import Observation, TABLE, INVTABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(nrow=10, npol=1, nfreq=2, pol_labels=None, name="test"):
    """Build a minimal Observation for testing.

    Returns an Observation with random visibilities, unit weights, and
    simple antenna positions.  The antpos DataFrame mirrors the MultiIndex
    column structure produced by ``Observation.load_npz``.
    """
    rng = np.random.default_rng(42)
    if pol_labels is None:
        pol_labels = np.array(["I"]) if npol == 1 else np.array(["XX", "YY"])[:npol]
    pol = np.asarray(pol_labels)
    freq = np.linspace(1e9, 2e9, nfreq)

    uvw = rng.standard_normal((nrow, 3))
    ant1 = np.zeros((nrow, 1), dtype=int)
    ant2 = np.arange(nrow, dtype=int).reshape(-1, 1)
    time = np.linspace(0, 1, nrow).reshape(-1, 1)
    dfs = []
    for a, k in zip([uvw, ant1, ant2, time],
                     [['u', 'v', 'w'], ['ant1'], ['ant2'], ['time']]):
        dfs.append(pd.DataFrame(a, columns=[k]))
    antpos = pd.concat(dfs, axis=1)

    idx = pd.MultiIndex.from_product([pol, freq], names=["pol", "freq"])
    vis_arr = rng.standard_normal((nrow, len(idx))) + 1j * rng.standard_normal((nrow, len(idx)))
    vis = pd.DataFrame(vis_arr.astype(np.complex128), columns=idx)
    wgt = pd.DataFrame(np.ones_like(vis_arr, dtype=np.float64), columns=idx)
    return Observation(pol, freq, antpos, vis, wgt, name=name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestObservationProperties:
    """Test basic property accessors."""

    def test_pol(self):
        obs = _make_obs()
        np.testing.assert_array_equal(obs.pol, ["I"])

    def test_freq_shape(self):
        obs = _make_obs(nfreq=4)
        assert obs.freq.shape == (4,)

    def test_nrow(self):
        obs = _make_obs(nrow=15)
        assert obs.nrow == 15

    def test_npol(self):
        obs = _make_obs(npol=2, pol_labels=np.array(["XX", "YY"]))
        assert obs.npol == 2

    def test_nfreq(self):
        obs = _make_obs(nfreq=3)
        assert obs.nfreq == 3

    def test_vis_shape(self):
        obs = _make_obs(nrow=8, npol=1, nfreq=3)
        assert obs.vis.shape == (1, 8, 3)

    def test_weight_shape(self):
        obs = _make_obs(nrow=8, npol=1, nfreq=3)
        assert obs.weight.shape == (1, 8, 3)

    def test_uvw_shape(self):
        obs = _make_obs(nrow=10)
        assert obs.uvw.shape == (10, 3)

    def test_uvwlen(self):
        obs = _make_obs(nrow=5)
        expected = np.linalg.norm(obs.uvw, axis=1)
        np.testing.assert_allclose(obs.uvwlen, expected)

    def test_ant1_ant2(self):
        obs = _make_obs(nrow=5)
        assert obs.ant1.shape == (5,)
        assert obs.ant2.shape == (5,)

    def test_time(self):
        obs = _make_obs(nrow=5)
        assert obs.time.shape == (5,)


class TestObservationFlags:
    """Test flag / mask / useful_fraction."""

    def test_all_unflagged(self):
        obs = _make_obs()
        assert obs.useful_fraction == pytest.approx(1.0)
        assert obs.mask.all().all()
        assert not obs.flags.any().any()

    def test_partially_flagged(self):
        obs = _make_obs(nrow=10, nfreq=2)
        # Zero-out some weights
        obs._wgt.iloc[:5] = 0.0
        assert obs.useful_fraction < 1.0
        assert obs.nvis_effective < obs.nvis


class TestObservationPrecision:
    """Test precision helpers."""

    def test_double_precision(self):
        obs = _make_obs()
        assert obs.precision == "double"

    def test_single_precision(self):
        obs = _make_obs()
        obs_sp = obs.to_single_precision()
        assert obs_sp.precision == "single"

    def test_to_double_roundtrip(self):
        obs = _make_obs()
        obs2 = obs.to_single_precision().to_double_precision()
        assert obs2.precision == "double"


class TestObservationSaveLoad:
    """Test pickle save / load roundtrip."""

    def test_save_load_pkl(self, tmp_path):
        obs = _make_obs(nrow=6, nfreq=2, name="roundtrip")
        fpath = str(tmp_path / "obs.pkl")
        obs.save(fpath)
        loaded = Observation.load(fpath)
        np.testing.assert_array_equal(loaded.pol, obs.pol)
        np.testing.assert_allclose(loaded.freq, obs.freq)
        np.testing.assert_allclose(loaded.vis, obs.vis)
        assert loaded.name == "roundtrip"

    def test_save_appends_pkl(self, tmp_path):
        obs = _make_obs()
        fpath = str(tmp_path / "obs")
        obs.save(fpath)
        assert (tmp_path / "obs.pkl").exists()

    def test_load_unknown_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown file format"):
            Observation.load(str(tmp_path / "obs.xyz"))


class TestObservationRepr:
    """Test __repr__ does not raise."""

    def test_repr(self):
        obs = _make_obs()
        r = repr(obs)
        assert "Observation:" in r
        assert "test" in r


class TestObservationNvis:
    """Test nvis computation."""

    def test_nvis(self):
        obs = _make_obs(nrow=10, npol=1, nfreq=3)
        assert obs.nvis == 10 * 1 * 3


class TestObservationBaselines:
    """Test baseline properties."""

    def test_baselines_set(self):
        obs = _make_obs(nrow=5)
        bl = obs.baselines
        assert isinstance(bl, set)
        assert len(bl) <= obs.nrow

    def test_nbaselines(self):
        obs = _make_obs(nrow=5)
        assert obs.nbaselines == len(obs.baselines)


class TestObservationFlagsToNan:
    """Test flags_to_nan conversion."""

    def test_no_flags_unchanged(self):
        obs = _make_obs()
        obs2 = obs.flags_to_nan()
        # same object returned when fraction is 1.0
        assert obs2 is obs

    def test_with_flags(self):
        obs = _make_obs(nrow=10, nfreq=2)
        obs._wgt.iloc[:2] = 0.0
        obs2 = obs.flags_to_nan()
        assert np.isnan(obs2.vis).any()


class TestObservationSubsample:
    """Test subsample_rows."""

    def test_fraction(self):
        obs = _make_obs(nrow=20)
        sub = obs.subsample_rows(0.5)
        assert sub.nrow == 10

    def test_integer(self):
        obs = _make_obs(nrow=20)
        sub = obs.subsample_rows(5)
        assert sub.nrow == 5

    def test_invalid_raises(self):
        obs = _make_obs(nrow=10)
        with pytest.raises(ValueError):
            obs.subsample_rows(-1)


class TestObservationRestrictPol:
    """Test restrict_to_pol."""

    def test_restrict(self):
        obs = _make_obs(npol=2, pol_labels=np.array(["XX", "YY"]))
        sub = obs.restrict_to_pol("XX")
        assert sub.npol == 1
        np.testing.assert_array_equal(sub.pol, ["XX"])

    def test_absent_pol_raises(self):
        obs = _make_obs()
        with pytest.raises(ValueError, match="not present"):
            obs.restrict_to_pol("XX")


class TestObservationAverageStokesI:
    """Test average_stokesi."""

    def test_already_stokesi(self):
        obs = _make_obs()
        res = obs.average_stokesi()
        assert res is obs

    def test_xx_yy(self):
        obs = _make_obs(npol=2, pol_labels=np.array(["XX", "YY"]))
        avg = obs.average_stokesi()
        np.testing.assert_array_equal(avg.pol, ["I"])
        assert avg.nfreq == obs.nfreq
        # rows preserved
        assert avg.nrow == obs.nrow


class TestObservationTable:
    """Test TABLE and INVTABLE mappings."""

    def test_roundtrip(self):
        for k, v in TABLE.items():
            assert INVTABLE[v] == k
