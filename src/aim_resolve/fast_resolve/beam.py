"""Primary-beam attenuation for the fast-resolve forward model."""

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np
from nifty.re import Model

# Validity ranges [MHz] of the MeerKAT JimBeam models (Mauch et al. 2020).
MEERKAT_BANDS = {
    "MKAT-AA-L-JIM-2020": (856.0, 1712.0),
    "MKAT-AA-UHF-JIM-2020": (544.0, 1088.0),
}


def default_beam_model(freq):
    """MeerKAT JimBeam model covering all frequencies ``freq`` [Hz]."""
    f_mhz = np.asarray(freq, dtype="float64") / 1e6
    for name, (lo, hi) in MEERKAT_BANDS.items():
        if f_mhz.min() >= lo and f_mhz.max() <= hi:
            return name
    raise ValueError(
        f"No MeerKAT beam model covers {f_mhz.min():.1f}-{f_mhz.max():.1f} MHz; "
        f"available: {MEERKAT_BANDS}. Set `model` explicitly."
    )


def beam_offsets(grid, pointing_offset=(0.0, 0.0)):
    """Offsets [deg] of every pixel of ``grid`` from the pointing centre.

    The phase centre is at the 0-based pixel N / 2 of the full (``factor``-
    refined) grid with pixel size ``distances / factor`` [rad], as in the
    interferometric response; ``grid.center`` (coarse pixels) shifts it.
    """
    pix = np.asarray(grid.dis, dtype="float64") / grid.factor
    cen = np.asarray(grid.cen, dtype="float64") * np.asarray(grid.dis, dtype="float64")
    off = [
        np.rad2deg((np.arange(n) - n / 2) * p + c) - o
        for n, p, c, o in zip(grid.shape, pix, cen, pointing_offset, strict=True)
    ]
    return np.meshgrid(*off, indexing="ij")


def _channel_weights(obs):
    """Total visibility weight per data channel."""
    w = np.asarray(getattr(obs.weight, "val", obs.weight), dtype="float64")
    return w.reshape(-1, w.shape[-1]).sum(axis=0)


def build_primary_beam(
    grid,
    freq,
    obs=None,
    model=None,
    channel_average=True,
    n_channels=8,
    pointing_offset=(0.0, 0.0),
):
    """Primary beam (Stokes I) on ``grid`` for every model frequency.

    The beam is evaluated once (NumPy) and returned as a constant array of the
    sky model's target shape: ``(nfreq, X, Y)``, or ``(X, Y)`` for a single
    frequency.

    Parameters
    ----------
    grid : SignalGrid
        Sky grid; the pointing centre is assumed at its phase centre.
    freq : array-like
        Model frequencies [Hz].
    obs : Observation, optional
        Data, needed for ``channel_average`` (channel frequencies and weights).
    model : str, optional
        MeerKAT JimBeam model (``MKAT-AA-L-JIM-2020`` or
        ``MKAT-AA-UHF-JIM-2020``); chosen from ``freq`` by default.
    channel_average : bool, optional
        Average the beam over the data channels of each model-frequency bin
        (bins as in ``resolve.InterferometryResponse``), weighted by the total
        channel weight. Otherwise the beam is taken at the model frequencies.
    n_channels : int, optional
        Number of evenly spaced data channels per bin used for the average.
    pointing_offset : tuple of float, optional
        Pointing centre relative to the phase centre [deg].

    Notes
    -----
    Assumes the same, time-invariant beam for all antennas (JimBeam is close to
    azimuthally symmetric) and Stokes I only.
    """
    from resolve import IRGSpace
    from resolve.library.meerkat_beam import JimBeam

    freq = np.atleast_1d(np.asarray(freq, dtype="float64"))
    model = default_beam_model(freq) if model is None else model
    beam = JimBeam(model)
    xx, yy = beam_offsets(grid, pointing_offset)

    if channel_average and obs is not None and freq.size > 1:
        ch_freq = np.asarray(obs.freq, dtype="float64")
        ch_wgt = _channel_weights(obs)
        bounds = IRGSpace(freq).binbounds()
        pb = np.empty((freq.size,) + grid.shape, dtype="float64")
        for i in range(freq.size):
            upper = ch_freq <= bounds[i + 1] if i == freq.size - 1 else ch_freq < bounds[i + 1]
            idx = np.nonzero((ch_freq >= bounds[i]) & upper)[0]
            if idx.size == 0 or ch_wgt[idx].sum() <= 0:
                pb[i] = beam.I(xx, yy, freq[i] / 1e6)
                continue
            idx = idx[np.unique(np.linspace(0, idx.size - 1, min(n_channels, idx.size)).round().astype(int))]
            w = ch_wgt[idx] / ch_wgt[idx].sum()
            pb[i] = sum(wi * beam.I(xx, yy, f / 1e6) for wi, f in zip(w, ch_freq[idx], strict=True))
    else:
        pb = np.stack([beam.I(xx, yy, f / 1e6) for f in freq])

    pb = pb.astype("float32")
    return pb[0] if freq.size == 1 else pb


class BeamedSky(Model):
    """Apparent sky: the intrinsic sky model multiplied by the primary beam.

    Parameters
    ----------
    sky : Model
        Intrinsic sky model with ``grid`` and ``freq`` attributes.
    beam : array-like
        Primary beam of the sky model's target shape.
    """

    beam: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, sky, beam):
        if np.shape(beam) != sky.target.shape:
            raise ValueError(f"beam shape {np.shape(beam)} != sky shape {sky.target.shape}")
        self.sky = sky
        self.grid = sky.grid
        self.freq = sky.freq
        self.beam = jnp.asarray(beam)
        super().__init__(domain=sky.domain, init=sky.init)

    def __call__(self, x, *args, **kwargs):
        return self.beam * self.sky(x, *args, **kwargs)
