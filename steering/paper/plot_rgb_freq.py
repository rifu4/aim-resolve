# %%
# ---------------------------------------------------------------------------
# GPU / JAX environment setup (must run before importing jax).
# ---------------------------------------------------------------------------
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# %%
# ---------------------------------------------------------------------------
# Imports.
# ---------------------------------------------------------------------------
import pickle

import jax.numpy as jnp
import numpy as np
import nifty.re as jft

import aim_resolve as aim
from aim_resolve.model.util import to_shape, is_val
from jax import vmap
from jax.scipy.ndimage import map_coordinates

# %%
# ===========================================================================
# Multi-color (spectral -> RGB) rendering.
#
# Ported (CPU / numpy only) from the `mobula` package:
#   - mobula/service/spectral_rgb.py
#   - mobula/service/acceleration/multispectral_common.py
#   - mobula/service/acceleration/multispectral_cpu.py
# The GPU (cupy) branches, the `xp` array-module abstraction and the FastAPI
# service wrapper have been dropped; the numerical logic is unchanged.
#
# Renders a spectral cube of shape (f, n, m) -- f a frequency axis, (n, m) the
# spatial dimensions -- into an (n, m, 3) sRGB image: the per-pixel spectral
# shape sets the hue (via the CIE XYZ color-matching functions), the total flux
# sets the brightness.
#
# This is the fused version of plot_multi_color.py + plot_multi_color_cs.py: it
# produces, in order, the full-sky image (with box markers), the two galaxy
# components as a column, and the brightest tiles as a grid.
# ===========================================================================

# --- CIE 1931 2-deg XYZ color-matching functions (380..780 nm, 5 nm steps) ---
_XYZ_CMF = np.array(
    [[
        0.000160, 0.000662, 0.002362, 0.007242, 0.019110, 0.043400,
        0.084736, 0.140638, 0.204492, 0.264737, 0.314679, 0.357719,
        0.383734, 0.386726, 0.370702, 0.342957, 0.302273, 0.254085,
        0.195618, 0.132349, 0.080507, 0.041072, 0.016172, 0.005132,
        0.003816, 0.015444, 0.037465, 0.071358, 0.117749, 0.172953,
        0.236491, 0.304213, 0.376772, 0.451584, 0.529826, 0.616053,
        0.705224, 0.793832, 0.878655, 0.951162, 1.014160, 1.074300,
        1.118520, 1.134300, 1.123990, 1.089100, 1.030480, 0.950740,
        0.856297, 0.754930, 0.647467, 0.535110, 0.431567, 0.343690,
        0.268329, 0.204300, 0.152568, 0.112210, 0.081261, 0.057930,
        0.040851, 0.028623, 0.019941, 0.013842, 0.009577, 0.006605,
        0.004553, 0.003145, 0.002175, 0.001506, 0.001045, 0.000727,
        0.000508, 0.000356, 0.000251, 0.000178, 0.000126, 0.000090,
        0.000065, 0.000046, 0.000033,
    ],
     [
         0.000017, 0.000072, 0.000253, 0.000769, 0.002004, 0.004509,
         0.008756, 0.014456, 0.021391, 0.029497, 0.038676, 0.049602,
         0.062077, 0.074704, 0.089456, 0.106256, 0.128201, 0.152761,
         0.185190, 0.219940, 0.253589, 0.297665, 0.339133, 0.395379,
         0.460777, 0.531360, 0.606741, 0.685660, 0.761757, 0.823330,
         0.875211, 0.923810, 0.961988, 0.982200, 0.991761, 0.999110,
         0.997340, 0.982380, 0.955552, 0.915175, 0.868934, 0.825623,
         0.777405, 0.720353, 0.658341, 0.593878, 0.527963, 0.461834,
         0.398057, 0.339554, 0.283493, 0.228254, 0.179828, 0.140211,
         0.107633, 0.081187, 0.060281, 0.044096, 0.031800, 0.022602,
         0.015905, 0.011130, 0.007749, 0.005375, 0.003718, 0.002565,
         0.001768, 0.001222, 0.000846, 0.000586, 0.000407, 0.000284,
         0.000199, 0.000140, 0.000098, 0.000070, 0.000050, 0.000036,
         0.000025, 0.000018, 0.000013,
     ],
     [
         0.000705, 0.002928, 0.010482, 0.032344, 0.086011, 0.197120,
         0.389366, 0.656760, 0.972542, 1.282500, 1.553480, 1.798500,
         1.967280, 2.027300, 1.994800, 1.900700, 1.745370, 1.554900,
         1.317560, 1.030200, 0.772125, 0.570060, 0.415254, 0.302356,
         0.218502, 0.159249, 0.112044, 0.082248, 0.060709, 0.043050,
         0.030451, 0.020584, 0.013676, 0.007918, 0.003988, 0.001091,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000,
     ]],
    dtype=np.float64,
)

_MATRIX_SRGB_D65 = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)

_CMF_WAVELENGTHS_NM = np.linspace(380.0, 780.0, _XYZ_CMF.shape[1], dtype=np.float64)


def _gamma_corr(inp):
    mask = np.zeros(inp.shape, dtype=np.float64)
    mask[inp <= 0.0031308] = 1.0
    r1 = 12.92 * inp
    a = 0.055
    r2 = (1 + a) * (np.maximum(inp, 0.0031308) ** (1 / 2.4)) - a
    return r1 * mask + r2 * (1.0 - mask)


def _xyz_from_wavelengths(wavelengths_nm):
    lam = np.clip(wavelengths_nm, _CMF_WAVELENGTHS_NM[0], _CMF_WAVELENGTHS_NM[-1])
    x = np.interp(lam, _CMF_WAVELENGTHS_NM, _XYZ_CMF[0])
    y = np.interp(lam, _CMF_WAVELENGTHS_NM, _XYZ_CMF[1])
    z = np.interp(lam, _CMF_WAVELENGTHS_NM, _XYZ_CMF[2])
    return np.stack([x, y, z], axis=0)


def _integration_weights(axis_values):
    if axis_values.ndim != 1:
        raise ValueError("axis_values must be one-dimensional")
    if axis_values.size == 1:
        return np.ones(1, dtype=np.float64)
    delta = np.abs(np.diff(axis_values))
    weights = np.empty(axis_values.size, dtype=np.float64)
    weights[0] = 0.5 * delta[0]
    weights[-1] = 0.5 * delta[-1]
    if axis_values.size > 2:
        weights[1:-1] = 0.5 * (delta[:-1] + delta[1:])
    normalizer = weights.sum()
    if normalizer > 0:
        weights /= normalizer
    return weights


def _to_logscale(arr, lo, hi):
    arr = np.asarray(arr, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    eps = np.finfo(np.float64).tiny
    lo = np.maximum(lo, eps)
    hi = np.maximum(hi, lo * (1.0 + 1e-12))
    clipped = arr.clip(lo, hi)
    return np.log(clipped / lo) / np.log(hi / lo)


def _xyz_to_srgb(xyz_data):
    rgb_linear = xyz_data @ _MATRIX_SRGB_D65.T
    return _gamma_corr(rgb_linear).clip(0.0, 1.0)


def build_visible_wavelength_axis(nu_coords, *, axis_scale, lambda_min=400.0, lambda_max=700.0):
    """Map frequency coordinates onto the visible band (low freq -> red, high -> blue)."""
    nu_arr = np.asarray(nu_coords, dtype=np.float64).reshape(-1)
    if nu_arr.size < 1:
        raise ValueError("nu_coords must be non-empty")

    axis_scale_applied = axis_scale
    if axis_scale == "log":
        if np.any(nu_arr <= 0) or np.any(~np.isfinite(nu_arr)):
            scaled = nu_arr
            axis_scale_applied = "linear"
        else:
            scaled = np.log10(nu_arr)
    else:
        scaled = nu_arr

    finite = np.isfinite(scaled)
    if not np.any(finite):
        t = np.linspace(0.0, 1.0, nu_arr.size, dtype=np.float64)
    else:
        lo = float(np.min(scaled[finite]))
        hi = float(np.max(scaled[finite]))
        if hi <= lo:
            t = np.linspace(0.0, 1.0, nu_arr.size, dtype=np.float64)
        else:
            t = (scaled - lo) / (hi - lo)
            t = np.where(np.isfinite(t), t, 0.0)

    # Low frequency maps to red (longer wavelengths), high frequency to blue.
    wavelength_nm = lambda_max - t * (lambda_max - lambda_min)
    return wavelength_nm.astype(np.float64), axis_scale_applied


def convert_mf_to_rgb_new(
    spectral_cube,
    *,
    wavelength_axis_nm,
    intensity_scale="log",
    clip_min=0.0,
    clip_max=1.0,
    dynamic_range=2.5e3,
    after_log_gammacorr=None,
    reuse_brightness_scale=False,
    channel_relative_clip=False,
    channel_clip_reference=None,
):
    """Integrate a spectral cube (freq axis LAST) against the CIE CMFs -> sRGB."""
    shp = spectral_cube.shape[:-1] + (3,)
    n_freqs = spectral_cube.shape[-1]
    spectral_cube = np.asarray(spectral_cube, dtype=np.float64).reshape((-1, n_freqs))
    wavelength_axis_nm = np.asarray(wavelength_axis_nm, dtype=np.float64).reshape(-1)
    if wavelength_axis_nm.size != n_freqs:
        raise ValueError("wavelength_axis_nm must have shape (n_freqs,)")

    if reuse_brightness_scale:
        maxval = float(reuse_brightness_scale)
    else:
        finite = np.isfinite(spectral_cube)
        if np.any(finite):
            maxval = float(np.max(spectral_cube[finite]))
        else:
            maxval = 1.0
    if not np.isfinite(maxval) or maxval <= 0:
        maxval = 1.0

    clip_min = float(np.clip(clip_min, 0.0, 1.0))
    clip_max = float(np.clip(clip_max, 0.0, 1.0))
    if clip_max <= clip_min:
        clip_max = min(1.0, clip_min + 1.0e-3)

    if channel_relative_clip:
        if channel_clip_reference is not None:
            ref = np.asarray(channel_clip_reference, dtype=np.float64).reshape((-1, n_freqs))
            if ref.shape != spectral_cube.shape:
                raise ValueError("channel_clip_reference must have the same shape as spectral_cube")
        else:
            ref = spectral_cube
        finite = np.isfinite(ref)
        mostly_nonnegative = False
        if np.any(finite):
            negative_fraction = float(np.mean(ref[finite] < 0.0))
            mostly_nonnegative = negative_fraction <= 0.25
        valid = np.any(finite, axis=0)
        mins = np.where(finite, ref, np.inf).min(axis=0)
        maxs = np.where(finite, ref, -np.inf).max(axis=0)
        mins = np.where(valid, mins, 0.0)
        maxs = np.where(valid, maxs, mins + 1.0)
        span = np.maximum(maxs - mins, 1.0e-12)
        lo = mins + clip_min * span
        hi = mins + clip_max * span
        hi = np.maximum(hi, lo + 1.0e-12)
        lo = lo[np.newaxis, :]
        hi = hi[np.newaxis, :]
        if mostly_nonnegative:
            lo = np.maximum(lo, 0.0)
            hi = np.maximum(hi, lo + 1.0e-12)

        finite_cur = np.isfinite(spectral_cube)
        cur_mins = np.where(finite_cur, spectral_cube, np.inf).min(axis=0, keepdims=True)
        cur_maxs = np.where(finite_cur, spectral_cube, -np.inf).max(axis=0, keepdims=True)
        valid_cur = np.any(finite_cur, axis=0, keepdims=True)
        if clip_min <= 0.0:
            if mostly_nonnegative:
                cur_floor = np.maximum(cur_mins, 0.0)
                lo = np.where(valid_cur, np.minimum(lo, cur_floor), lo)
                lo = np.maximum(lo, 0.0)
            else:
                lo = np.where(valid_cur, np.minimum(lo, cur_mins), lo)
        if clip_max >= 1.0:
            hi = np.where(valid_cur, np.maximum(hi, cur_maxs), hi)
        hi = np.maximum(hi, lo + 1.0e-12)

        clipped = np.clip(spectral_cube, lo, hi)
        rel = np.maximum(clipped - lo, 0.0)
        span = np.maximum(hi - lo, 1.0e-12)
        span_global = float(np.max(span))
        if not np.isfinite(span_global) or span_global <= 0.0:
            span_global = 1.0
        channel_gain = span / span_global

        if intensity_scale == "log":
            floor = np.maximum(span / max(dynamic_range, 1.0 + 1.0e-12), np.finfo(np.float64).tiny)
            denom = np.log1p(span / floor)
            spectral_norm = np.log1p(rel / floor) / np.maximum(denom, 1.0e-12)
            spectral_norm = spectral_norm * channel_gain
        elif intensity_scale == "sqrt":
            spectral_norm = np.sqrt(rel / span) * channel_gain
        else:
            spectral_norm = rel / span_global
    else:
        hi = maxval * clip_max
        if intensity_scale == "log":
            if clip_min > 0.0:
                lo = maxval * clip_min
            else:
                lo = hi / max(dynamic_range, 1.0 + 1e-12)
            spectral_norm = _to_logscale(spectral_cube, hi=hi, lo=lo)
        else:
            lo = maxval * clip_min
            lo = max(lo, 0.0)
            hi = max(hi, lo + 1.0e-12)
            spectral_norm = np.clip((spectral_cube - lo) / (hi - lo), 0.0, 1.0)
            if intensity_scale == "sqrt":
                spectral_norm = np.sqrt(spectral_norm)
    if after_log_gammacorr is not None:
        spectral_norm = np.float_power(spectral_norm, after_log_gammacorr)

    xyz_response = _xyz_from_wavelengths(wavelength_axis_nm)
    weights = _integration_weights(wavelength_axis_nm)
    weighted_response = xyz_response * weights[np.newaxis, :]
    xyz_data = np.tensordot(spectral_norm, weighted_response, axes=[-1, -1])
    rgb_data = _xyz_to_srgb(xyz_data)
    return rgb_data.reshape(shp), maxval


# --- pipeline helpers (ported from multispectral_common.py, numpy only) ------
def normalize_total_flux_brightness(total_flux, *, intensity_mode, clip_min, clip_max, dynamic_range=2.5e3, maxval=None):
    arr = np.asarray(total_flux, dtype=np.float64)
    finite = np.isfinite(arr)
    if not bool(np.any(finite)):
        return np.zeros_like(arr, dtype=np.float64)

    # `maxval` (the total flux mapped to full brightness) defaults to this
    # image's own max; pass a shared reference to match brightness across images.
    maxval = float(np.max(arr[finite])) if maxval is None else float(maxval)
    if not np.isfinite(maxval) or maxval <= 0:
        return np.zeros_like(arr, dtype=np.float64)

    hi = maxval * clip_max
    if intensity_mode == "log":
        if clip_min > 0.0:
            lo = maxval * clip_min
        else:
            lo = hi / max(dynamic_range, 1.0 + 1e-12)
        lo = max(lo, np.finfo(np.float64).tiny)
        hi = max(hi, lo * (1.0 + 1e-12))
        clipped = np.clip(arr, lo, hi)
        return np.log(clipped / lo) / np.log(hi / lo)

    lo = max(maxval * clip_min, 0.0)
    hi = max(hi, lo + 1.0e-12)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if intensity_mode == "sqrt":
        norm = np.sqrt(norm)
    return norm


def normalize_spectrum(arr, *, normalize_boost, ref_spectrum=None):
    arr64 = np.asarray(arr, dtype=np.float64)
    if ref_spectrum is None:
        mean_spectrum = np.mean(arr64, axis=(1, 2), dtype=np.float64)
    else:
        # Fixed per-channel reference (e.g. the full-sky mean) so the hue
        # normalization is shared across images instead of being per-image.
        mean_spectrum = np.asarray(ref_spectrum, dtype=np.float64).reshape(-1)
    finite = np.isfinite(mean_spectrum)
    if bool(np.any(finite)):
        median_abs = float(np.median(np.abs(mean_spectrum[finite])))
    else:
        median_abs = 1.0
    if not np.isfinite(median_abs) or median_abs <= 0.0:
        median_abs = 1.0
    floor = max(np.finfo(np.float64).tiny, median_abs * 1.0e-12)
    scale = np.where(finite & (np.abs(mean_spectrum) >= floor), mean_spectrum, 1.0)
    normalized = arr64 / scale[:, np.newaxis, np.newaxis]
    if normalize_boost != 1.0:
        positive = normalized > 0.0
        boosted = np.empty_like(normalized, dtype=np.float64)
        boosted[positive] = np.float_power(np.maximum(normalized[positive], floor), normalize_boost)
        boosted[~positive] = normalized[~positive] * normalize_boost
        return boosted.astype(np.float32)
    return normalized.astype(np.float32)


def apply_deslope(arr, nu_coords, *, deslope):
    if float(deslope) == 0.0:
        return arr, None

    nu_abs = np.abs(np.asarray(nu_coords, dtype=np.float64))
    valid = np.isfinite(nu_abs) & (nu_abs > 0)
    if not bool(np.any(valid)):
        return arr, None

    deslope_ref = float(np.median(nu_abs[valid]))
    weights = np.ones(arr.shape[0], dtype=np.float64)
    weights[valid] = np.power(nu_abs[valid] / deslope_ref, float(deslope))
    return arr * weights[:, np.newaxis, np.newaxis].astype(np.float32), deslope_ref


def prepare_chroma(arr):
    arr_rgb = np.moveaxis(arr, 0, -1).astype(np.float64)
    arr_chroma = np.maximum(arr_rgb, 0.0)
    denom = np.sum(arr_chroma, axis=-1, keepdims=True, dtype=np.float64)
    denom = np.maximum(denom, np.finfo(np.float64).tiny)
    return arr_chroma / denom


def apply_brightness_scale(rgb_cube, brightness_source, *, intensity_mode, clip_min, clip_max, dynamic_range=2.5e3, brightness_max=None):
    total_flux = np.sum(np.maximum(brightness_source, 0.0), axis=0, dtype=np.float64)
    brightness = normalize_total_flux_brightness(
        total_flux, intensity_mode=intensity_mode, clip_min=clip_min, clip_max=clip_max,
        dynamic_range=dynamic_range, maxval=brightness_max,
    )
    luma = 0.2126 * rgb_cube[:, :, 0] + 0.7152 * rgb_cube[:, :, 1] + 0.0722 * rgb_cube[:, :, 2]
    scale = brightness / np.maximum(luma, 1.0e-6)
    return np.clip(rgb_cube * scale[:, :, np.newaxis], 0.0, 1.0)


def spectral_cube_to_rgb(
    cube,
    nu_coords,
    *,
    nu_axis_scale="linear",
    deslope=0.0,
    normalize_spectrum_enabled=False,
    normalize_spectrum_boost=1.0,
    intensity_scale="linear",
    range_min=0.0,
    range_max=100.0,
    dynamic_range=2.5e3,
    lambda_min=400.0,
    lambda_max=700.0,
    brightness_max=None,
    spectrum_ref=None,
):
    """
    Render a spectral cube of shape (f, n, m) into an (n, m, 3) sRGB image.

    Mirrors mobula's `build_multispectral_response` + `CpuMultispectralBackend`:
    the per-pixel spectral shape sets the hue, the total flux sets brightness.

    Parameters
    ----------
    cube : np.ndarray
        Spectral cube, shape (f, n, m) with f the frequency axis.
    nu_coords : np.ndarray
        Frequency coordinates, shape (f,).
    nu_axis_scale : {"linear", "log"}
        Frequency -> visible-wavelength mapping.
    deslope : float
        Spectral tilt exponent applied as (nu / nu_ref) ** deslope.
    normalize_spectrum_enabled : bool
        Divide each channel by its spatial mean before colorizing.
    normalize_spectrum_boost : float
        Power boost for the normalized spectrum (mobula range [0.25, 8.0]).
    intensity_scale : {"linear", "sqrt", "log"}
        Brightness curve applied to the total flux.
    range_min, range_max : float
        Brightness clip percentiles (0..100), -> clip_min/clip_max = range/100.
    dynamic_range : float
        Log-scale floor (hi / dynamic_range) used when clip_min == 0.
    lambda_min, lambda_max : float
        Visible-band endpoints in nm.
    brightness_max : float, optional
        Total-flux value mapped to full brightness. Defaults to this cube's own
        max; pass a shared reference (e.g. the full sky) to match brightness
        across images.
    spectrum_ref : array-like, optional
        Per-channel divisor for `normalize_spectrum`, shape (f,). Defaults to
        this cube's spatial mean; pass a shared reference to match hue.
    """
    arr = np.asarray(cube, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"cube must be 3D (f, n, m), got shape {arr.shape}")
    nu_coords = np.asarray(nu_coords, dtype=np.float64).reshape(-1)
    if nu_coords.size != arr.shape[0]:
        raise ValueError("nu_coords must have shape (f,) matching cube.shape[0]")
    if arr.shape[0] < 3:
        raise ValueError("need at least 3 spectral channels for multispectral RGB")

    clip_min = float(np.clip(range_min / 100.0, 0.0, 1.0))
    clip_max = float(np.clip(range_max / 100.0, 0.0, 1.0))

    wavelength_axis_nm, _ = build_visible_wavelength_axis(
        nu_coords, axis_scale=nu_axis_scale, lambda_min=lambda_min, lambda_max=lambda_max
    )

    brightness_source = np.asarray(arr, dtype=np.float64).copy()

    if normalize_spectrum_enabled:
        arr = normalize_spectrum(arr, normalize_boost=normalize_spectrum_boost, ref_spectrum=spectrum_ref)

    arr, _ = apply_deslope(arr, nu_coords, deslope=deslope)
    arr_chroma = prepare_chroma(arr)

    rgb_cube, _ = convert_mf_to_rgb_new(
        arr_chroma,
        wavelength_axis_nm=wavelength_axis_nm,
        intensity_scale="linear",
        clip_min=0.0,
        clip_max=1.0,
        channel_relative_clip=False,
    )

    rgb_cube = apply_brightness_scale(
        rgb_cube, brightness_source, intensity_mode=intensity_scale, clip_min=clip_min, clip_max=clip_max,
        dynamic_range=dynamic_range, brightness_max=brightness_max,
    )
    return rgb_cube


def _gap_position(render_nu, label_nu, norm, *, min_ratio=1.5):
    """
    Normalized [0, 1] bar position of the largest gap between consecutive
    frequency labels, or None if there is no gap notably larger than the rest.

    `render_nu` are the channel positions on the bar, `label_nu` the real
    frequencies; the gap is placed midway (in bar coordinates) between the two
    channels straddling the biggest jump in `label_nu`. Returns None unless that
    jump exceeds `min_ratio` times the median jump (so evenly-spaced axes get no
    fade).
    """
    order = np.argsort(np.asarray(render_nu, dtype=np.float64))
    r = np.asarray(render_nu, dtype=np.float64)[order]
    l = np.asarray(label_nu, dtype=np.float64)[order]
    if r.size < 3:
        return None
    diffs = np.abs(np.diff(l))
    i = int(np.argmax(diffs))
    med = float(np.median(diffs))
    if not np.isfinite(med) or med <= 0 or diffs[i] < min_ratio * med:
        return None
    mid = 0.5 * (r[i] + r[i + 1])
    return float(norm(mid))


def frequency_colormap(nu_coords, *, nu_axis_scale="linear", lambda_min=400.0, lambda_max=700.0, n=256, spectral=False, brightness_normalize=True, gap_pos=None, gap_width=0.05, gap_depth=1.0, gap_core=0.03):
    """
    Build a (cmap, norm) pair mapping frequency -> the color a source at that
    frequency takes in the multi-color images, as a legend for the color axis.

    By default this is a true RGB ramp: red (low freq) -> green -> blue (high
    freq), matching the additive red/green/blue convention of the images. Set
    `spectral=True` for the physical spectral-locus (rainbow) colors from the
    CIE color-matching functions. Suitable for
    `fig.colorbar(ScalarMappable(norm=norm, cmap=cmap))`.

    If `gap_pos` (a normalized position in [0, 1], e.g. from `_gap_position`) is
    given, the bar is faded to transparent around it -- fully within a central
    core of half-width `gap_core` (the figure background shows through), with
    Gaussian shoulders of width `gap_width` and peak strength `gap_depth` --
    marking a larger frequency gap between two labels.
    """
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Normalize

    nu = np.asarray(nu_coords, dtype=np.float64).reshape(-1)
    nu_min, nu_max = float(np.min(nu)), float(np.max(nu))

    use_log = nu_axis_scale == "log" and nu_min > 0
    if use_log:
        norm = LogNorm(vmin=nu_min, vmax=nu_max)
    else:
        norm = Normalize(vmin=nu_min, vmax=nu_max)

    if spectral:
        nu_dense = np.geomspace(nu_min, nu_max, n) if use_log else np.linspace(nu_min, nu_max, n)
        wl, _ = build_visible_wavelength_axis(
            nu_dense, axis_scale=("log" if use_log else "linear"),
            lambda_min=lambda_min, lambda_max=lambda_max,
        )
        rgb = _xyz_to_srgb(_xyz_from_wavelengths(wl).T)  # (n, 3)
        if brightness_normalize:
            # Pure spectral colors vary strongly in luminance; scale each to full
            # brightness so the bar reads as a clean hue ramp.
            rgb = rgb / np.maximum(rgb.max(axis=1, keepdims=True), 1e-6)
    else:
        # Clean red -> green -> blue ramp (low freq red, high freq blue).
        base = LinearSegmentedColormap.from_list(
            "freq_rgb", [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], N=n
        )
        rgb = base(np.linspace(0.0, 1.0, n))[:, :3]

    if gap_pos is not None:
        # Fade the bar to transparent around the gap -- fully so in a small
        # central core (figure background shows through), with Gaussian shoulders
        # -- so the bar visibly "breaks".
        t = np.linspace(0.0, 1.0, n)
        d = np.abs(t - gap_pos)
        w = np.where(d <= gap_core, 1.0, np.exp(-0.5 * ((d - gap_core) / max(gap_width, 1e-6)) ** 2))
        w = np.clip(gap_depth * w, 0.0, 1.0)
        rgba = np.concatenate([np.clip(rgb, 0.0, 1.0), (1.0 - w)[:, None]], axis=1)
        return ListedColormap(rgba), norm

    return ListedColormap(np.clip(rgb, 0.0, 1.0)), norm


def box_markers(cfg, ps_map, grid, it):
    """
    Point-source and object-box scatter markers (copied from eso_components.py).

    Returns a dict of scatter specs: `ps_mrk` marks point sources (pixels where
    `ps_map > 0`) with small white circles, `oj_mrk` outlines the object boxes
    drawn by `aim.draw_boxes` with white ",". Each spec is passed straight to
    `ax.scatter(**spec)`.
    """
    from aim_resolve import draw_boxes

    px, py = np.argwhere(ps_map > 0).T
    ps_mrk = dict(x=px, y=py, s=20, facecolors="none", edgecolors="white", linewidths=0.25, marker="o")
    box_map = draw_boxes(cfg.sections, grid, it)
    ox, oy = np.argwhere(box_map > 0).T
    oj_mrk = dict(x=ox, y=oy, s=0.05, c="white", marker=",")

    return dict(ps_mrk=ps_mrk, oj_mrk=oj_mrk)


# ---------------------------------------------------------------------------
# SignalSpace and signal mapping (copied from eso_components.py), used to crop
# the full 2deg sky grid to a component sub-field of view and to zoom each tile
# into its central-half field of view.
# ---------------------------------------------------------------------------
class SignalSpace():
    '''Class to represent a signal space at a specific location in the sky. Use `build` function to create the space.'''

    def __init__(self, shape, distances, center=(0., 0.), n_copies=1):
        self.shape = shape
        self.distances = distances
        self.center = center
        self.n_copies = n_copies

    @classmethod
    def build(cls, *, shape, distances=None, fov=None, center=None, n_copies=1):
        shp = to_shape(shape, (2,), 'int64')
        dis = to_shape(distances, (2,), 'float64')
        fov = to_shape(fov, (2,), 'float64')
        cen = to_shape(center, (n_copies, 2), 'float64')

        shape = tuple(shp.tolist())

        if is_val(dis):
            distances = tuple(dis.tolist())
        elif is_val(fov):
            distances = tuple((fov / shp).tolist())
        else:
            distances = tuple((1 / shp).tolist())

        if not is_val(cen):
            cen = np.zeros_like(cen)
        center = tuple(map(tuple, cen.tolist()))

        if n_copies == 1:
            center = center[0]

        return cls(shape, distances, center, n_copies)

    @property
    def shp(self):
        return np.array(self.shape)

    @property
    def dis(self):
        return np.array(self.distances)

    @property
    def fov(self):
        return self.shp * self.dis

    @property
    def cen(self):
        return np.array(self.center)

    @property
    def coos(self):
        if self.n_copies == 1:
            return space_coos(self.shp, self.dis, self.cen)
        else:
            return vmap(space_coos, in_axes=(None, None, 0, 0))(self.shp, self.dis, self.cen)


def space_coos(shp, dis, cen):
    '''Generate the coordinates of the space.'''
    coos = jnp.indices(shp).astype(float)
    coos_T = coos.T.reshape(-1, 2)
    coos_T -= 0.5 * (shp - 1)
    coos_T *= dis
    coos_T += cen
    return coos_T.reshape(coos.T.shape).T


def map_signal(x, in_space, out_space, order=0, vmap_sum=True):
    '''Map one or more signals from a SignalSpace to another SignalSpace.'''
    if x.ndim == 2:
        return map_one_signal(x, in_space.dis, in_space.cen, out_space.coos, order)
    else:
        if in_space.n_copies > 1:
            vmap_one_signal = vmap(map_one_signal, in_axes=(0, None, 0, None, None))
            res = vmap_one_signal(x, in_space.dis, in_space.cen, out_space.coos, order)
        else:
            vmap_one_signal = vmap(map_one_signal, in_axes=(0, None, None, None, None))
            res = vmap_one_signal(x, in_space.dis, in_space.cen, out_space.coos, order)
        if vmap_sum:
            return jnp.sum(res, axis=0)
        else:
            return res


def map_one_signal(x, in_dis, in_cen, out_coos, order=0):
    x = jnp.asarray(x)
    out_coos = jnp.asarray(out_coos)
    out_coos_T = out_coos.T.reshape(-1, 2)
    out_coos_T -= in_cen
    out_coos_T /= in_dis
    out_coos_T += 0.5 * (jnp.array(x.shape) - 1)
    out_coos = out_coos_T.reshape(out_coos.T.shape).T
    return map_coordinates(x, out_coos, order)


def crop_component(nifty_array, rel_fov, center):
    '''
    Crop a full-grid (2deg) nifty image to a component sub-field of view.

    Acts on the last two dimensions, so a (nfreq, nx, ny) cube is cropped
    per-frequency and keeps its leading frequency axis.
    '''
    nifty_array = np.asarray(nifty_array)
    rel_fov = np.array(rel_fov)
    space = SignalSpace.build(shape=nifty_array.shape[-2:], fov=("2deg", "2deg"))
    sub = SignalSpace.build(
        shape=space.shp * rel_fov, fov=space.fov * rel_fov, center=center
    )
    return map_signal(nifty_array, space, sub, vmap_sum=False)


def _fade_cbar_outline(cb, *, orientation, gap_pos, gap_width=0.05, gap_depth=1.0, gap_core=0.03, n=200, lw=0.8):
    """
    Replace a colorbar's solid outline with one whose two long edges fade to
    transparent around `gap_pos` (normalized 0..1), matching the colormap gap so
    the frame visibly "breaks" there. The short end-caps stay solid. Also makes
    the colorbar's axes patch transparent so the fully-faded core shows the
    figure background.
    """
    from matplotlib.collections import LineCollection

    ax = cb.ax
    cb.outline.set_visible(False)
    ax.patch.set_alpha(0.0)  # let the figure background show through the faded core
    t = np.linspace(0.0, 1.0, n)
    d = np.abs(t - gap_pos)
    w = np.where(d <= gap_core, 1.0, np.exp(-0.5 * ((d - gap_core) / max(gap_width, 1e-6)) ** 2))
    alpha = 1.0 - np.clip(gap_depth * w, 0.0, 1.0)

    if orientation == "horizontal":
        edges = [np.column_stack([t, np.full_like(t, y)]) for y in (0.0, 1.0)]
        short = [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]]
    else:
        edges = [np.column_stack([np.full_like(t, x), t]) for x in (0.0, 1.0)]
        short = [[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]

    long_segs, long_alpha = [], []
    for pts in edges:
        for k in range(n - 1):
            long_segs.append([pts[k], pts[k + 1]])
            long_alpha.append(0.5 * (alpha[k] + alpha[k + 1]))
    long_colors = np.zeros((len(long_segs), 4))
    long_colors[:, 3] = long_alpha

    ax.add_collection(LineCollection(long_segs, colors=long_colors, linewidths=lw, transform=ax.transAxes, clip_on=False, zorder=5))
    ax.add_collection(LineCollection(short, colors=[(0.0, 0.0, 0.0, 1.0)] * len(short), linewidths=lw, transform=ax.transAxes, clip_on=False, zorder=5))


def plot_freq_brightness_legend(
    fig,
    rect,
    *,
    render_nu,
    label_nu,
    nu_axis_scale="linear",
    lambda_min=400.0,
    lambda_max=700.0,
    spectral=False,
    gap_pos=None,
    gap_width=0.10,
    gap_depth=1.0,
    gap_core=0.03,
    intensity_mode="log",
    clip_min=0.0,
    clip_max=1.0,
    dynamic_range=2.5e3,
    brightness_max=1.0,
    n_freq=256,
    n_bright=256,
    freq_scale=1e-6,
    freq_decimals=0,
    freq_label="frequency [MHz]",
    bright_label="mJy / arcsec$^2$",
):
    """
    Draw a 2D legend for the multi-color images at figure-fraction `rect`
    (`[x0, y0, w, h]`): x-axis = frequency (hue ramp, with the gap fade), y-axis
    = brightness = total flux (ticks/label on the right).

    Each cell is the pure-frequency hue scaled to that brightness exactly as
    `apply_brightness_scale` does (rgb_hue * b / luma), so the swatch is a
    faithful decoder of the image colors. The flux axis inverts the brightness
    normalization (`normalize_total_flux_brightness`) using the same
    `intensity_mode`, `clip_min/clip_max`, `dynamic_range` and `brightness_max`.
    """
    cmap, norm = frequency_colormap(
        render_nu, nu_axis_scale=nu_axis_scale, lambda_min=lambda_min, lambda_max=lambda_max,
        spectral=spectral, gap_pos=gap_pos, gap_width=gap_width, gap_depth=gap_depth, gap_core=gap_core,
    )
    cols = np.asarray(cmap(np.linspace(0.0, 1.0, n_freq)))  # (n_freq, 4)
    rgb_hue = cols[:, :3]
    col_alpha = cols[:, 3]
    luma = 0.2126 * rgb_hue[:, 0] + 0.7152 * rgb_hue[:, 1] + 0.0722 * rgb_hue[:, 2]

    # Rows = brightness in [0, 1], columns = frequency; scale hue by b / luma.
    b = np.linspace(0.0, 1.0, n_bright)
    scale = b[:, None] / np.maximum(luma[None, :], 1e-6)
    rgb = np.clip(rgb_hue[None, :, :] * scale[:, :, None], 0.0, 1.0)
    a = np.broadcast_to(col_alpha[None, :, None], rgb.shape[:2] + (1,))
    swatch = np.concatenate([rgb, a], axis=-1)  # (n_bright, n_freq, 4)

    ax = fig.add_axes(rect)
    ax.patch.set_alpha(0.0)  # let the figure background show through the gap
    ax.imshow(swatch, origin="lower", extent=[0.0, 1.0, 0.0, 1.0], aspect="auto", interpolation="nearest", zorder=2)

    # x-axis: frequency ticks at the channel positions.
    ax.set_xticks([float(norm(v)) for v in np.asarray(render_nu, dtype=np.float64).reshape(-1)])
    ax.set_xticklabels([f"{v * freq_scale:.{freq_decimals}f}" for v in np.asarray(label_nu, dtype=np.float64).reshape(-1)])
    if freq_label:
        ax.set_xlabel(freq_label)

    # y-axis: brightness -> flux (invert the brightness normalization), on the right.
    hi = brightness_max * clip_max
    if intensity_mode == "log":
        lo = brightness_max * clip_min if clip_min > 0.0 else hi / max(dynamic_range, 1.0 + 1e-12)
        lo = max(lo, np.finfo(np.float64).tiny)
        b_of = lambda fx: np.log(np.asarray(fx) / lo) / np.log(hi / lo)
        k0, k1 = int(np.ceil(np.log10(lo))), int(np.floor(np.log10(hi)))
        fx = [10.0 ** k for k in range(k0, k1 + 1)] or [lo, hi]
    else:
        lo = max(brightness_max * clip_min, 0.0)
        span = max(hi - lo, 1e-12)
        if intensity_mode == "sqrt":
            b_of = lambda fx: np.sqrt(np.clip((np.asarray(fx) - lo) / span, 0.0, 1.0))
        else:
            b_of = lambda fx: np.clip((np.asarray(fx) - lo) / span, 0.0, 1.0)
        fx = list(np.linspace(lo, hi, 4))
    fx = [f for f in fx if lo <= f <= hi]
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_yticks([float(np.clip(b_of(f), 0.0, 1.0)) for f in fx])
    ax.set_yticklabels([f"{f:.3g}" for f in fx])
    if bright_label:
        ax.set_ylabel(bright_label, rotation=270, labelpad=14)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    return ax


# ---------------------------------------------------------------------------
# Plotters: single image (+ markers + cbar), a column of images, and a grid.
# ---------------------------------------------------------------------------
def plot_multi_color(
    cube,
    nu_coords,
    *,
    odir=None,
    name=None,
    figsize=(10, 10),
    dpi=500,
    origin="lower",
    marker=None,
    cbar=False,
    cbar_label="frequency [MHz]",
    cbar_loc="right",
    cbar_width=0.0225,
    cbar_nu=None,
    cbar_freq_scale=1e-6,
    cbar_freq_decimals=0,
    cbar_spectral=False,
    cbar_gap=False,
    cbar_gap_width=0.05,
    cbar_gap_depth=1.0,
    cbar_gap_core=0.01,
    cbar_2d=False,
    cbar2d_size=(0.20, 0.24),
    cbar2d_bright_label="mJy / arcsec$^2$",
    **rgb_kwargs,
):
    """
    Render a spectral cube (f, n, m) as a multi-color RGB image and save to PNG.

    All keyword arguments beyond the plotting ones are forwarded to
    `spectral_cube_to_rgb` (nu_axis_scale, deslope, intensity_scale, range_min,
    range_max, ...). If `odir` and `name` are given the figure is saved,
    otherwise it is shown.

    `marker` overplots scatter markers (e.g. `box_markers(...)`): a single
    `{'x', 'y', ...}` spec, or a dict of such specs (each passed to
    `ax.scatter`).

    If `cbar=True` a frequency colorbar is drawn on the right: a true RGB ramp
    (low freq -> red, high -> blue; `cbar_spectral=True` for the rainbow
    spectral-locus). The hue ramp is positioned by `nu_coords` (so it matches
    the image); ticks sit at each channel, labelled with `cbar_nu` (real
    frequencies, e.g. `sky_mf.freq`) if given, else `nu_coords`, scaled by
    `cbar_freq_scale` (default Hz -> MHz) and formatted with `cbar_freq_decimals`
    digits.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    rgb = spectral_cube_to_rgb(cube, nu_coords, **rgb_kwargs)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Match the (a.T, origin="lower") orientation used in eso_compare.py.
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin=origin, aspect="auto")
    ax.set_box_aspect(rgb.shape[1] / rgb.shape[0])

    if marker:
        # Accept a single {x, y, ...} spec or a dict of such subdicts.
        specs = {"m0": marker} if all(k in marker for k in ("x", "y")) else marker
        for spec in specs.values():
            ax.scatter(**spec)

    ax.axis("off")

    # Frequency (1D) or frequency x brightness (2D) colorbar on the right.
    if cbar:
        render_nu = np.asarray(nu_coords, dtype=np.float64).reshape(-1)
        label_nu = render_nu if cbar_nu is None else np.asarray(cbar_nu, dtype=np.float64).reshape(-1)
        cmap_kw = dict(
            nu_axis_scale=rgb_kwargs.get("nu_axis_scale", "linear"),
            lambda_min=rgb_kwargs.get("lambda_min", 400.0),
            lambda_max=rgb_kwargs.get("lambda_max", 700.0),
            spectral=cbar_spectral,
        )
        _, norm = frequency_colormap(render_nu, **cmap_kw)
        gap_pos = _gap_position(render_nu, label_nu, norm) if cbar_gap else None
        fig.canvas.draw()
        box = ax.get_position()
        pad = 0.011  # fixed image-to-colorbar gap, matching plot_column.
        if cbar_2d:
            w, h = cbar2d_size
            x0 = (box.x0 - pad - w) if cbar_loc == "left" else (box.x1 + pad)
            y0 = box.y0 + 0.5 * (box.height - h)
            bmax = rgb_kwargs.get("brightness_max")
            if bmax is None:
                bmax = float(np.sum(np.maximum(np.asarray(cube, dtype=np.float64), 0.0), axis=0).max())
            plot_freq_brightness_legend(
                fig, [x0, y0, w, h], render_nu=render_nu, label_nu=label_nu, **cmap_kw,
                gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core,
                intensity_mode=rgb_kwargs.get("intensity_scale", "linear"),
                clip_min=float(np.clip(rgb_kwargs.get("range_min", 0.0) / 100.0, 0.0, 1.0)),
                clip_max=float(np.clip(rgb_kwargs.get("range_max", 100.0) / 100.0, 0.0, 1.0)),
                dynamic_range=rgb_kwargs.get("dynamic_range", 2.5e3),
                brightness_max=bmax,
                freq_scale=cbar_freq_scale, freq_decimals=cbar_freq_decimals,
                freq_label=cbar_label, bright_label=cbar2d_bright_label,
            )
        else:
            cmap, _ = frequency_colormap(render_nu, **cmap_kw)
            if gap_pos is not None:
                cmap, _ = frequency_colormap(
                    render_nu, **cmap_kw, gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core
                )
            x0 = (box.x0 - pad - cbar_width) if cbar_loc == "left" else (box.x1 + pad)
            cax = fig.add_axes([x0, box.y0, cbar_width, box.height])
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            cb.set_ticks(render_nu)
            cb.set_ticklabels([f"{v * cbar_freq_scale:.{cbar_freq_decimals}f}" for v in label_nu])
            if cbar_label:
                cb.set_label(cbar_label)
            if gap_pos is not None:
                _fade_cbar_outline(cb, orientation="vertical", gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core)

    if odir and name:
        os.makedirs(odir, exist_ok=True)
        if ".png" not in name:
            name += ".png"
        plt.savefig(os.path.join(odir, name), bbox_inches="tight")
        print("saved:", os.path.join(odir, name))
    else:
        plt.show()
    plt.close()
    return rgb


def plot_multi_color_column(
    cubes,
    nu_coords,
    *,
    odir=None,
    name=None,
    labels=None,
    label_color="white",
    frame=False,
    origin="lower",
    fig_width=5.0,
    hspace=0.025,
    dpi=300,
    cbar=False,
    cbar_label="frequency [MHz]",
    cbar_loc="right",
    cbar_width=0.0225,
    cbar_nu=None,
    cbar_freq_scale=1e-6,
    cbar_freq_decimals=0,
    cbar_spectral=False,
    cbar_gap=False,
    cbar_gap_width=0.05,
    cbar_gap_depth=1.0,
    cbar_gap_core=0.01,
    cbar_2d=False,
    cbar2d_size=(0.20, 0.24),
    cbar2d_bright_label="mJy / arcsec$^2$",
    **rgb_kwargs,
):
    """
    Render several spectral cubes (each (f, n, m)) as multi-color RGB images
    stacked in a single column, all sharing the same width in x.

    Mirrors the layout of `plot_column` in eso_components.py (same figure width
    per row, heights following each image's aspect ratio). Rendering kwargs are
    forwarded to `spectral_cube_to_rgb`.

    If `cbar=True` a shared frequency colorbar is drawn alongside the stack: a
    true RGB ramp (low freq -> red, high -> blue; `cbar_spectral=True` for the
    rainbow spectral-locus instead). The hue ramp is positioned by the rendering
    `nu_coords` (so it matches the images), while the tick labels use `cbar_nu`
    (the real frequencies, e.g. `sky_mf.freq`) if given, else `nu_coords`,
    scaled by `cbar_freq_scale` (default Hz -> MHz) and formatted with
    `cbar_freq_decimals` digits after the dot (0 -> "1350", 3 -> "1.350").
    Pass `cbar_nu` when `nu_coords` is just channel indices.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    rgbs = [spectral_cube_to_rgb(c, nu_coords, **rgb_kwargs) for c in cubes]
    n = len(rgbs)

    row_labels = ([None] * n) if labels is None else list(labels) + [None] * (n - len(labels))

    hspace = hspace if hspace > 0 else 0.025
    # Same width in x for every image; height_ratios give each its own aspect.
    aspects = [r.shape[1] / r.shape[0] for r in rgbs]
    fig_h = fig_width * sum(aspects) * (1 + hspace)
    figure, axes = plt.subplots(
        n,
        1,
        figsize=(fig_width, fig_h),
        dpi=dpi,
        gridspec_kw={"hspace": hspace, "height_ratios": aspects},
    )
    axes = np.atleast_1d(axes).ravel().tolist()

    for ax, r, lab in zip(axes, rgbs, row_labels):
        ax.imshow(np.transpose(r, (1, 0, 2)), origin=origin, aspect="auto")

        if lab:
            ax.annotate(
                lab, xy=(0.97, 1.0), xycoords="axes fraction",
                xytext=(0, -12), textcoords="offset points",
                ha="right", va="top", color=label_color,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(frame)
            if frame:
                spine.set_color("black")
                spine.set_linewidth(0.8)

    # Shared frequency colorbar spanning the combined height of the stack. The
    # hue ramp is positioned by the rendering `nu_coords` (so it matches the
    # images); ticks sit at each channel and are labelled with `cbar_nu`.
    if cbar:
        render_nu = np.asarray(nu_coords, dtype=np.float64).reshape(-1)
        label_nu = render_nu if cbar_nu is None else np.asarray(cbar_nu, dtype=np.float64).reshape(-1)
        cmap_kw = dict(
            nu_axis_scale=rgb_kwargs.get("nu_axis_scale", "linear"),
            lambda_min=rgb_kwargs.get("lambda_min", 400.0),
            lambda_max=rgb_kwargs.get("lambda_max", 700.0),
            spectral=cbar_spectral,
        )
        _, norm = frequency_colormap(render_nu, **cmap_kw)
        gap_pos = _gap_position(render_nu, label_nu, norm) if cbar_gap else None
        figure.canvas.draw()
        boxes = [ax.get_position() for ax in axes]
        top = max(b.y1 for b in boxes)
        bottom = min(b.y0 for b in boxes)
        pad = 0.011  # fixed image-to-colorbar gap, matching plot_column.
        if cbar_2d:
            w, h = cbar2d_size
            x0 = (min(b.x0 for b in boxes) - pad - w) if cbar_loc == "left" else (max(b.x1 for b in boxes) + pad)
            y0 = bottom + 0.5 * ((top - bottom) - h)
            bmax = rgb_kwargs.get("brightness_max")
            if bmax is None:
                bmax = float(max(np.sum(np.maximum(np.asarray(c, dtype=np.float64), 0.0), axis=0).max() for c in cubes))
            plot_freq_brightness_legend(
                figure, [x0, y0, w, h], render_nu=render_nu, label_nu=label_nu, **cmap_kw,
                gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core,
                intensity_mode=rgb_kwargs.get("intensity_scale", "linear"),
                clip_min=float(np.clip(rgb_kwargs.get("range_min", 0.0) / 100.0, 0.0, 1.0)),
                clip_max=float(np.clip(rgb_kwargs.get("range_max", 100.0) / 100.0, 0.0, 1.0)),
                dynamic_range=rgb_kwargs.get("dynamic_range", 2.5e3),
                brightness_max=bmax,
                freq_scale=cbar_freq_scale, freq_decimals=cbar_freq_decimals,
                freq_label=cbar_label, bright_label=cbar2d_bright_label,
            )
        else:
            cmap, _ = frequency_colormap(render_nu, **cmap_kw)
            if gap_pos is not None:
                cmap, _ = frequency_colormap(
                    render_nu, **cmap_kw, gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core
                )
            x0 = (min(b.x0 for b in boxes) - pad - cbar_width) if cbar_loc == "left" else (max(b.x1 for b in boxes) + pad)
            cax = figure.add_axes([x0, bottom, cbar_width, top - bottom])
            cb = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            cb.set_ticks(render_nu)
            # Fixed decimals after the dot (0 -> "1350" for MHz, 3 -> "1.350" for GHz).
            cb.set_ticklabels([f"{v * cbar_freq_scale:.{cbar_freq_decimals}f}" for v in label_nu])
            if cbar_label:
                cb.set_label(cbar_label)
            if gap_pos is not None:
                _fade_cbar_outline(cb, orientation="vertical", gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core)

    if odir and name:
        os.makedirs(odir, exist_ok=True)
        if ".png" not in name:
            name += ".png"
        plt.savefig(os.path.join(odir, name), bbox_inches="tight")
        print("saved:", os.path.join(odir, name))
    else:
        plt.show()
    plt.close()
    return rgbs


def plot_multi_color_grid(
    cubes,
    nu_coords,
    *,
    rows=6,
    cols=6,
    odir=None,
    name=None,
    labels=None,
    label_color="white",
    label_fontsize=12,
    frame=False,
    origin="lower",
    tile_size=2.0,
    space=0.04,
    scale=1.0,
    dpi=300,
    cbar=False,
    cbar_label="frequency [MHz]",
    cbar_nu=None,
    cbar_freq_scale=1e-6,
    cbar_freq_decimals=0,
    cbar_spectral=False,
    cbar_gap=False,
    cbar_gap_width=0.05,
    cbar_gap_depth=1.0,
    cbar_gap_core=0.01,
    cbar_2d=False,
    cbar2d_size=(0.20, 0.24),
    cbar2d_bright_label="mJy / arcsec$^2$",
    **rgb_kwargs,
):
    """
    Render several spectral cubes (each (f, n, m)) as multi-color RGB images in a
    `rows` x `cols` grid, mirroring the layout of `plot_tiles_grid` in
    eso_components.py. Rendering kwargs are forwarded to `spectral_cube_to_rgb`.

    If `cbar=True` a shared horizontal frequency colorbar is drawn at the bottom
    (true RGB ramp, `cbar_spectral=True` for the rainbow spectral-locus). Ticks
    sit at each channel, labelled with `cbar_nu` (real frequencies) if given else
    `nu_coords`, scaled by `cbar_freq_scale` and formatted with
    `cbar_freq_decimals` digits.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    rgbs = [spectral_cube_to_rgb(c, nu_coords, **rgb_kwargs) for c in cubes]

    # Layout (inches), matching plot_tiles_grid: side margins, a top margin and
    # a bottom strip for the shared colorbar; the figure height makes cells square.
    tile_size = tile_size * scale
    margin_x = 0.02 * tile_size * cols
    margin_top = 0.1 * scale
    if cbar and cbar_2d:
        cbar_strip = 2.2 * scale  # taller strip to fit the 2D legend swatch
    elif cbar:
        cbar_strip = 0.95 * scale
    else:
        cbar_strip = 0.1 * scale
    cbar_height = 0.28 * scale

    fig_w = tile_size * cols
    grid_w = fig_w - 2 * margin_x
    cell_w = grid_w / (cols + (cols - 1) * space)
    grid_h = cell_w * (rows + (rows - 1) * space)
    fig_h = grid_h + margin_top + cbar_strip

    cbar_strip_gap = space * cell_w

    left = margin_x / fig_w
    right = 1 - margin_x / fig_w
    bottom = cbar_strip / fig_h
    top = 1 - margin_top / fig_h

    figure, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=dpi)
    figure.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom, wspace=space, hspace=space
    )
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i < len(rgbs):
            ax.imshow(np.transpose(rgbs[i], (1, 0, 2)), origin=origin)
            if labels is not None and i < len(labels) and labels[i]:
                ax.text(
                    0.05, 0.93, labels[i],
                    transform=ax.transAxes, ha="left", va="top",
                    color=label_color, fontsize=label_fontsize,
                )
        else:
            ax.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(frame)
            if frame:
                spine.set_linewidth(0.5)

    # Single horizontal frequency colorbar spanning the grid width.
    if cbar:
        render_nu = np.asarray(nu_coords, dtype=np.float64).reshape(-1)
        label_nu = render_nu if cbar_nu is None else np.asarray(cbar_nu, dtype=np.float64).reshape(-1)
        cmap_kw = dict(
            nu_axis_scale=rgb_kwargs.get("nu_axis_scale", "linear"),
            lambda_min=rgb_kwargs.get("lambda_min", 400.0),
            lambda_max=rgb_kwargs.get("lambda_max", 700.0),
            spectral=cbar_spectral,
        )
        _, norm = frequency_colormap(render_nu, **cmap_kw)
        gap_pos = _gap_position(render_nu, label_nu, norm) if cbar_gap else None
        if cbar_2d:
            legend_h_in = 1.3 * scale
            legend_w_in = 3.0 * scale
            rect = [
                (fig_w - legend_w_in) / 2.0 / fig_w,
                (0.35 * scale) / fig_h,
                legend_w_in / fig_w,
                legend_h_in / fig_h,
            ]
            bmax = rgb_kwargs.get("brightness_max")
            if bmax is None:
                bmax = float(max(np.sum(np.maximum(np.asarray(c, dtype=np.float64), 0.0), axis=0).max() for c in cubes))
            plot_freq_brightness_legend(
                figure, rect, render_nu=render_nu, label_nu=label_nu, **cmap_kw,
                gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core,
                intensity_mode=rgb_kwargs.get("intensity_scale", "linear"),
                clip_min=float(np.clip(rgb_kwargs.get("range_min", 0.0) / 100.0, 0.0, 1.0)),
                clip_max=float(np.clip(rgb_kwargs.get("range_max", 100.0) / 100.0, 0.0, 1.0)),
                dynamic_range=rgb_kwargs.get("dynamic_range", 2.5e3),
                brightness_max=bmax,
                freq_scale=cbar_freq_scale, freq_decimals=cbar_freq_decimals,
                freq_label=cbar_label, bright_label=cbar2d_bright_label,
            )
        else:
            cmap, _ = frequency_colormap(render_nu, **cmap_kw)
            if gap_pos is not None:
                cmap, _ = frequency_colormap(
                    render_nu, **cmap_kw, gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core
                )
            cax = figure.add_axes(
                [left, (cbar_strip - cbar_strip_gap - cbar_height) / fig_h, right - left, cbar_height / fig_h]
            )
            cb = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
            cb.set_ticks(render_nu)
            cb.set_ticklabels([f"{v * cbar_freq_scale:.{cbar_freq_decimals}f}" for v in label_nu])
            if cbar_label:
                cb.set_label(cbar_label)
            if gap_pos is not None:
                _fade_cbar_outline(cb, orientation="horizontal", gap_pos=gap_pos, gap_width=cbar_gap_width, gap_depth=cbar_gap_depth, gap_core=cbar_gap_core)

    # Save without bbox_inches="tight" to preserve the computed layout (plot_figure).
    if odir and name:
        os.makedirs(odir, exist_ok=True)
        if ".png" not in name:
            name += ".png"
        plt.savefig(os.path.join(odir, name))
        print("saved:", os.path.join(odir, name))
    else:
        plt.show()
    plt.close()
    return rgbs


# %%
# ---------------------------------------------------------------------------
# Load the aim-resolve reconstruction: config, sky model and posterior samples
# for the chosen multi-frequency run (same as save_sky_hdf5.py / eso_compare.py).
# ---------------------------------------------------------------------------
dir = "/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b"

mf_rec = "4_rec_3z_4_6f_1_it_1_it_1"
mf_it = 6

ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
CONV_FACTOR = 1000 * AS2RAD**2

opt_yml = f"{dir}/opt/{mf_rec}/opt.yml"
print("load:", opt_yml)

optim_cfg = aim.OptimizeKLConfig.from_file(opt_yml, aim.get_builders)

sky_mf = optim_cfg.instantiate_sec(f"sky.{mf_it}")
print("sky components:", [c.prefix for c in sky_mf.models])

freq = [f"{round(f / 1e6)} MHz" for f in sky_mf.freq]
print("sky freqs:", freq)

with open(f"{dir}/opt/{mf_rec}/last.pkl", "rb") as f:
    samples_mf, *_ = pickle.load(f)
print("samples:", len(samples_mf))

# Point-source / object-box markers on the full sky grid (as in eso_components.py).
setup_cfg = aim.SetupKLConfig.from_file(opt_yml)
sky_ps = sky_mf.points[0]
ps_map = aim.map_signal(sky_ps.points.grid, sky_ps.grid)(np.ones(sky_ps.shape)).sum(axis=0)
markers_mf = box_markers(setup_cfg, ps_map, sky_mf.grid, mf_it)
print("markers:", [f"{k}: {len(v['x'])}" for k, v in markers_mf.items()])

# %%
# ---------------------------------------------------------------------------
# Shared rendering settings and output directory. The full-sky normalization
# references (brightness scale + hue) are reused by the component and tile plots
# so all three share the same color mapping (see spectral_cube_to_rgb).
# ---------------------------------------------------------------------------
odir = "/scratch/users/rfuchs/packages/aim-resolve/steering/paper/rgb_freq3"
nu_idx = [1, 2, 3, 4, 5, 6]  # even channel indices for rendering: distinct hues
# (real frequencies clump colors when the band has a gap; real freqs go on the
# colorbar labels instead, with the gap-fade marking the missing band).

color_dict = dict(
    nu_axis_scale="linear",
    intensity_scale="log",
    range_min=0.0,
    range_max=100.0,
    deslope=1.5,
    normalize_spectrum_enabled=True,
    normalize_spectrum_boost=7.5,
    dynamic_range=1e4,
    lambda_min=400.0,
    lambda_max=700.0,
)

sky = np.asarray(samples_mf.mean(sky_mf)) * CONV_FACTOR
print("sky model result shape:", sky.shape)

sky_brightness_max = float(np.sum(np.maximum(sky, 0.0), axis=0).max())
sky_spectrum_ref = np.mean(sky, axis=(1, 2))
print("sky brightness max:", sky_brightness_max)

# %%
# ---------------------------------------------------------------------------
# 1) Full sky, multi-color, with point-source / object-box markers.
# ---------------------------------------------------------------------------
plot_multi_color(
    sky,
    nu_idx,
    odir=odir,
    name="sky_rgb_box",
    marker=markers_mf,
    cbar=True,
    cbar_nu=sky_mf.freq,   # real frequencies (Hz) for the colorbar labels
    cbar_freq_scale=1e-6,  # Hz -> MHz
    cbar_gap=False,         # fade the bar where the frequency coverage has a gap
    cbar_spectral=True,    # CIE spectral-locus colors (match the renderer)
    **color_dict,
)

# %%
# ---------------------------------------------------------------------------
# 2) The two galaxy components (ESO137-006, ESO137-007) as a column. Each is the
# isolated per-object model mapped onto the sky grid and cropped to its sub-FoV
# (as in eso_components.py). Shared full-sky references keep colors consistent.
# ---------------------------------------------------------------------------
galaxy_labels = ["ESO137-006", "ESO137-007"]
components = [
    dict(obj=sky_mf.objects[0], rel_fov=(0.16, 0.08), center=(0, "-0.05deg")),        # ESO137-006
    dict(obj=sky_mf.objects[1], rel_fov=(0.25, 0.09), center=("0.18deg", "0.31deg")),  # ESO137-007
]

flux_comps = []
for c in components:
    obj = c["obj"]
    obj_sky = aim.map_signal(obj.grid, sky_mf.grid)(samples_mf.mean(obj)) * CONV_FACTOR
    flux_comps.append(np.asarray(crop_component(obj_sky, c["rel_fov"], c["center"])))
print("component shapes:", [c.shape for c in flux_comps])

plot_multi_color_column(
    flux_comps,
    nu_idx,
    odir=odir,
    name="cs_rgb",
    labels=galaxy_labels,
    fig_width=10.0,
    cbar=True,
    cbar_nu=sky_mf.freq,
    cbar_freq_scale=1e-6,
    cbar_gap=False,
    cbar_spectral=True,    # CIE spectral-locus colors (match the renderer)
    brightness_max=sky_brightness_max,  # match the full-sky brightness scale
    spectrum_ref=sky_spectrum_ref,      # match the full-sky hue normalization
    **color_dict,
)

# %%
# ---------------------------------------------------------------------------
# 3) The 36 brightest sky tiles as multi-color images on a 6x6 grid. Shared
# full-sky references keep colors consistent (as in eso_components.py tiles plot).
# ---------------------------------------------------------------------------
tiles = sky_mf.tiles[0]
tiles_rf = tiles.ref_freq_model

# Full frequency cube per tile (n_tiles, 6, ny, nx) and ref-freq map, mJy/arcsec^2.
tiles_cube = np.asarray(jft.mean(tuple(tiles(s, map=False) * CONV_FACTOR for s in samples_mf)))
tiles_rf_val = np.asarray(jft.mean(tuple(tiles_rf(s, map=False) * CONV_FACTOR for s in samples_mf)))
print("tiles cube shape:", tiles_cube.shape)

# Zoom each tile into its central-half FoV (as in eso_components.py), applied
# per (tile, frequency) via a flattened leading axis.
n_tiles, n_freq, ny, nx = tiles_cube.shape
spc_0 = SignalSpace.build(shape=(ny, nx), fov=(2, 2))
spc_1 = SignalSpace.build(shape=(ny, nx), fov=(1, 1))
tiles_cube = np.asarray(
    map_signal(tiles_cube.reshape(n_tiles * n_freq, ny, nx), spc_0, spc_1, order=1, vmap_sum=False)
).reshape(n_tiles, n_freq, ny, nx)
tiles_rf_val = np.asarray(map_signal(tiles_rf_val, spc_0, spc_1, order=1, vmap_sum=False))

# 36 brightest tiles by reference-frequency peak (same selection as eso_components).
tiles_peak = np.array([np.max(t) for t in tiles_rf_val])
tiles_order = np.argsort(tiles_peak)[::-1][:36]
bright_cubes = [tiles_cube[i] for i in tiles_order]

plot_multi_color_grid(
    bright_cubes,
    nu_idx,
    rows=6,
    cols=6,
    odir=odir,
    name="tiles_rgb",
    cbar=True,
    cbar_nu=sky_mf.freq,
    cbar_freq_scale=1e-6,
    cbar_gap=False,
    cbar_spectral=True,    # CIE spectral-locus colors (match the renderer)
    brightness_max=sky_brightness_max,  # match the full-sky brightness scale
    spectrum_ref=sky_spectrum_ref,      # match the full-sky hue normalization
    **color_dict,
)

# %%
