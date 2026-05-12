"""FFT-based convolution operators for fast-resolve likelihood evaluation."""

import dataclasses
import os
import pickle
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax.lax import dynamic_slice
from jax.numpy.fft import fftn, ifftn
from nifty.re import Model, Vector, smap

# from ..model.map import downsample, upsample
from ..model.noise import NoiseModel
from ..optimize.samples import domain_tree, model_init
from .kernel import build_n_inv_kernel, build_psf_kernel


class PSFConvolve(Model):
    """Convolution with the PSF kernel using FFTs.

    Use the ``build`` class method to create an instance.

    Parameters
    ----------
    sky : Model
        Sky model providing the signal to convolve.
    psf_kernel : np.ndarray
        Pre-computed PSF kernel array.
    """

    fft_kernel: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, sky, psf_kernel):
        self.sky = sky
        self.grid = sky.grid
        self.fft_kernel = build_fft_kernel(psf_kernel, psf_kernel.shape, self.grid.dvol)
        print("psf kernel shape:", self.fft_kernel.shape)
        self.padder = build_padder(self.sky.target.shape, self.fft_kernel.shape)
        self.slicer = build_slicer(
            tuple(k // 2 for k in psf_kernel.shape), self.sky.target.shape
        )
        super().__init__(domain=sky.domain, init=sky.init)

    def __call__(self, x, old_rec=0):
        # TODO: set old_rec to static and add to init (similar to fft-kernel) to avoid recompilation when it changes
        res = self.sky(x) - old_rec
        res = self.padder(res)
        res = fft_convolve(res, self.fft_kernel)
        res = self.slicer(res)
        return res

    @classmethod
    def build(cls, *, sky, RNR_l, psf_kernel_fn="", split=None):
        """Build a PSF convolution operator.

        Loads or creates the PSF kernel and optionally applies
        kernel-splitting for a hybrid high/low-resolution scheme.

        Parameters
        ----------
        sky : Model
            Sky model whose target shape defines the kernel grid.
        RNR_l : Operator
            Padded RNR operator used to compute the kernel (when no
            cached file is available).
        psf_kernel_fn : str, optional
            Path to load/save the PSF kernel. Default is ``''``.
        split : dict, optional
            Kernel-splitting parameters (``size`` and ``factor``). An
            empty dict disables splitting. Default is ``{}``.

        Returns
        -------
        PSFConvolve or PSFSplitConvolve
            The constructed convolution operator.

        Raises
        ------
        ValueError
            If the cached kernel shape does not match the expected shape.
        """
        if split is None:
            split = {}
        if os.path.isfile(psf_kernel_fn):
            with open(psf_kernel_fn, "rb") as f:
                psf_kernel = pickle.load(f)
        else:
            psf_kernel = build_psf_kernel(RNR_l)
            if psf_kernel_fn:
                with open(psf_kernel_fn, "wb") as f:
                    pickle.dump(psf_kernel, f)

        rk_shape = sky.target.shape[:-2] + tuple(s * 2 for s in sky.target.shape[-2:])
        if psf_kernel.shape != rk_shape:
            raise ValueError(
                f"psf kernel has wrong shape, expected {rk_shape}, got {psf_kernel.shape}."
            )

        if split:
            return PSFSplitConvolve(sky, psf_kernel, **split)
        else:
            return cls(sky, psf_kernel)


class PSFSplitConvolve(Model):
    """Hybrid high/low-resolution PSF convolution with kernel-splitting.

    Created via ``PSFConvolve.build`` when *split* parameters are given.

    Parameters
    ----------
    sky : Model
        Sky model providing the signal to convolve.
    psf_kernel : np.ndarray
        Full PSF kernel (split internally).
    size : int
        High-resolution kernel crop size.
    factor : int
        Down-sampling factor for the low-resolution part.
    """

    kernel_high: Any = dataclasses.field(metadata=dict(static=False))
    kernel_low: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, sky, psf_kernel, *, size, factor):
        self.sky = sky
        self.grid = sky.grid
        self.size = size
        self.factor = factor
        self.kernel_high, self.kernel_low = build_split_kernel(
            psf_kernel, self.grid.shape, self.size, self.factor, self.grid.dvol
        )
        print("split psf kernel shapes:", self.kernel_high.shape, self.kernel_low.shape)

        shape_high = (sky.freq.size,) + self.grid.shape
        self.padder_high = build_padder(shape_high, self.kernel_high.shape)
        self.slicer_high = build_slicer((self.size // 2,) * 2, shape_high)

        shape_low = (sky.freq.size,) + tuple(s // self.factor for s in self.grid.shape)
        self.padder_low = build_padder(shape_low, self.kernel_low.shape)
        self.slicer_low = build_slicer(
            tuple(k * (self.factor - 1) // (2 * self.factor) for k in psf_kernel.shape),
            shape_low,
        )

        super().__init__(domain=sky.domain, init=sky.init)

    def __call__(self, x, old_rec=0):
        res = self.sky(x) - old_rec
        res = split_fft_convolve(
            res,
            self.kernel_high,
            self.kernel_low,
            self.padder_high,
            self.padder_low,
            self.slicer_high,
            self.slicer_low,
            self.factor,
        )
        return res


class NInvConvolve(Model):
    """Convolution with the inverse-noise kernel using FFTs.

    Use the ``build`` class method to create an instance.

    Parameters
    ----------
    psf_conv : PSFConvolve or PSFSplitConvolve
        PSF convolution model.
    n_inv_kernel : np.ndarray
        Pre-computed inverse-noise kernel.
    noise_model : NoiseModel
        Learnable noise scaling model.
    """

    fft_kernel: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, psf_conv, n_inv_kernel, noise_model):
        self.psf_conv = psf_conv
        self.grid = psf_conv.grid
        self.fft_kernel = 1.0 / jnp.sqrt(n_inv_kernel)
        print("n inv kernel shape:", self.fft_kernel.shape)
        self.noise_model = noise_model
        super().__init__(
            domain=Vector(domain_tree((self.psf_conv, self.noise_model), error=False)),
            init=model_init((self.psf_conv, self.noise_model), error=False),
        )

    def __call__(self, x, old_rec=0, res_data=0):
        res = self.psf_conv(x, old_rec) - res_data
        ker = self.fft_kernel * self.noise_model(x)
        res = fft_convolve(res, ker)
        return res

    @classmethod
    def build(cls, *, psf_conv, RNR, n_inv_kernel_fn="", noise=None):
        """Build an inverse-noise convolution operator.

        Loads or creates the inverse-noise kernel and wraps it together
        with a ``NoiseModel``.

        Parameters
        ----------
        psf_conv : PSFConvolve or PSFSplitConvolve
            PSF convolution model.
        RNR : Operator
            RNR operator for kernel construction (when no cached file
            is available).
        n_inv_kernel_fn : str, optional
            Path to load/save the noise kernel. Default is ``''``.
        noise : dict or None, optional
            Noise configuration forwarded to ``NoiseModel.build``.

        Returns
        -------
        NInvConvolve
            The constructed noise convolution operator.

        Raises
        ------
        ValueError
            If the cached kernel shape does not match the expected shape.
        """
        if os.path.isfile(n_inv_kernel_fn):
            with open(n_inv_kernel_fn, "rb") as f:
                n_inv_kernel = pickle.load(f)
        else:
            n_inv_kernel = build_n_inv_kernel(RNR, 1e-3)
            if n_inv_kernel_fn:
                with open(n_inv_kernel_fn, "wb") as f:
                    pickle.dump(n_inv_kernel, f)

        nk_shape = psf_conv.target.shape
        if n_inv_kernel.shape != nk_shape:
            raise ValueError(
                f"n inv kernel has wrong shape, expected {nk_shape}, got {n_inv_kernel.shape}."
            )

        noise_model = NoiseModel.build(shape=psf_conv.target.shape, **noise)

        return cls(psf_conv, n_inv_kernel, noise_model)


def downsample(array, factor):
    """Down-sample the last two axes by averaging blocks of *factor*.

    Parameters
    ----------
    array : np.ndarray or jnp.ndarray
        Input array whose last two dimensions are divisible by *factor*.
    factor : int
        Down-sampling factor.

    Returns
    -------
    np.ndarray or jnp.ndarray
        Down-sampled array.
    """
    h, w = array.shape[-2], array.shape[-1]
    assert h % factor == 0 and w % factor == 0, "Dims must be divisible by factor."
    shape = array.shape[:-2] + (h // factor, factor, w // factor, factor)
    return array.reshape(shape).mean(axis=(-3, -1))


def upsample(array, factor):
    """Up-sample the last two axes by repeating pixels.

    Parameters
    ----------
    array : np.ndarray or jnp.ndarray
        Input array.
    factor : int
        Up-sampling factor.

    Returns
    -------
    np.ndarray or jnp.ndarray
        Up-sampled array.
    """
    return array.repeat(factor, axis=-2).repeat(factor, axis=-1)


def fft_convolve_2d(x, kernel):
    """Convolve a single 2-D image with a 2-D kernel via FFT."""
    return ifftn(kernel * fftn(x)).real


def fft_convolve(x, kernel):
    """Batch-convolve along the leading axis using ``fft_convolve_2d``."""
    return smap(fft_convolve_2d, in_axes=(0, 0))(x, kernel)


def split_fft_convolve(
    x, kernel_high, kernel_low, padder_high, padder_low, slicer_high, slicer_low, factor
):
    """Perform a split high/low-resolution FFT convolution.

    Parameters
    ----------
    x : jnp.ndarray
        Input signal array.
    kernel_high : jnp.ndarray
        High-resolution FFT kernel.
    kernel_low : jnp.ndarray
        Low-resolution FFT kernel.
    padder_high, padder_low : callable
        Padding functions for each resolution.
    slicer_high, slicer_low : callable
        Slicing functions to crop the results.
    factor : int
        Down-sampling factor for the low-resolution branch.

    Returns
    -------
    jnp.ndarray
        Combined convolution result.
    """
    if x.ndim == 2:
        x = x[None, :, :]

    x_high = padder_high(x)
    x_high = fft_convolve(x_high, kernel_high)
    x_high = slicer_high(x_high)

    x_low = padder_low(downsample(x, factor))
    x_low = fft_convolve(x_low, kernel_low)
    x_low = upsample(slicer_low(x_low), factor)

    return jnp.squeeze(x_high + x_low)


def build_fft_kernel(kernel, shape, dvol=1.0):
    """Build an FFT-space kernel from a spatial-domain kernel."""
    return jnp.fft.fftn(kernel, shape[-2:], axes=(-2, -1)) * dvol


def build_padder(in_shape, out_shape):
    """Return a zero-padding function from *in_shape* to *out_shape*."""
    p_h, p_w = (o - i for i, o in zip(in_shape[-2:], out_shape[-2:], strict=False))
    pad_width = [(0, 0)] * (len(in_shape) - 2) + [(0, p_h), (0, p_w)]
    return lambda x: jnp.pad(x, pad_width)


def build_slicer(start_indices, out_shape):
    """Return a slicing function that extracts *out_shape* from a padded array."""
    start_indices = (0,) * (len(out_shape) - 2) + start_indices[-2:]
    return lambda x: dynamic_slice(x, start_indices, out_shape)


def build_split_kernel(kernel, shape, size, factor, dvol=1.0):
    """Split a PSF kernel into high- and low-resolution FFT kernels.

    Parameters
    ----------
    kernel : np.ndarray
        Full spatial-domain PSF kernel.
    shape : tuple of int
        Spatial shape of the sky model.
    size : int
        Crop size for the high-resolution kernel.
    factor : int
        Down-sampling factor for the low-resolution kernel.
    dvol : float, optional
        Volume element scaling. Default is 1.0.

    Returns
    -------
    kernel_high : jnp.ndarray
        FFT-space high-resolution kernel.
    kernel_low : jnp.ndarray
        FFT-space low-resolution kernel.
    """
    kernel = kernel[None] if kernel.ndim == 2 else kernel
    n_freq = kernel.shape[0]
    slices = (slice(0, n_freq),)
    slices += tuple(
        slice(k // 2 - size // 2, k // 2 + size // 2) for k in kernel.shape[-2:]
    )

    # high-res kernel
    fshape_high = (n_freq,) + tuple(s + size for s in shape)
    kernel_high = np.array(kernel[slices])
    kernel_high = build_fft_kernel(kernel_high, fshape_high, dvol)

    # low-res kernel
    fshape_low = (n_freq,) + tuple(k // factor for k in kernel.shape[-2:])
    kernel_low = np.array(kernel.copy())
    kernel_low[slices] = 0
    kernel_low = downsample(kernel_low, factor)
    kernel_low = build_fft_kernel(kernel_low, fshape_low, dvol * (factor**2))

    return kernel_high, kernel_low
