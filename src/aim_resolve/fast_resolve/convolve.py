import dataclasses
import jax.numpy as jnp
import numpy as np
import os
import pickle
from jax import vmap
from jax.numpy.fft import fftn, ifftn
from jax.lax import dynamic_slice
from nifty.re import Model, Vector
from typing import Any

from .kernel import build_psf_kernel, build_n_inv_kernel
# from ..model.map import downsample, upsample
from ..model.noise import NoiseModel
from ..optimize.samples import domain_tree, model_init



class PSFConvolve(Model):
    '''Convolution with the PSF kernel using FFTs. Use `build` function to create the model.'''

    fft_kernel: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, sky, psf_kernel):
        self.sky = sky
        self.grid = sky.grid
        self.fft_kernel = build_fft_kernel(psf_kernel, psf_kernel.shape, self.grid.dvol)
        print('psf kernel shape:', self.fft_kernel.shape)
        self.padder = build_padder(self.sky.target.shape, self.fft_kernel.shape)
        self.slicer = build_slicer(tuple(k//2 for k in psf_kernel.shape), self.sky.target.shape)
        super().__init__(domain=sky.domain, init=sky.init)

    def __call__(self, x, old_rec=0):
        res = self.sky(x) - old_rec
        res = self.padder(res)
        res = fft_convolve(res, self.fft_kernel)
        res = self.slicer(res)
        return res

    @classmethod
    def build(cls, *, sky, RNR_l, psf_kernel_fn='', split=0):
        '''
        Build the convolution with the psf kernel.

        Parameters
        ----------
        model: jft.Model
            The input model (RNRApproximation).
        RNR_l : ift.Operator
            The RNR operator acting on the padded model space.
        psf_kernel_fn : str
            The filename to load or save the psf kernel. Default is None.
        split : int
            The high-resolution kernel size for a kernel-split. Default is 0 (no split).
        '''
        if os.path.isfile(psf_kernel_fn):
            psf_kernel = pickle.load(open(psf_kernel_fn, 'rb'))
        else:
            psf_kernel = build_psf_kernel(RNR_l)
            if psf_kernel_fn:
                pickle.dump(psf_kernel, open(psf_kernel_fn, 'wb'))

        rk_shape = sky.target.shape[:-2] + tuple(s*2 for s in sky.target.shape[-2:])
        if psf_kernel.shape != rk_shape:
            raise ValueError(f'psf kernel has wrong shape, expected {rk_shape}, got {psf_kernel.shape}.')
        
        if split > 0:
            return PSFSplitConvolve(sky, psf_kernel, split)
        else:
            return cls(sky, psf_kernel)
    


class PSFSplitConvolve(Model):
    '''Convolution with the PSF kernel using FFTs and kernel-splitting. Use `build` function of `PSFConvolve` to create the model.'''

    fft_kernels: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, sky, psf_kernel, split):
        self.sky = sky
        self.grid = sky.grid
        self.fft_kernel = build_split_kernel(psf_kernel, self.grid.shape, split, self.grid.dvol)
        print('split psf kernel shape:', self.fft_kernel.shape)
        shape1 = (sky.freq.size, ) + self.grid.shape
        shape2 = (sky.freq.size, ) + tuple(s//2 for s in self.grid.shape)
        self.padder1 = build_padder(shape1, self.fft_kernel.shape)
        self.padder2 = build_padder(shape2, self.fft_kernel.shape)
        self.slicer1 = build_slicer((split//2,) * 2, shape1)
        self.slicer2 = build_slicer(tuple(k//4 for k in psf_kernel.shape), shape2)
        super().__init__(domain=sky.domain, init=sky.init)

    def __call__(self, x, old_rec=0):
        res = self.sky(x) - old_rec
        res = split_fft_convolve(res, self.fft_kernel, self.padder1, self.padder2, self.slicer1, self.slicer2)
        return res


class NInvConvolve(Model):
    '''Convolution with the inverse noise kernel using FFTs. Use `build` function to create the model.'''

    fft_kernel: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, psf_conv, n_inv_kernel, noise_model):
        self.psf_conv = psf_conv
        self.grid = psf_conv.grid
        self.fft_kernel = 1. / jnp.sqrt(n_inv_kernel)
        print('n inv kernel shape:', self.fft_kernel.shape)
        self.noise_model = noise_model
        super().__init__(
            domain = Vector(domain_tree((self.psf_conv, self.noise_model), error=False)), 
            init = model_init((self.psf_conv, self.noise_model), error=False),
        )

    def __call__(self, x, old_rec=0, res_data=0):
        res = self.psf_conv(x, old_rec) - res_data
        ker = self.fft_kernel * self.noise_model(x)
        res = fft_convolve(res, ker)
        return res

    @classmethod
    def build(cls, *, psf_conv, RNR, n_inv_kernel_fn='', noise=None):
        '''
        Build the convolution with the inverse noise kernel.

        Parameters
        ----------
        psf_conv: jft.Model
            The input model (PSFConvolve or PSFSplitConvolve).
        RNR : ift.Operator
            The RNR operator acting on the model space.
        n_inv_kernel_fn : str
            The filename to load or save the noise kernel. Default is None.
        noise_model : ift.Operator
            The noise model that should be used for the inference. Default is None.
        '''
        if os.path.isfile(n_inv_kernel_fn):
            n_inv_kernel = pickle.load(open(n_inv_kernel_fn, "rb"))
        else:
            n_inv_kernel = build_n_inv_kernel(RNR, 1e-3)
            if n_inv_kernel_fn:
                pickle.dump(n_inv_kernel, open(n_inv_kernel_fn, "wb"))

        nk_shape = psf_conv.target.shape
        if n_inv_kernel.shape != nk_shape:
            raise ValueError(f'n inv kernel has wrong shape, expected {nk_shape}, got {n_inv_kernel.shape}.')

        noise_model = NoiseModel.build(shape=psf_conv.target.shape, **noise)

        return cls(psf_conv, n_inv_kernel, noise_model)



def downsample(array, factor):
    h, w = array.shape[-2], array.shape[-1]
    assert h % factor == 0 and w % factor == 0, "Dims must be divisible by factor."
    shape = array.shape[:-2] + (h // factor, factor, w // factor, factor)
    return array.reshape(shape).mean(axis=(-3, -1))
    

def upsample(array, factor):
    return array.repeat(factor, axis=-2).repeat(factor, axis=-1)



def fft_convolve_2d(x, kernel):
    return ifftn(kernel * fftn(x)).real


def fft_convolve(x, kernel):
    if x.ndim == 2:
        return fft_convolve_2d(x, kernel)
    elif x.ndim == 3:
        return vmap(fft_convolve_2d, in_axes=(0, 0))(x, kernel)
    else:
        raise ValueError("Input must be 2D (x,y) or 3D (f,x,y)")
    

def split_fft_convolve(x, kernel, padder1, padder2, slicer1, slicer2):
    x = x[None] if x.ndim == 2 else x
    x1 = padder1(x)
    x2 = padder2(downsample(x, 2))
    xx = jnp.concatenate([x1, x2], axis=0)
    rr = fft_convolve(xx, kernel)
    rr = rr.reshape(2, -1, *rr.shape[1:])
    r1 = slicer1(rr[0])
    r2 = upsample(slicer2(rr[1]), 2)
    return r1 + r2



def build_fft_kernel(kernel, shape, dvol=1.):
    return jnp.fft.fftn(kernel, shape[-2:], axes=(-2, -1)) * dvol


def build_padder(in_shape, out_shape):
    p_h, p_w = (o - i for i, o in zip(in_shape[-2:], out_shape[-2:]))
    pad_width = [(0, 0)] * (len(in_shape) - 2) + [(0, p_h), (0, p_w)]
    return lambda x: jnp.pad(x, pad_width)


def build_slicer(start_indices, out_shape):
    start_indices = (0,) * (len(out_shape) - 2) + start_indices[-2:]
    return lambda x: dynamic_slice(x, start_indices, out_shape)



def build_split_kernel(kernel, shape, size, dvol=1.):
    kernel = kernel[None] if kernel.ndim == 2 else kernel
    fshape = tuple(s+size for s in shape)
    n_freq = kernel.shape[0]
    split_kernel = np.zeros((2, n_freq) + fshape, dtype='complex128')

    for n in range(1, 3):
        if n == 1:
            slc_in = (slice(0, 0), slice(0, 0))
            slc_out = tuple(slice(k//2 - size//2, k//2 + size//2) for k in kernel.shape[-2:])
        else:
            slc_in = slc_out
            slc_out = tuple(slice(0, k) for k in kernel.shape[-2:])
        
        for f in range(n_freq):
            ker = np.array(kernel[f][slc_out])
            ker[slc_in] = 0

            if n == 2:
                ker = downsample(ker, 2)

            fft_ker = build_fft_kernel(ker, fshape, dvol * (n**2))
            split_kernel[n-1][f] = fft_ker

    split_kernel = split_kernel.reshape(-1, *fshape)

    return split_kernel
