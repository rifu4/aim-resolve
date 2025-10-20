import dataclasses
import jax.numpy as jnp
import numpy as np
import os
import pickle
from jax import vmap
from jax.numpy.fft import fftn, ifftn
from jax.lax import dynamic_slice
from nifty8.re import Model, Vector
from typing import Any

from .kernel import build_response_kernel, build_noise_kernel
from ..model.map import downsample, upsample
from ..model.noise import NoiseModel
from ..optimize.samples import domain_tree, model_init



class PSFConvolve(Model):
    '''Convolution with the PSF kernel using FFTs. Use `build` function to create the model.'''

    fft_kernel: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, sky, kernel):
        self.sky = sky
        self.grid = sky.grid
        self.fft_kernel = build_fft_kernel(kernel, kernel.shape, self.grid.dvol)
        print('fft_kernel.shape:', self.fft_kernel.shape)
        self.padder = build_padder(self.grid.shape, self.fft_kernel.shape)
        self.slicer = build_slicer(tuple(k//2 for k in kernel.shape), self.grid.shape)
        super().__init__(domain=sky.domain, init=sky.init)

    def __call__(self, x, old_rec=0):
        y = self.sky(x) - old_rec
        y = self.padder(y)
        y = ifftn(self.fft_kernel * fftn(y)).real
        y = self.slicer(y)
        return y

    @classmethod
    def build(cls, *, sky, RNR_l, response_kernel_fn='', split=0):
        '''
        Build the convolution with the response kernel.

        Parameters
        ----------
        model: jft.Model
            The input model (RNRApproximation).
        RNR_l : ift.Operator
            The RNR response operator acting on the padded model space.
        response_kernel_fn : str
            The filename to load or save the response kernel. Default is None.
        split : int
            The high-resolution kernel size for a kernel-split. Default is 0 (no split).
        '''
        if os.path.isfile(response_kernel_fn):
            response_kernel = pickle.load(open(response_kernel_fn, 'rb'))
        else:
            response_kernel = build_response_kernel(RNR_l)
            if response_kernel_fn:
                pickle.dump(response_kernel, open(response_kernel_fn, 'wb'))
        
        if split > 0:
            return PSFSplitConvolve(sky, response_kernel, split)
        else:
            return cls(sky, response_kernel)
    


class PSFSplitConvolve(Model):
    '''Convolution with the RNR response kernel using FFTs and kernel-splitting. Use `build` function of `RNRConvolve` to create the model.'''

    fft_kernels: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, sky, kernel, split):
        self.sky = sky
        self.grid = sky.grid
        self.fft_kernels = build_split_kernel(kernel, self.grid.shape, split, self.grid.dvol)
        print('fft_kernel.shape:', self.fft_kernels.shape)
        self.padder1 = build_padder(self.grid.shape, self.fft_kernels.shape[1:])
        self.padder2 = build_padder(tuple(s//2 for s in self.grid.shape), self.fft_kernels.shape[1:])
        self.slicer1 = build_slicer((split//2,) * 2, self.grid.shape)
        self.slicer2 = build_slicer(tuple(k//4 for k in kernel.shape), tuple(s//2 for s in self.grid.shape))
        super().__init__(domain=sky.domain, init=sky.init)

    def __call__(self, x, old_rec=0):
        y = self.sky(x) - old_rec
        y1 = self.padder1(y)
        y2 = self.padder2(downsample(y, 2))
        yy = jnp.stack([y1, y2])
        yy = vmap(lambda ki,xi: ifftn(ki * fftn(xi)).real)(self.fft_kernels, yy)
        y1 = self.slicer1(yy[0])
        y2 = upsample(self.slicer2(yy[1]), 2)
        return y1 + y2



class NInvConvolve(Model):
    '''Convolution with the inverse noise kernel using FFTs. Use `build` function to create the model.'''

    fft_kernel: Any = dataclasses.field(metadata=dict(static=False))

    def __init__(self, psf_conv, kernel, noise_model):
        self.psf_conv = psf_conv
        self.grid = psf_conv.grid
        self.fft_kernel = 1. / jnp.sqrt(kernel)
        self.noise_model = noise_model
        super().__init__(
            domain = Vector(domain_tree((self.psf_conv, self.noise_model), error=False)), 
            init = model_init((self.psf_conv, self.noise_model), error=False),
        )

    def __call__(self, x, old_rec=0, res_data=0):
        y = self.psf_conv(x, old_rec) - res_data
        y = ifftn(self.noise_model(x) * self.fft_kernel * fftn(y)).real
        return y

    @classmethod
    def build(cls, *, psf_conv, RNR, noise_kernel_fn='', noise=None):
        '''
        Build the convolution with the inverse noise kernel.

        Parameters
        ----------
        psf_conv: jft.Model
            The input model (PSFConvolve or PSFSplitConvolve).
        RNR : ift.Operator
            The RNR response operator acting on the model space.
        noise_kernel_fn : str
            The filename to load or save the noise kernel. Default is None.
        noise_model : ift.Operator
            The noise model that should be used for the inference. Default is None.
        '''
        if os.path.isfile(noise_kernel_fn):
            noise_kernel = pickle.load(open(noise_kernel_fn, "rb"))
        else:
            noise_kernel = build_noise_kernel(RNR, 1e-3)
            if noise_kernel_fn:
                pickle.dump(noise_kernel, open(noise_kernel_fn, "wb"))

        noise_model = NoiseModel.build(shape=psf_conv.grid.shape, **noise)

        return cls(psf_conv, noise_kernel, noise_model)



def build_fft_kernel(kernel, shape, dvol=1.):
    return jnp.fft.fftn(kernel, shape) * dvol


def build_padder(in_shape, out_shape):
    p0, p1 = [(s - a) for s,a in zip(out_shape, in_shape)]
    return lambda x: jnp.pad(x, ((0, p0), (0, p1)))


def build_slicer(start_indices, out_shape):
    s0, s1 = start_indices
    return lambda x: dynamic_slice(x, (s0, s1), out_shape)


def build_split_kernel(kernel, shape, size, dvol=1.):
    fshape = tuple(s+size for s in shape)

    fkernel = np.zeros((2,) +  fshape, dtype='complex128')
    for f in range(1, 3):
        if f == 1:
            slc_in = (slice(0, 0), slice(0, 0))
            slc_out = tuple(slice(k//2 - size//2, k//2 + size//2) for k in kernel.shape)
        else:
            slc_in = slc_out
            slc_out = tuple(slice(0, k) for k in kernel.shape)

        ker = np.array(kernel[slc_out])
        ker[slc_in] = 0

        if f == 2:
            ker = downsample(ker, f)
        
        fker = build_fft_kernel(ker, fshape, dvol * (f**2))
        fkernel[f-1] = fker

    return fkernel
