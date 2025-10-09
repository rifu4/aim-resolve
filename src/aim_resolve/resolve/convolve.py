import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.numpy.fft import fftn, ifftn
from jax.lax import dynamic_slice


def build_fft_kernel(kernel, shape, dvol=1.):
    return np.fft.fftn(kernel, shape) * dvol


def build_padder(in_shape, out_shape):
    p0, p1 = [(s - a) for s,a in zip(out_shape, in_shape)]
    return lambda x: jnp.pad(x, ((0, p0), (0, p1)))


def build_slicer(start_indices, out_shape):
    s0, s1 = start_indices
    return lambda x: dynamic_slice(x, (s0, s1), out_shape)


def fast_fftconvolve(kernel, shape, dvol=1.):
    fft_kernel = build_fft_kernel(kernel, kernel.shape, dvol)
    fft_kernel = jnp.array(fft_kernel)
    print('fast fft shape:', fft_kernel.shape)

    padder = build_padder(shape, fft_kernel.shape)
    slicer = build_slicer(tuple(k//2 for k in kernel.shape), shape)

    def fun(x):
        x = padder(x)
        r = ifftn(fft_kernel * fftn(x)).real
        r = slicer(r)
        return r

    return fun


def downsample(array, factor):
    return array.reshape(array.shape[0]//factor, factor, array.shape[1]//factor, factor).mean(axis=(1,3))
    

def upsample(array, factor):
    return array.repeat(factor, axis=0).repeat(factor, axis=1)


def build_split_kernel(kernel, shape, size, dvol=1.):
    fshape = tuple(s+size for s in shape)
    print('split fft shape:', fshape)

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


def split_fftconvolve(kernel, shape, size, dvol=1.):
    fft_kernels = build_split_kernel(kernel, shape, size, dvol)
    fft_kernels = jnp.array(fft_kernels)

    padder1 = build_padder(shape, fft_kernels.shape[1:])
    padder2 = build_padder(tuple(s//2 for s in shape), fft_kernels.shape[1:])
    slicer1 = build_slicer((size//2,) * 2, shape)
    slicer2 = build_slicer(tuple(k//4 for k in kernel.shape), tuple(s//2 for s in shape))

    def fun(x):
        x1 = padder1(x)
        x2 = padder2(downsample(x, 2))
        xx = jnp.stack([x1, x2])
        rr = vmap(lambda ki,xi: ifftn(ki * fftn(xi)).real)(fft_kernels, xx)
        r1 = slicer1(rr[0])
        r2 = upsample(slicer2(rr[1]), 2)
        return r1 + r2

    return fun
