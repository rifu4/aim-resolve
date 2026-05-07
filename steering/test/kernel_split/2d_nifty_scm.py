import jax
import jax.numpy as jnp
import nifty8 as ift
import nifty8.re as jft
import numpy as np
import resolve as rve
from jax.lax import slice as jax_slice
from jax.scipy.signal import convolve as jax_convolve

from aim_resolve import (
    Observation,
    SignalSpace,
    check_type,
    correlated_field_model,
    optimize_kl,
    plot_arrays,
)

jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(123)


# %%
class SignalGrid:
    """Class to represent a signal grid at a specific location in the sky. Use `build` function to create the grid."""

    def __init__(self, space, center=(0, 0), factor=1, distances=(1.0, 1.0)):
        check_type(space, tuple, int)
        check_type(factor, int)
        check_type(center, tuple, int)
        check_type(distances, tuple, float)

        self.space = space
        self.center = center
        self.factor = factor
        self.shape = tuple(d * self.factor for d in self.space)
        self.distances = distances

    def __repr__(self):
        return f"SignalGrid(space={self.space}, center={self.center}, factor={self.factor}, distances={self.distances})"

    def __mul__(self, other):
        return self.multiply_space(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    @property
    def dom(self):
        return np.array(self.space)

    @property
    def cen(self):
        return np.array(self.center)

    @property
    def shp(self):
        return np.array(self.shape)

    @property
    def size(self):
        return self.shp.prod()

    @property
    def dis(self):
        return np.array(self.distances) / self.factor

    @property
    def coos(self):
        coos = np.indices(self.shp).astype(float)
        coos_T = coos.T.reshape(-1, 2)
        coos_T -= 0.5 * (self.shp - 1)
        coos_T /= self.factor
        coos_T += self.cen
        return coos_T.reshape(coos.T.shape).T

    @property
    def llp(self):
        return self.cen - 0.5 * (self.shp - 1) / self.factor

    @property
    def urp(self):
        return self.cen + 0.5 * (self.shp - 1) / self.factor

    def update(self, space=None, center=None, factor=None, distances=None):
        space = self.space if space is None else space
        center = self.center if center is None else center
        factor = self.factor if factor is None else factor
        distances = self.distances if distances is None else distances
        return SignalGrid(space, center, factor, distances)

    def multiply_space(self, factor):
        return SignalGrid.update(
            self, space=tuple(int(si * factor) for si in self.space)
        )


def downsample(array, factor):
    if factor == 1:
        return array
    if factor in (2, 4, 8):
        return array.reshape(
            array.shape[0] // factor, factor, array.shape[1] // factor, factor
        ).mean(axis=(1, 3))
    else:
        raise ValueError(f"Invalid zoom factor: {factor}")


def upsample(array, factor):
    if factor == 1:
        return array
    elif factor in (2, 4, 8):
        return array.repeat(factor, axis=0).repeat(factor, axis=1)
    else:
        raise ValueError(f"Invalid zoom factor: {factor}")


def map_array(in_array, in_grid, out_grid):
    factor = 1
    if out_grid.factor < in_grid.factor:
        in_array = downsample(in_array, in_grid.factor // out_grid.factor)
        in_grid = in_grid.update(factor=out_grid.factor)
    else:
        factor = out_grid.factor // in_grid.factor
        out_grid = out_grid.update(factor=in_grid.factor)

    if in_grid.center != out_grid.center or in_grid.space != out_grid.space:
        out_array = jnp.zeros(out_grid.shape)

        llp_dif = (out_grid.llp - in_grid.llp).astype("int64")
        urp_dif = (out_grid.urp - in_grid.urp).astype("int64")

        in_min = np.maximum(llp_dif * in_grid.factor, 0)
        in_max = np.minimum(urp_dif * in_grid.factor + in_grid.shp, in_grid.shp)
        in_slc = tuple(slice(in_min[i], in_max[i]) for i in range(2))

        out_min = np.maximum(-llp_dif * out_grid.factor, 0)
        out_max = np.minimum(out_grid.shp - urp_dif * out_grid.factor, out_grid.shp)
        out_slc = tuple(slice(out_min[i], out_max[i]) for i in range(2))

        out_array = out_array.at[out_slc].set(in_array[in_slc])
    else:
        out_array = in_array.copy()

    if factor > 1:
        out_array = upsample(out_array, factor)

    return out_array


# %%
def build_exact_responses(
    obs,
    space,
):
    check_type(space, SignalSpace)
    sdom = ift.RGSpace(space.shape, distances=space.distances)
    sky_dom = rve.default_sky_domain(sdom=sdom)
    R = rve.InterferometryResponse(obs, sky_dom, True, 1e-9, verbosity=0, nthreads=8)

    space_l = space.multiply_fov(2)
    sdom_l = ift.RGSpace(space_l.shape, distances=space_l.distances)
    sky_dom_l = rve.default_sky_domain(sdom=sdom_l)
    R_l = rve.InterferometryResponse(
        obs, sky_dom_l, True, 1e-9, verbosity=0, nthreads=8
    )

    dch_l = ift.DomainChangerAndReshaper(R_l.domain[3], R_l.domain)
    R_l = R_l @ dch_l
    dch = ift.DomainChangerAndReshaper(R.domain[3], R.domain)
    R = R @ dch

    N_inv = ift.DiagonalOperator(obs.weight)
    RNR = R.adjoint @ N_inv @ R
    RNR_l = R_l.adjoint @ N_inv @ R_l

    return R, R_l, RNR, RNR_l


def compute_psf_kernel(RNR_l):
    dom_l = RNR_l.domain
    shp_l = dom_l.shape

    delta = np.zeros(shp_l)
    delta[shp_l[0] // 2, shp_l[1] // 2] = 1 / dom_l.scalar_weight()
    delta = ift.makeField(dom_l, delta)
    kernel = RNR_l(delta)

    return kernel.val


obs = Observation.load("/Users/rf/Development/data/eso_986-1137mhz.npz")
obs = obs.to_double_precision()
obs = obs.to_resolve_obs()

space = SignalSpace.build(shape=(512, 512), fov=("1deg", "1deg"))

R, R_l, RNR, RNR_l = build_exact_responses(obs, space)
kernel = compute_psf_kernel(RNR_l)

plot_arrays(kernel, norm="log", dpi=100)


# %%
class SignalModel(jft.Model):
    def __init__(self, grid, i0, prefix="sm"):
        self.grid = grid
        self.i0 = i0
        self.prefix = prefix
        super().__init__(domain=self.i0.domain, init=self.i0.init)

    def __call__(self, x, *, out_grid=None):
        res = jnp.exp(self.i0(x))
        if out_grid:
            return map_array(res, self.grid, out_grid)
        return res


# %%
def jax_fft(grid):
    check_type(grid, SignalGrid)
    dvol = grid.dis.prod()
    return lambda x: dvol * jnp.fft.fftn(x)


def jax_ifft(grid):
    check_type(grid, SignalGrid)
    dvol = 1.0 / (grid.shp * grid.dis).prod()
    npix = grid.size
    return lambda x: dvol * npix * jnp.fft.ifftn(x)


def jax_fft_convolve(fft_kernel, grid):
    grid_l = grid.update(space=tuple(s * 2 for s in grid.space))
    fft_l = jax_fft(grid_l)
    ifft_l = jax_ifft(grid_l)
    padding = grid.shp // 2

    def jax_fft_conv(array):
        p_array = jnp.pad(array, (2 * (padding[0],), 2 * (padding[1],)))
        c_array = ifft_l(fft_kernel * fft_l(p_array)).real
        s_array = jax_slice(c_array, padding, padding + grid.shp)
        return s_array

    return jax_fft_conv


def shift_kernel(kernel):
    s_kernel = jnp.roll(kernel, -kernel.shape[0] // 2, axis=0)
    s_kernel = jnp.roll(s_kernel, -kernel.shape[1] // 2, axis=1)
    return s_kernel


def build_fft_kernel(kernel, grid):
    check_type(grid, SignalGrid)
    grid_l = grid.update(space=tuple(s * 2 for s in grid.space))
    s_kernel = shift_kernel(kernel)
    f_kernel = jax_fft(grid_l)(s_kernel)
    return f_kernel


# %%
class HighResResponse(jft.Model):
    def __init__(self, model, kernel):
        self.model = model
        self.kernel = kernel
        self.dvol = model.grid.dis.prod()
        super().__init__(domain=self.model.domain, init=self.model.init)

    def __call__(self, x):
        sig = self.model(x)
        rsp = jax_convolve(sig, self.kernel, mode="same", method="fft") * self.dvol
        return rsp


class LowResResponse(jft.Model):
    def __init__(self, model, kernel):
        self.model = model
        self.factor = model.grid.factor
        self.kernel = downsample(kernel, self.factor)
        self.dvol = self.model.grid.dis.prod()
        super().__init__(domain=self.model.domain, init=self.model.init)

    def __call__(self, x):
        sig = self.model(x)
        fsig = downsample(sig, self.factor)
        frsp = (
            jax_convolve(fsig, self.kernel, mode="same", method="fft")
            * self.dvol
            * (self.factor**2)
        )
        rsp = upsample(frsp, self.factor)
        return rsp


def split_kernel(kernel, factors):
    ksize = kernel.shape[0]
    fsize = ksize // max(factors)

    skernel = np.zeros((len(factors), fsize, fsize))
    for i, f in enumerate(factors):
        ker = np.zeros_like(kernel)
        if i == 0:
            slc_in = slice(0, 0)
            slc_out = slice(ksize // 2 - fsize // 2, ksize // 2 + fsize // 2)
        else:
            slc_in = slc_out
            slc_out = slice(ksize // 2 - f * fsize // 2, ksize // 2 + f * fsize // 2)
        ker[slc_out, slc_out] = kernel[slc_out, slc_out]
        ker[slc_in, slc_in] = 0
        skernel[i] = downsample(ker[slc_out, slc_out], f)

    return skernel


class MultiKernelResponse(jft.Model):
    def __init__(self, model, kernel):
        self.model = model
        self.factors = np.array([f for f in [1, 2, 4, 8] if f <= model.grid.factor])
        print("factors:", self.factors)
        self.kernels = split_kernel(kernel, self.factors)
        print("kernels shape:", self.kernels.shape)
        self.dvol = model.grid.dis.prod()
        super().__init__(domain=self.model.domain, init=self.model.init)

    def __call__(self, x):
        sig = self.model(x)
        rsp = jnp.zeros_like(sig)
        for i, f in enumerate(self.factors):
            fsig = downsample(sig, f)
            frsp = (
                jax_convolve(fsig, self.kernels[i], mode="same", method="fft")
                * self.dvol
                * (f**2)
            )
            rsp += upsample(frsp, f)
        return rsp


# %%
shape = (512, 512)
fmax = 4

space = SignalSpace.build(shape=tuple(s // fmax for s in shape), fov=("2deg", "2deg"))

grid = SignalGrid(
    space=tuple(s // fmax for s in shape),
    center=(0, 0),
    factor=fmax,
    distances=space.distances,
)
print(grid)

bg, pw = correlated_field_model(
    prefix="bg ",
    shape=grid.shape,
    distances=grid.distances,
    offset_mean=12,
    offset_std=(1, 1),
    fluctuations=(5, 1),
    loglogavgslope=(-2, 0.5),
    flexibility=(1.2, 0.4),
    asperity=(0.2, 0.2),
)
signal = SignalModel(grid, bg, prefix="bg")


hr_response = HighResResponse(signal, kernel)

lr_response = LowResResponse(signal, kernel)

mk_response = MultiKernelResponse(signal, kernel)


key, subkey = jax.random.split(key)
prior_xi = signal.init(subkey)

plot_arrays(signal(prior_xi), rows=1, norm="log")

plot_arrays(
    [hr_response(prior_xi), lr_response(prior_xi), mk_response(prior_xi)],
    rows=1,
    norm="log",
)

# %%
import pickle

from aim_resolve import OptimizeKLConfig, get_builders, map_signal

base_yml = "/Users/rf/Development/packages/aim-resolve/steering/runs/fast_vi_512c/files/base.yml"
exp_yml = "/Users/rf/Development/packages/aim-resolve/steering/runs/fast_vi_512c/files/5_pre.yml"

optim_cfg = OptimizeKLConfig.from_file((base_yml, exp_yml), get_builders, "total")


sky_bg = optim_cfg.instantiate_sec("sky_bg.3")
sky_o0 = optim_cfg.instantiate_sec("sky_o0.3")
sky_o1 = optim_cfg.instantiate_sec("sky_o1.3")
sky_o2 = optim_cfg.instantiate_sec("sky_o2.3")
sky_t0 = optim_cfg.instantiate_sec("sky_t0.3")
sky_p0 = optim_cfg.instantiate_sec("sky_p0.3")


with open(
    "/Users/rf/Development/packages/aim-resolve/steering/runs/fast_vi_512c/opt/3_rec/last.pkl",
    "rb",
) as f:
    samples, *_ = pickle.load(f)

rec_bg = upsample(downsample(samples.mean(sky_bg), fmax), fmax)
rec_o0 = map_signal(samples.mean(sky_o0), sky_o0.space, sky_bg.space)
rec_o1 = map_signal(samples.mean(sky_o1), sky_o1.space, sky_bg.space)
rec_o2 = map_signal(samples.mean(sky_o2), sky_o2.space, sky_bg.space)
rec_t0 = samples.mean(sky_t0)
rec_p0 = samples.mean(sky_p0)

rec = rec_bg + rec_o0 + rec_o1 + rec_o2 + rec_t0 + rec_p0
# rec = downsample(rec, 2)


def generate_data(rec, kernel, grid, n_std=1e-4):
    truth = jax_convolve(rec, kernel, mode="same", method="fft") * grid.dis.prod()
    noise = n_std * jax.random.normal(key, truth.shape)
    data = truth + noise
    return data


data = generate_data(rec, kernel, grid)


plot_arrays(rec, norm="log", vmin=3e2, vmax=rec.max(), dpi=100)
plot_arrays(data, rows=1, norm="log")


# %%
lh_dct = dict(
    data=data,
    model=hr_response,
    noise_cov_inv=None,
    noise_std_inv=(1e-4) ** -1,
)


def callback(samples, it):
    plot_arrays([samples.mean(signal)], norm="log")


hr_samples, _ = optimize_kl(
    likelihood=lh_dct,
    key=45,
    n_total_iterations=5,
    n_samples=0,
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name=None,
            cg_kwargs=dict(name=None),
            miniter=10,
            maxiter=10,
        ),
    ),
    callback=callback,
    sample_mode="linear_resample",
)

# %%
lh_dct = dict(
    data=data,
    model=lr_response,
    noise_cov_inv=None,
    noise_std_inv=(1e-4) ** -1,
)


def callback(samples, it):
    plot_arrays([samples.mean(signal)], norm="log")


lr_samples, _ = optimize_kl(
    likelihood=lh_dct,
    key=45,
    n_total_iterations=5,
    n_samples=0,
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name=None,
            cg_kwargs=dict(name=None),
            miniter=10,
            maxiter=10,
        ),
    ),
    callback=callback,
    sample_mode="linear_resample",
)

# %%
lh_dct = dict(
    data=data,
    model=mk_response,
    noise_cov_inv=None,
    noise_std_inv=(1e-4) ** -1,
)


def callback(samples, it):
    plot_arrays([samples.mean(signal)], norm="log")


mk_samples, _ = optimize_kl(
    likelihood=lh_dct,
    key=45,
    n_total_iterations=5,
    n_samples=0,
    kl_kwargs=dict(
        minimize_kwargs=dict(
            name=None,
            cg_kwargs=dict(name=None),
            miniter=10,
            maxiter=10,
        ),
    ),
    callback=callback,
    sample_mode="linear_resample",
)

# %%
plot_arrays(
    [hr_samples.mean(signal), lr_samples.mean(signal), mk_samples.mean(signal), rec],
    norm="log",
    rows=2,
    ticks=0,
    cbar=[False, True, False, True],
    vmin=1e2,
    vmax=rec.max(),
)

# %%
plot_arrays(
    [
        np.abs(mk_samples.mean(signal) - hr_samples.mean(signal))
        / hr_samples.mean(signal),
        np.abs(lr_samples.mean(signal) - hr_samples.mean(signal))
        / hr_samples.mean(signal),
    ],
    norm="log",
    rows=1,
    ticks=0,
)

# %%
