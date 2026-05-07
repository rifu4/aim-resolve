from copy import deepcopy

import jax
import jax.numpy as jnp
import nifty8 as ift
import numpy as np
import resolve as rve
from jax.lax import slice as jax_slice

from aim_resolve import Observation, SignalSpace, check_type, plot_arrays

jax.config.update("jax_enable_x64", True)


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
        out_array = np.zeros(out_grid.shape)

        llp_dif = (out_grid.llp - in_grid.llp).astype("int64")
        urp_dif = (out_grid.urp - in_grid.urp).astype("int64")

        in_min = np.maximum(llp_dif * in_grid.factor, 0)
        in_max = np.minimum(urp_dif * in_grid.factor + in_grid.shp, in_grid.shp)
        in_slc = tuple(slice(in_min[i], in_max[i]) for i in range(2))

        out_min = np.maximum(-llp_dif * out_grid.factor, 0)
        out_max = np.minimum(out_grid.shp - urp_dif * out_grid.factor, out_grid.shp)
        out_slc = tuple(slice(out_min[i], out_max[i]) for i in range(2))

        out_array[out_slc] = in_array[in_slc]
    else:
        out_array = in_array.copy()

    if factor > 1:
        out_array = upsample(out_array, factor)

    return out_array


class PointGrid:
    """Class to represent a signal grid at a specific location in the sky. Use `build` function to create the grid."""

    def __init__(self, coordinates, factor=1, n_copies=1):
        check_type(coordinates, tuple, tuple, float)
        check_type(factor, int)
        check_type(n_copies, int)

        self.coordinates = coordinates
        self.factor = factor
        self.n_copies = n_copies
        self.shape = (1, 1)

    def __repr__(self):
        return f"PointGrid(coordinates={self.coordinates}, factor={self.factor}, n_copies={self.n_copies})"

    @property
    def coos(self):
        return np.array(self.coordinates)

    @property
    def shp(self):
        return np.array(self.shape)

    @property
    def size(self):
        return self.shp.prod()

    def update(self, coordinates=None, factor=None, n_copies=None):
        coordinates = self.coordinates if coordinates is None else coordinates
        factor = self.factor if factor is None else factor
        n_copies = self.n_copies if n_copies is None else n_copies
        return PointGrid(coordinates, factor, n_copies)


def map_point(in_array, in_grid, out_grid):
    if out_grid.factor < in_grid.factor:
        factor = out_grid.factor / in_grid.factor
        in_coos = (
            np.floor(in_grid.coos * out_grid.factor) / out_grid.factor
            + 0.5 / out_grid.factor
        )
        in_grid = in_grid.update(
            coordinates=tuple(map(tuple, in_coos.tolist())), factor=out_grid.factor
        )
    else:
        factor = out_grid.factor // in_grid.factor
        out_grid = out_grid.update(factor=in_grid.factor)

    out_array = np.zeros(out_grid.shape)

    for i in range(in_grid.n_copies):
        llp_dif = (out_grid.factor * (in_grid.coos[i] - out_grid.llp)).astype("int64")
        out_array[llp_dif[0], llp_dif[1]] += in_array[i, 0, 0] * factor**2

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


def shift_kernel(kernel):
    shifted_kernel = np.roll(kernel, -kernel.shape[0] // 2, axis=0)
    shifted_kernel = np.roll(shifted_kernel, -kernel.shape[1] // 2, axis=1)
    return shifted_kernel


def fft_convolve_2d(array, kernel, grid):
    fft_kernel = transform_psf_kernel(kernel, grid)
    res = apply_psf_kernel(fft_kernel, grid)(array)
    return res


def transform_psf_kernel(kernel, grid):
    check_type(grid, SignalGrid)
    grid_l = grid.update(space=tuple(s * 2 for s in grid.space))
    shifted_kernel = shift_kernel(kernel)
    fourier_kernel = fft_fun(grid_l)(shifted_kernel)
    return fourier_kernel


def apply_psf_kernel(psf_kernel, grid):
    grid_l = grid.update(space=tuple(s * 2 for s in grid.space))

    fft_l = fft_fun(grid_l)
    ifft_l = ifft_fun(grid_l)
    psf_kernel = jnp.array(psf_kernel)
    padding = grid.shp // 2

    def apply_psf(x):
        res = jnp.pad(x, (2 * (padding[0],), 2 * (padding[1],)))
        res = ifft_l(psf_kernel * fft_l(res)).real
        res = jax_slice(res, padding, padding + grid.shp)
        return res

    return apply_psf


def fft_fun(grid):
    check_type(grid, SignalGrid)
    dvol = grid.dis.prod()
    return lambda x: dvol * jnp.fft.fftn(x)


def ifft_fun(grid):
    check_type(grid, SignalGrid)
    dvol = 1.0 / (grid.shp * grid.dis).prod()
    npix = grid.size
    return lambda x: dvol * npix * jnp.fft.ifftn(x)


# %%
obs = Observation.load("/Users/rf/Development/data/eso_986-1137mhz.npz")
obs = obs.to_double_precision()
obs = obs.to_resolve_obs()
print(obs)

# %%
space_512 = SignalSpace.build(shape=(512, 512), fov=("1deg", "1deg"))

R, R_l, RNR, RNR_l = build_exact_responses(obs, space_512)

kernel_512 = compute_psf_kernel(RNR_l)

plot_arrays(kernel_512, norm="log", dpi=100)

# %%
import pickle

from aim_resolve import OptimizeKLConfig, get_builders, map_signal

base_yml = "/Users/rf/Development/packages/aim-resolve/steering/runs/fast_vi_512c/files/base.yml"
exp_yml = "/Users/rf/Development/packages/aim-resolve/steering/runs/fast_vi_512c/files/5_pre.yml"

optim_cfg = OptimizeKLConfig.from_file((base_yml, exp_yml), get_builders, "total")


exp_it = 3
sky_it = optim_cfg.instantiate_sec(f"sky.{exp_it}")

skies_c, means_c = [], []
with open(
    "/Users/rf/Development/packages/aim-resolve/steering/runs/fast_vi_512c/opt/3_rec/last.pkl",
    "rb",
) as f:
    smp_c, *_ = pickle.load(f)
for sky_c in sky_it.signals + sky_it.points:
    print(sky_c.prefix)
    mean = smp_c.mean(sky_c)
    skies_c += [
        sky_c,
    ]
    means_c += [
        map_signal(mean, sky_c.space, skies_c[0].space),
    ]

signal_bg, space_bg = downsample(means_c[0], 1), skies_c[0].space * 1
signal_t0, space_t0 = downsample(means_c[1], 1), skies_c[1].space * 1
signal_t1, space_t1 = downsample(means_c[2], 1), skies_c[2].space * 1
signal_t2, space_t2 = downsample(means_c[3], 1), skies_c[3].space * 1
signal_ps, mean_ps = downsample(means_c[4], 1), smp_c.mean(sky_it.points[0].points)
signal_sum = signal_bg + signal_t0 + signal_t1 + signal_t2 + signal_ps
signal_ps = signal_ps.at[0, 0].set(1e-3)

# %%
zoom = 4

dis = tuple(d * zoom for d in space_bg.distances)
grid_sum = SignalGrid(
    space=(512 // zoom, 512 // zoom), center=(0, 0), factor=zoom, distances=dis
)
grid_bg = SignalGrid(
    space=(512 // zoom, 512 // zoom), center=(0, 0), factor=1, distances=dis
)
grid_t0 = SignalGrid(
    space=(80 // zoom, 112 // zoom),
    center=(-8 // zoom, -16 // zoom),
    factor=zoom,
    distances=dis,
)
grid_t1 = SignalGrid(
    space=(40 // zoom, 40 // zoom),
    center=(-48 // zoom, 62 // zoom),
    factor=zoom,
    distances=dis,
)
grid_t2 = SignalGrid(
    space=(32 // zoom, 40 // zoom),
    center=(-48 // zoom, 96 // zoom),
    factor=zoom,
    distances=dis,
)

ps_coos = np.argwhere(signal_ps > 1)
ps_new = np.zeros_like(ps_coos).astype("float64")
for i in range(ps_coos.shape[0]):
    ps_new[i, :] = grid_sum.coos[:, ps_coos[i, 0], ps_coos[i, 1]]
grid_p0 = PointGrid(
    coordinates=tuple(map(tuple, ps_new.tolist())),
    factor=zoom,
    n_copies=ps_new.shape[0],
)

signal_dict = {
    "sum": {"val": signal_sum, "low": downsample(signal_sum, zoom), "spc": grid_sum},
    "t0": {"val": signal_t0, "low": downsample(signal_t0, zoom), "spc": grid_t0},
    "t1": {"val": signal_t1, "low": downsample(signal_t1, zoom), "spc": grid_t1},
    "t2": {"val": signal_t2, "low": downsample(signal_t2, zoom), "spc": grid_t2},
    "ps": {
        "val": signal_ps,
        "low": downsample(signal_ps, zoom),
        "spc": grid_p0,
        "pix": mean_ps,
    },
    "bg": {"val": signal_bg, "low": downsample(signal_bg, zoom), "spc": grid_bg},
    "fac": zoom,
}

kernel = kernel_512.copy()

plot_arrays(
    [sig["val"] for sig in signal_dict.values() if isinstance(sig, dict)]
    + [sig["low"] for sig in signal_dict.values() if isinstance(sig, dict)],
    norm="log",
    dpi=300,
    rows=2,
    vmin=1e3,
)

# %%
response_high = fft_convolve_2d(
    signal_dict["sum"]["val"], kernel, signal_dict["sum"]["spc"]
)
response_low = fft_convolve_2d(
    signal_dict["sum"]["low"],
    downsample(kernel, signal_dict["fac"]),
    signal_dict["bg"]["spc"],
)
response_low = upsample(response_low, signal_dict["fac"])

plot_arrays([response_high, response_low], dpi=100, rows=1)

# %%
rsp_err = []
low_res_err = np.zeros_like(signal_dict["sum"]["val"])
ker_low = downsample(kernel, zoom)
for k, sig in signal_dict.items():
    if isinstance(sig, dict):
        rsp_low = fft_convolve_2d(sig["low"], ker_low, signal_dict["bg"]["spc"])
        rsp_low = upsample(rsp_low, zoom)
        rsp_val = fft_convolve_2d(sig["val"], kernel, signal_dict["sum"]["spc"])
        rsp_err += [
            rsp_val - rsp_low,
        ]
        if k != "sum" and k != "ps":
            low_res_err += (rsp_val - rsp_low) * (
                1
                - map_array(
                    np.ones(sig["spc"].shape), sig["spc"], signal_dict["sum"]["spc"]
                )
            )

plot_arrays(
    [np.abs(err) for err in rsp_err], dpi=300, rows=1, norm="log", vmin=2e-5, vmax=1e-2
)


# %%
def multi_scale_kernel(signal_dict, kernel):
    signal_dict = deepcopy(signal_dict)
    fac = signal_dict["fac"]
    grid = signal_dict["sum"]["spc"]
    grid_low = signal_dict["bg"]["spc"]

    kernel_low = downsample(kernel, fac)
    response_low = fft_convolve_2d(signal_dict["sum"]["low"], kernel_low, grid_low)
    response_val = upsample(response_low, fac)

    for key, sig in filter(lambda x: "t" in x[0], signal_dict.items()):
        sig_grid = sig["spc"]
        sig_grid_low = sig_grid.update(factor=1)

        sig_high = map_array(sig["val"], grid, sig_grid)
        ker_high = map_array(
            kernel,
            grid.update(factor=2 * fac),
            sig_grid.update(center=(0, 0), factor=2 * fac),
        )
        rsp_high = fft_convolve_2d(sig_high, ker_high, sig_grid)

        sig_low = map_array(sig["low"], grid_low, sig_grid_low)
        ker_low = downsample(ker_high, fac)
        rsp_low = fft_convolve_2d(sig_low, ker_low, sig_grid_low)

        rsp_slc = map_array(response_low, grid_low, sig_grid_low)
        rsp_sub = rsp_slc - rsp_low
        rsp_add = upsample(rsp_sub, fac) + rsp_high
        rsp_val = map_array(rsp_add, sig_grid, grid)

        if key == "t0":
            plot_arrays(
                [rsp_slc, rsp_low, rsp_high, rsp_add],
                dpi=300,
                rows=1,
                norm="log",
                vmin=1e-5,
                vmax=1e-2,
            )

        msk_val = map_array(np.ones(sig_high.shape), sig_grid, grid)
        response_val = response_val * (1 - msk_val).clip(0, 1) + rsp_val

    for key, sig in filter(lambda x: "ps" in x[0], signal_dict.items()):
        ker_size = 8
        ps_grid = sig["spc"]
        ps_cen = np.round(ps_grid.coos).astype("int64")

        for i in range(ps_grid.n_copies):
            pi_grid = PointGrid(
                coordinates=(ps_grid.coordinates[i],), factor=ps_grid.factor, n_copies=1
            )
            sig_grid = SignalGrid(
                space=(ker_size, ker_size),
                center=tuple(ps_cen[i].tolist()),
                factor=ps_grid.factor,
                distances=grid.distances,
            )
            sig_grid_low = sig_grid.update(factor=1)

            sig_high = map_point(sig["pix"][i][None], pi_grid, sig_grid)
            ker_high = map_array(
                kernel,
                grid.update(factor=2 * fac),
                sig_grid.update(center=(0, 0), factor=2 * fac),
            )
            rsp_high = fft_convolve_2d(sig_high, ker_high, sig_grid)

            sig_low = map_point(sig["pix"][i][None], pi_grid, sig_grid_low)
            ker_low = downsample(ker_high, fac)
            rsp_low = fft_convolve_2d(sig_low, ker_low, sig_grid_low)

            rsp_slc = map_array(response_low, grid_low, sig_grid_low)
            rsp_sub = rsp_slc - rsp_low
            rsp_add = upsample(rsp_sub, fac) + rsp_high
            rsp_val = map_array(rsp_add, sig_grid, grid)

            if i == 0:
                plot_arrays(
                    [rsp_slc, rsp_low, rsp_high, rsp_add],
                    dpi=300,
                    rows=1,
                    norm="log",
                    vmin=1e-5,
                    vmax=1e-2,
                )

            msk_val = map_array(np.ones(sig_high.shape), sig_grid, grid)
            response_val = response_val * (1 - msk_val).clip(0, 1) + rsp_val

    return response_val


response_high = fft_convolve_2d(
    signal_dict["sum"]["val"], kernel, signal_dict["sum"]["spc"]
)
response_low = fft_convolve_2d(
    signal_dict["sum"]["low"],
    downsample(kernel, signal_dict["fac"]),
    signal_dict["bg"]["spc"],
)
response_low = upsample(response_low, signal_dict["fac"])


response_ms = multi_scale_kernel(signal_dict, kernel)

plot_arrays(
    [response_high, response_low, response_ms], dpi=300, rows=1, vmin=1e-5, vmax=1e-1
)
plot_arrays(
    [response_high, response_low, response_ms],
    dpi=300,
    rows=1,
    norm="log",
    vmin=1e-5,
    vmax=1e-1,
)

response_errors = [
    low_res_err,
    response_high - response_low,
    response_high - response_ms,
]

plot_arrays(response_errors, dpi=300, rows=1, vmin=1e-5, vmax=1e-2)
plot_arrays(
    [np.abs(err) for err in response_errors],
    dpi=300,
    rows=1,
    norm="log",
    vmin=1e-5,
    vmax=1e-2,
)
plot_arrays(
    [np.abs(err / response_high) for err in response_errors],
    dpi=300,
    rows=1,
    norm="log",
    vmin=1e-4,
    vmax=1e4,
)

# %%
import matplotlib.pyplot as plt


def costs(N):
    M = 2 * N
    return 2 * M * np.log(M) + M**2


scm = costs(1024 * 4)

mcm = (
    costs(1024)
    + (costs(256 * 4) + costs(256 * 2)) * 2
    + (costs(64 * 4) + costs(64 * 2)) * 60
)
print(scm, mcm, scm / mcm)


n = (2, 13)
x = (np.ones(n[1] - n[0]) * 2) ** np.arange(n[0], n[1])
plt.plot(x, costs(x))
plt.show()
