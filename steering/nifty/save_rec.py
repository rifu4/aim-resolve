"""CLI script for exporting a reconstruction (its posterior samples) to FITS files.

For the sky section ``sky.<it>`` of a run, every posterior sample and the posterior
mean are written as multi-extension FITS files in the format of the fast-resolve
products (``eso136-006_1053MHz_posterior_mean.fits``):

- ``<prefix>_<nu>MHz_posterior_<tag>.fits``: sky brightness [mJy/arcsec^2] at every
  model frequency, the total sky in the primary HDU
- ``<prefix>_spectral_index_posterior_<tag>.fits``: spectral index at the reference
  frequency (multi-frequency models)
- ``<prefix>_spectral_curvature_posterior_<tag>.fits``: spectral curvature
  (multi-frequency models with a curvature term)

with ``<tag>`` = ``sample_<k>`` for every sample and ``mean`` for their mean. Each file
holds one layer per sky component:

- BG: background, full sky grid
- O0..On: objects, each on its own (native) grid as a cutout with its own WCS (CRPIX
  shifted by the cutout offset on the sky grid)
- TILES: tiles on the full sky grid, NaN outside the tiles
- TILECUBE: the individual tiles as a cube (tile, x, y), since overlapping tiles cannot
  be separated in TILES
- TILEPOS: table of the tile positions on the sky grid (row = cube index)
- POINTS: table of the point sources. Each source covers the sub-pixels of the point
  grid within one coarse pixel; positions are the brightness-weighted centroids.

Single-frequency models give one brightness file per tag at the frequency of the data
(their model frequency is a placeholder): the weighted mean frequency of the data
channels, or ``freq`` of the config.

Header and axis conventions follow ``resolve.ubik_tools.fits.field2fits``: 4D data
(STOKES, FREQ, DEC, RA), array axis 0 -> RA (NAXIS1, CDELT1 < 0), axis 1 -> DEC. The
phase centre is at the 0-based pixel N / 2, i.e. CRPIX = N / 2 + 1.

Primary beam: without ``pb_correction`` in the likelihood the reconstruction is the
apparent (beam-attenuated) sky. With ``pb_correction: true`` in the ``fits`` section,
every sample is then divided by the MeerKAT Stokes I beam before writing (brightness
layers, tiles, cutouts and point fluxes): the beam at each model frequency, or for a
single-frequency model the beam averaged over the data channels (weighted by the channel
weights, as the likelihood does). The spectral index is corrected by the beam's spectral
slope at the reference frequency, the curvature by the beam's curvature (its q(u)
coefficient, as the model's beta). Below ``pb_min`` the correction is unreliable:
``pb_mode: clip`` divides by max(PB, pb_min) there, ``pb_mode: nan`` sets those pixels
to NaN. The files are then written to ``fits_pbcor/``. With the beam in the forward
model the samples are already the intrinsic sky and no correction is applied.

Settings: the ``fits`` section of the YAML file given with ``--config`` (see
``FITS_DEFAULTS``, an example is ``config/rec_products.yml``).
"""

import os

import click

FITS_DEFAULTS = dict(
    rec_dir=None,  # run directory with opt.yml and the samples (required)
    it=None,  # iteration of the sky section to export, sky.<it> (required)
    samples="last.pkl",  # pickled samples in rec_dir
    prefix="sky",  # file name prefix
    ref_idx=None,  # reference-frequency index; None: ref_freq_index of the sky's background, else nfreq // 2
    freq=None,  # single-frequency models: frequency of the files [Hz]; None: weighted mean of the data channels
    data=None,  # data section (phase centre, channel weights); None: the data of the likelihood of sky.<it>
    pb_correction=True,  # divide by the primary beam (if it is not in the forward model)
    pb_min=0.1,  # primary-beam level below which the correction is clipped / masked
    pb_mode="clip",  # "clip": divide by max(PB, pb_min), "nan": NaN where PB < pb_min
    pb_model=None,  # MeerKAT JimBeam model; None: chosen from the frequencies
    n_band=32,  # single-frequency models: number of data channels for the band-averaged beam
    odir=None,  # output directory; None: <rec_dir>/fits, <rec_dir>/fits_pbcor with the beam correction
)

ARCMIN2RAD = 3.141592653589793 / 60 / 180
AS2RAD = ARCMIN2RAD / 60
CONV_FACTOR = 1000 * AS2RAD**2  # Jy/sr -> mJy/arcsec^2


def load_section(config, section, defaults):
    """Section of a YAML file merged into its defaults; unknown keys raise."""
    import aim_resolve as aim

    dct = aim.yaml_load(config).get(section) or {}
    unknown = set(dct) - set(defaults)
    if unknown:
        raise ValueError(f"unknown keys in the {section} section: {sorted(unknown)}")
    return {**defaults, **dct}


@click.command()
@click.option("--config", required=True, help="Path to the YAML file with the `fits` section")
@click.option(
    "--cuda_device",
    default="",
    help='CUDA device to use (e.g. "0"), default is "" for CPU',
)
def main(config, cuda_device):
    """Export the posterior samples of a reconstruction to FITS files.

    Parameters
    ----------
    config : str
        Path to the YAML file with the ``fits`` section.
    cuda_device : str
        CUDA device identifier, empty string for CPU.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if not cuda_device:
        os.environ["JAX_PLATFORMS"] = "cpu"

    fits_cfg = load_section(config, "fits", FITS_DEFAULTS)
    for key in ("rec_dir", "it"):
        if fits_cfg[key] is None:
            raise ValueError(f"`{key}` is required in the fits section")
    export(fits_cfg)


def ref_freq_index(cfg, sky_sec, nfreq):
    """``ref_freq_index`` of the background's parameters in the run config (as the
    model builds it), else nfreq // 2 (the model's default)."""
    sections = cfg.sections
    bg = sections[sky_sec].get("background", "")
    bg_sec = sections.get(bg[1:], {}) if isinstance(bg, str) and bg.startswith("=") else {}
    params = bg_sec.get("params", {}) if isinstance(bg_sec, dict) else {}
    while isinstance(params, dict):
        if "ref_freq_index" in params:
            return int(params["ref_freq_index"])
        base = params.get("base")
        params = sections.get(base) if isinstance(base, str) else None
    return nfreq // 2


def export(fits_cfg):
    """Write the FITS files of one sky section (see the module docstring)."""
    import pickle
    import time

    import astropy.io.fits as pyfits
    import numpy as np
    from astropy.time import Time
    from astropy.wcs import WCS
    from resolve.library.meerkat_beam import JimBeam

    import aim_resolve as aim
    from aim_resolve.fast_resolve.beam import default_beam_model
    from aim_resolve.model.map import point_slice_indices, signal_slice_indices

    rec_dir, it, prefix = fits_cfg["rec_dir"], int(fits_cfg["it"]), fits_cfg["prefix"]
    pb_correction, pb_min, pb_mode = bool(fits_cfg["pb_correction"]), float(fits_cfg["pb_min"]), fits_cfg["pb_mode"]
    if pb_mode not in ("clip", "nan"):
        raise ValueError(f"pb_mode must be 'clip' or 'nan', not {pb_mode!r}")

    opt_yml = os.path.join(rec_dir, "opt.yml")
    print("load:", opt_yml)
    optim_cfg = aim.OptimizeKLConfig.from_file(opt_yml, aim.get_builders)
    sky_sec = f"sky.{it}"
    sky = optim_cfg.instantiate_sec(sky_sec)
    print("sky components:", [c.prefix for c in sky.models])

    # Likelihood of the sky section: with `pb_correction` the beam is part of the forward model,
    # so the samples are already the intrinsic sky (no post-hoc correction)
    lh_secs = [
        dct for sec, dct in optim_cfg.sections.items()
        if sec.startswith("lh.") and isinstance(dct, dict) and dct.get("sky") == f"={sky_sec}"
    ] or [optim_cfg.sections.get(f"lh.{it}", {})]
    pb_in_model = any(dct.get("pb_correction") for dct in lh_secs)
    if pb_in_model and pb_correction:
        print("primary beam is in the forward model (pb_correction in lh): no primary-beam correction")
        pb_correction = False
    data_sec = fits_cfg["data"] or next(
        (dct["data"][1:] for dct in lh_secs if isinstance(dct.get("data"), str) and dct["data"].startswith("=")),
        f"data.{it}",
    )
    data_fn = optim_cfg.sections[data_sec]["fname"]
    odir = fits_cfg["odir"] or os.path.join(rec_dir, "fits" + ("_pbcor" if pb_correction else ""))

    # Frequencies: the model frequencies, or for a single-frequency model the frequency of the data
    multi = np.asarray(sky.freq).size > 1
    with np.load(data_fn, allow_pickle=True) as data:
        field_keys = list(data["auxtable_FIELD_0000"])
        phase_dir = data[f"auxtable_FIELD_{field_keys.index('PHASE_DIR') + 1:04d}"]
        delay_dir = data[f"auxtable_FIELD_{field_keys.index('DELAY_DIR') + 1:04d}"]
        if not multi:
            ch_freq = np.asarray(data["freq"], dtype="float64")
            weight = np.asarray(data["weight"])
            ch_wgt = weight.reshape(-1, weight.shape[-1]).sum(axis=0, dtype="float64")
            del weight
    if multi:
        freq_hz = np.asarray(sky.freq, dtype="float64")
        ref_idx = fits_cfg["ref_idx"] if fits_cfg["ref_idx"] is not None else ref_freq_index(optim_cfg, sky_sec, freq_hz.size)
    else:
        if np.all(ch_wgt <= 0):
            raise ValueError(f"the data {data_fn} have no channel with positive weight")
        nu = fits_cfg["freq"] if fits_cfg["freq"] is not None else np.sum(ch_wgt * ch_freq) / np.sum(ch_wgt)
        freq_hz, ref_idx = np.array([float(nu)]), 0
    nfreq = freq_hz.size
    print("sky freqs:", [f"{round(f / 1e6)} MHz" for f in freq_hz],
          f"| ref freq: {round(freq_hz[ref_idx] / 1e6)} MHz" if multi else "(single frequency: the data's)")

    with open(os.path.join(rec_dir, fits_cfg["samples"]), "rb") as f:
        samples, *_ = pickle.load(f)
    print("samples:", len(samples))

    # Phase centre (RA, DEC) [rad] from the FIELD table of the data
    if pb_correction and not np.allclose(phase_dir, delay_dir):
        raise ValueError("pointing centre != phase centre, the primary beam would need an offset")
    ra0, dec0 = np.asarray(phase_dir, dtype="float64").reshape(-1)[:2]
    print(f"phase centre: RA {np.rad2deg(ra0):.6f} deg, DEC {np.rad2deg(dec0):.6f} deg")

    # Pixel size [rad] of the full sky grid
    grid = sky.grid
    pix_rad = np.asarray(grid.dis, dtype="float64") / grid.fac
    nx, ny = grid.shape
    print(f"sky grid: {grid.shape} | pixel size: {np.rad2deg(pix_rad[0]) * 3600:.3f} arcsec")

    def with_freq(a, axis=0):
        """Array with a frequency axis at ``axis`` (single-frequency models have none)."""
        a = np.asarray(a)
        return a if multi else np.expand_dims(a, axis)

    # -----------------------------------------------------------------------------------------
    # Primary beam (single pointing at the phase centre, 0-based pixel N / 2). JimBeam.I takes
    # offsets in deg and the frequency in MHz. Single-frequency models: the beam averaged over
    # the data channels, weighted by the channel weights (as build_primary_beam).
    # -----------------------------------------------------------------------------------------
    if not multi:
        idx = np.nonzero(ch_wgt > 0)[0]
        idx = idx[np.unique(np.linspace(0, idx.size - 1, min(int(fits_cfg["n_band"]), idx.size)).round().astype(int))]
        band = list(zip(ch_wgt[idx] / ch_wgt[idx].sum(), ch_freq[idx], strict=True))
    pb_model = fits_cfg["pb_model"] or default_beam_model(freq_hz if multi else ch_freq)
    mk_beam = JimBeam(pb_model)
    pix_deg = np.rad2deg(pix_rad)

    def primary_beam(px, py, freq):
        """Stokes I beam at 0-based sky-grid pixel coordinates (px, py) [broadcast]."""
        dx, dy = (px - nx / 2) * pix_deg[0], (py - ny / 2) * pix_deg[1]
        if not multi:
            return sum(w * mk_beam.I(dx, dy, f / 1e6) for w, f in band)
        return mk_beam.I(dx, dy, freq / 1e6)

    def beam_slope(px, py, freq, eps=1e-3):
        """Spectral slope of the beam, d ln PB / d ln nu, at `freq`."""
        lo, hi = primary_beam(px, py, freq * (1 - eps)), primary_beam(px, py, freq * (1 + eps))
        return (np.log(hi) - np.log(lo)) / (np.log1p(eps) - np.log1p(-eps))

    def pixel_grid(shape, off=(0, 0)):
        """Sky-grid pixel coordinates of an image of `shape` starting at pixel `off`."""
        return np.meshgrid(off[0] + np.arange(shape[0]), off[1] + np.arange(shape[1]), indexing="ij")

    def beam_maps(px, py):
        """Beam to divide by at every frequency (nfreq, *px.shape) and its spectral slope at
        nu_ref (*px.shape; zero for single-frequency models). Below pb_min the beam is clipped
        to pb_min (slope 0) or NaN, depending on pb_mode."""
        pb = np.stack([primary_beam(px, py, f) for f in freq_hz])
        fill = pb_min if pb_mode == "clip" else np.nan
        if multi:
            slope = beam_slope(px, py, freq_hz[ref_idx])
            slope = np.where(pb[ref_idx] >= pb_min, slope, 0.0 if pb_mode == "clip" else np.nan)
        else:
            slope = np.zeros(np.shape(px))
        return np.where(pb >= pb_min, pb, fill), slope

    # curvature basis of the spectral model, q(u) = u^2 - (sum u^3 / sum u^2) u with
    # u = ln(nu / nu_ref) (SpectralIndexCurvature / MultiFrequencyModel)
    u_freq = np.log(freq_hz / freq_hz[ref_idx])
    q_freq = u_freq**2 - np.sum(u_freq**3) / np.sum(u_freq**2) * u_freq if multi else None

    def beam_curvature(px, py):
        """Spectral curvature of the beam, the q(u) coefficient of ln PB (as the model's beta),
        at sky-grid pixels. Not corrected where PB(nu_ref) < pb_min, as the slope."""
        ln_pb = np.log(np.stack([primary_beam(px, py, f) for f in freq_hz]))
        curv = np.tensordot(q_freq, ln_pb - ln_pb[ref_idx], axes=1) / np.sum(q_freq**2)
        ok = primary_beam(px, py, freq_hz[ref_idx]) >= pb_min
        return np.where(ok, curv, 0.0 if pb_mode == "clip" else np.nan)

    if pb_correction:
        pb_sky, slope_sky = beam_maps(*pixel_grid(grid.shape))
        print(f"primary beam {pb_model} ({pb_mode}{'' if multi else ', band average'}): "
              f"fraction of sky pixels with PB < {pb_min}:", [f"{np.mean(~(p > pb_min)):.3f}" for p in pb_sky])

    # -----------------------------------------------------------------------------------------
    # FITS helpers.
    # -----------------------------------------------------------------------------------------
    def fits_header(freq, bunit, btype=None, offset=(0, 0)):
        """Header of a single-frequency, single-Stokes sky image; `offset` is the 0-based start
        pixel of a cutout on the full sky grid (shifts CRPIX)."""
        h = pyfits.Header()
        h["ORIGIN"] = "aim-resolve"
        h["BUNIT"] = bunit
        if btype is not None:
            h["BTYPE"] = btype
        h["OBSRA"] = np.rad2deg(ra0)
        h["OBSDEC"] = np.rad2deg(dec0)
        h["CTYPE1"] = "RA---SIN"
        h["CRVAL1"] = np.rad2deg(ra0)
        h["CDELT1"] = -np.rad2deg(pix_rad[0])
        h["CRPIX1"] = nx / 2 + 1 - float(offset[0])
        h["CUNIT1"] = "deg"
        h["CTYPE2"] = "DEC--SIN"
        h["CRVAL2"] = np.rad2deg(dec0)
        h["CDELT2"] = np.rad2deg(pix_rad[1])
        h["CRPIX2"] = ny / 2 + 1 - float(offset[1])
        h["CUNIT2"] = "deg"
        h["CTYPE3"] = "FREQ"
        h["CRVAL3"] = float(freq)
        h["CDELT3"] = 1.0
        h["CRPIX3"] = 1.0
        h["CUNIT3"] = "Hz"
        h["CTYPE4"] = "STOKES"
        h["CRVAL4"] = 1.0
        h["CDELT4"] = 1.0
        h["CRPIX4"] = 1.0
        h["CUNIT4"] = ""
        h["DATE-MAP"] = Time(time.time(), format="unix").iso.split()[0]
        h["EQUINOX"] = 2000.0
        if not multi:
            h["FREQBAND"] = (f"{ch_freq.min() / 1e6:.2f}-{ch_freq.max() / 1e6:.2f}",
                             "[MHz] data band of the single-frequency model")
        if pb_correction:
            h["PBCOR"] = (True, "primary-beam corrected")
            h["PBMODEL"] = pb_model
            h["PBMIN"] = (pb_min, "PB clipped to / NaN below this level (PBMODE)")
            h["PBMODE"] = (pb_mode, "clip: divided by max(PB, PBMIN); nan: NaN below")
            if not multi:
                h["PBBAND"] = (True, "beam averaged over the data channels")
        if pb_in_model:
            h["PBINMOD"] = (True, "primary beam in the forward model: intrinsic sky")
        return h

    def image_hdu(array, header, name):
        """Image extension of a 2D (RA, DEC) array, stored as (STOKES, FREQ, DEC, RA) float32."""
        h = header.copy()
        h["EXTNAME"] = name
        return pyfits.ImageHDU(np.asarray(array, dtype="float32").T[None, None], header=h)

    def ext_name(kind, i, n):
        """Extension name: `kind` for a single model of that kind, else `kind<i>`."""
        return kind if n == 1 else f"{kind}{i}"

    def write_hdus(hdus, name):
        os.makedirs(odir, exist_ok=True)
        fn = os.path.join(odir, name)
        pyfits.HDUList(hdus).writeto(fn, overwrite=True)
        print(f"saved {fn}")

    # -----------------------------------------------------------------------------------------
    # Component geometry and the beam on every component.
    # -----------------------------------------------------------------------------------------
    dvol = float(grid.dvol)  # pixel solid angle [sr]
    ref_freq = freq_hz[ref_idx]
    wcs_sky = WCS(fits_header(ref_freq, "")).celestial
    obj_offsets = [np.asarray(signal_slice_indices(o.grid, grid)[1]) for o in sky.objects]
    for i, (o, off) in enumerate(zip(sky.objects, obj_offsets, strict=True)):
        print(f"  O{i} ({o.prefix}): cutout {o.grid.shape} at sky pixel {tuple(off.tolist())}")
    tile_offsets = []
    for t in sky.tiles:
        t_in, t_out = (np.asarray(v).reshape(-1, 2) for v in signal_slice_indices(t.tiles.grid, grid))
        if np.any(t_in != 0):
            raise ValueError("tiles reaching beyond the sky grid are not supported")
        tile_offsets.append(t_out)  # (n_tiles, 2) 0-based start pixel on the sky grid
    if pb_correction:
        pb_obj = [beam_maps(*pixel_grid(o.grid.shape, off)) for o, off in zip(sky.objects, obj_offsets, strict=True)]
        pb_tiles = []  # per tile model: PB (n_tiles, nfreq, tx, ty), slope (n_tiles, tx, ty)
        for t, off in zip(sky.tiles, tile_offsets, strict=True):
            tb = [beam_maps(*pixel_grid(t.tiles.grid.shape, o)) for o in off]
            pb_tiles.append((np.stack([b[0] for b in tb]), np.stack([b[1] for b in tb])))

    has_curvature = multi and any(m.has_spectral_curvature for m in sky.models)
    if multi:
        print("components with a spectral curvature:", [m.prefix for m in sky.models if m.has_spectral_curvature])
    if pb_correction and has_curvature:
        curv_sky = beam_curvature(*pixel_grid(grid.shape))
        curv_obj = [beam_curvature(*pixel_grid(o.grid.shape, off)) for o, off in zip(sky.objects, obj_offsets, strict=True)]
        curv_tiles = [np.stack([beam_curvature(*pixel_grid(t.tiles.grid.shape, o)) for o in off])
                      for t, off in zip(sky.tiles, tile_offsets, strict=True)]

    def tilecube_hdu(cube, i, header_cards, comment):
        """Cube extension of the individual tiles (n_tiles, tx, ty), stored as (tile, DEC, RA)
        like the sky images; the tile index is the TILEPOS row."""
        h = pyfits.Header()
        h["EXTNAME"] = ext_name("TILECUBE", i, len(sky.tiles))
        for key, val in header_cards.items():
            h[key] = val
        h["COMMENT"] = "Individual tiles: axis 3 = tile index (row of TILEPOS), axes 1 / 2 ="
        h["COMMENT"] = "tile pixels along RA / DEC (same orientation as the sky images)."
        h["COMMENT"] = comment
        return pyfits.ImageHDU(np.asarray(cube, dtype="float32").transpose(0, 2, 1), header=h)

    def tilepos_hdu(i):
        """TILEPOS table: start pixel, size and centre of every tile on the sky grid."""
        off = tile_offsets[i]
        tx, ty = sky.tiles[i].tiles.grid.shape
        ra, dec = wcs_sky.pixel_to_world_values(off[:, 0] + (tx - 1) / 2, off[:, 1] + (ty - 1) / 2)
        return pyfits.BinTableHDU.from_columns([
            pyfits.Column("TILE", "J", array=np.arange(len(off))),
            pyfits.Column("X0_PIX", "J", unit="pix", array=off[:, 0] + 1),  # FITS (1-based) sky pixel of tile pixel (1, 1)
            pyfits.Column("Y0_PIX", "J", unit="pix", array=off[:, 1] + 1),
            pyfits.Column("NX", "J", unit="pix", array=np.full(len(off), tx)),
            pyfits.Column("NY", "J", unit="pix", array=np.full(len(off), ty)),
            pyfits.Column("RA", "D", unit="deg", array=ra),  # tile centre
            pyfits.Column("DEC", "D", unit="deg", array=dec),
        ], name=ext_name("TILEPOS", i, len(sky.tiles)))

    def tile_cube(t, x, prop=None):
        """Individual tiles of a tile model, (n_tiles, nfreq, tx, ty), or of its spectral
        property `prop` (n_tiles, tx, ty)."""
        cube = np.asarray((getattr(t, prop) if prop else t)(x, map=False))
        if t.tiles.grid.n_copies == 1:
            cube = cube[None]
        return cube if prop else with_freq(cube, 1)

    def point_sources(m):
        """Sky-grid pixels (n_sub, 2) of the point-grid sub-pixels and their source index
        (sources = sub-pixels grouped by their coarse pixel)."""
        pix = np.asarray(point_slice_indices(m.points.grid, grid)[1], dtype="float64")  # (n_sub, 2)
        coarse = np.floor(np.asarray(m.points.grid.coos)).astype(int)
        _, src = np.unique(coarse, axis=0, return_inverse=True)
        return pix, src.reshape(-1)

    def point_catalogue(m, x):
        """Per point source: flux density [mJy] at every frequency (n_src, nfreq), spectral
        index at nu_ref (NaN for single-frequency models) and centroid pixel (0-based, sky grid)."""
        pix, src = point_sources(m)
        n_src, n_sub = src.max() + 1, src.size

        def per_source(v):
            return np.bincount(src, v, n_src)

        s_freq = np.asarray(m(x, map=False)).reshape(n_sub, nfreq)  # [Jy/sr] per sub-pixel
        s_ref = np.asarray(m.ref_freq_model(x, map=False)).reshape(-1) if multi else s_freq[:, 0]
        alpha = np.asarray(m.spectral_index(x, map=False)).reshape(-1) if multi else np.zeros(n_sub)
        w = per_source(s_ref)
        px = np.stack([per_source(s_ref * pix[:, i]) / w for i in range(2)], axis=1)  # apparent-brightness centroid
        if pb_correction:
            pb, slope = beam_maps(pix[:, 0], pix[:, 1])  # (nfreq, n_sub), (n_sub,)
            s_freq = s_freq / pb.T
            s_ref = s_ref / pb[ref_idx]
            alpha = alpha - slope
            w = per_source(s_ref)
        flux = np.stack([per_source(s_freq[:, fi]) for fi in range(nfreq)], axis=1) * dvol * 1e3
        alpha_src = per_source(s_ref * alpha) / w if multi else np.full(n_src, np.nan)
        return flux, alpha_src, px

    def point_curvature(m, x):
        """Per point source: spectral curvature, nu_ref-brightness-weighted over the sub-pixels
        as the spectral index (NaN without a curvature term)."""
        pix, src = point_sources(m)
        n_src = src.max() + 1
        if not m.has_spectral_curvature:
            return np.full(n_src, np.nan)
        s_ref = np.asarray(m.ref_freq_model(x, map=False)).reshape(-1)
        beta = np.asarray(m.spectral_curvature(x, map=False)).reshape(-1)
        if pb_correction:
            s_ref = s_ref / beam_maps(pix[:, 0], pix[:, 1])[0][ref_idx]
            beta = beta - beam_curvature(pix[:, 0], pix[:, 1])
        return np.bincount(src, s_ref * beta, n_src) / np.bincount(src, s_ref, n_src)

    def tile_spectral_index(m, x, prop="spectral_index"):
        """Spectral index (or another spectral property, e.g. `spectral_curvature`) of the
        tiles on the sky grid, NaN outside. Where tiles overlap it is brightness-weighted,
        sum_i S_i alpha_i / sum_i S_i; elsewhere (or if all weights vanish, e.g. far in the
        tapered tile wings) the plain tile index, so that the weights cannot underflow to 0 / 0."""
        ref, alpha = m.ref_freq_model, getattr(m, prop)
        s_ref, a_tile = ref(x, map=False), alpha(x, map=False)
        num = np.squeeze(np.asarray(ref.map_function(s_ref * a_tile)))
        den = np.squeeze(np.asarray(ref.map_function(s_ref)))
        a_sum = np.squeeze(np.asarray(alpha.map_function(a_tile)))
        n_cov = np.squeeze(np.asarray(alpha.map_function(np.ones_like(a_tile))))
        plain = np.where(n_cov > 0, a_sum / np.where(n_cov > 0, n_cov, 1.0), np.nan)
        weighted = num / np.where(den > 0, den, 1.0)
        return np.where((n_cov > 1) & (den > 0), weighted, plain)

    # -----------------------------------------------------------------------------------------
    # Evaluate every posterior sample once into all component layers; the posterior mean is
    # the average of these layers over the samples. Brightness layers are in mJy/arcsec^2,
    # point-source flux densities in mJy.
    # -----------------------------------------------------------------------------------------
    tile_masks = [np.asarray(t.mask) for t in sky.tiles]

    def sample_layers(x):
        """All component layers of one posterior sample `x` (float32)."""
        lay = dict(
            sky=with_freq(sky(x)) * CONV_FACTOR,  # (nfreq, nx, ny)
            bg=with_freq(sky.background(x)) * CONV_FACTOR,  # (nfreq, nx, ny)
            obj=[with_freq(o(x)) * CONV_FACTOR for o in sky.objects],  # (nfreq, *cutout)
            tiles=[np.where(mk, with_freq(t(x)) * CONV_FACTOR, np.nan) for t, mk in zip(sky.tiles, tile_masks, strict=True)],
            tile_cube=[tile_cube(t, x) * CONV_FACTOR for t in sky.tiles],  # (n_tiles, nfreq, tx, ty)
            points=[point_catalogue(p, x) for p in sky.points],  # (flux, alpha, px)
        )
        if multi:
            lay.update(
                bg_alpha=np.asarray(sky.background.spectral_index(x)),
                obj_alpha=[np.asarray(o.spectral_index(x)) for o in sky.objects],
                tiles_alpha=[tile_spectral_index(t, x) for t in sky.tiles],
                tile_cube_alpha=[tile_cube(t, x, "spectral_index") for t in sky.tiles],
            )
        if has_curvature:
            lay.update(curvature_layers(x, lay))
        if pb_correction:
            lay = pb_correct(lay)
        return as_float32(lay)

    def curvature_layers(x, lay):
        """Spectral curvature layers, shaped as the spectral index layers; NaN for the
        components without a curvature term."""
        def curv(m, alpha_like, fun):
            return fun() if m.has_spectral_curvature else np.full(np.shape(alpha_like), np.nan)

        bg = sky.background
        return dict(
            bg_beta=curv(bg, lay["bg_alpha"], lambda: np.asarray(bg.spectral_curvature(x))),
            obj_beta=[curv(o, a, lambda o=o: np.asarray(o.spectral_curvature(x)))
                      for o, a in zip(sky.objects, lay["obj_alpha"], strict=True)],
            tiles_beta=[curv(t, a, lambda t=t: tile_spectral_index(t, x, "spectral_curvature"))
                        for t, a in zip(sky.tiles, lay["tiles_alpha"], strict=True)],
            tile_cube_beta=[curv(t, a, lambda t=t: tile_cube(t, x, "spectral_curvature"))
                            for t, a in zip(sky.tiles, lay["tile_cube_alpha"], strict=True)],
            points_beta=[point_curvature(p, x) for p in sky.points],  # (n_src,), PB-corrected there
        )

    def pb_correct(lay):
        """Divide the brightness layers by the primary beam and subtract its spectral slope
        from the spectral index layers (point sources are corrected in `point_catalogue`). In
        tile overlaps the brightness-weighted index is
        sum_i (S_i / PB) (alpha_i - slope) / sum_i (S_i / PB) = alpha - slope as well."""
        out = dict(
            lay,
            sky=lay["sky"] / pb_sky,
            bg=lay["bg"] / pb_sky,
            obj=[o / pb for o, (pb, _) in zip(lay["obj"], pb_obj, strict=True)],
            tiles=[t / pb_sky for t in lay["tiles"]],
            tile_cube=[c / pb for c, (pb, _) in zip(lay["tile_cube"], pb_tiles, strict=True)],
        )
        if multi:
            out.update(
                bg_alpha=lay["bg_alpha"] - slope_sky,
                obj_alpha=[a - s for a, (_, s) in zip(lay["obj_alpha"], pb_obj, strict=True)],
                tiles_alpha=[a - slope_sky for a in lay["tiles_alpha"]],
                tile_cube_alpha=[a - s for a, (_, s) in zip(lay["tile_cube_alpha"], pb_tiles, strict=True)],
            )
        if has_curvature:
            out.update(
                bg_beta=lay["bg_beta"] - curv_sky,
                obj_beta=[b - c for b, c in zip(lay["obj_beta"], curv_obj, strict=True)],
                tiles_beta=[b - curv_sky for b in lay["tiles_beta"]],
                tile_cube_beta=[b - c for b, c in zip(lay["tile_cube_beta"], curv_tiles, strict=True)],
            )
        return out

    print("evaluating the posterior samples ...")
    layers = [sample_layers(s) for s in samples]
    mean_layers = average(layers)

    # Point-source positions: posterior-mean centroids, identical in every file, so that the
    # POINTS rows match across frequencies and samples.
    points = []
    for p, (_, _, px) in zip(sky.points, mean_layers["points"], strict=True):
        px = np.asarray(px, dtype="float64")
        ra, dec = wcs_sky.pixel_to_world_values(px[:, 0], px[:, 1])
        points.append(dict(px=px, ra=ra, dec=dec))
        print(f"  {p.prefix}: {len(px)} point sources")

    def point_table(i, extra_columns, header_cards):
        """POINTS table: name and position columns, followed by `extra_columns`."""
        pt = points[i]
        table = pyfits.BinTableHDU.from_columns([
            pyfits.Column("NAME", "A8", array=[f"P{i}_{k:03d}" for k in range(len(pt["ra"]))]),
            pyfits.Column("RA", "D", unit="deg", array=pt["ra"]),
            pyfits.Column("DEC", "D", unit="deg", array=pt["dec"]),
            pyfits.Column("X_PIX", "D", unit="pix", array=pt["px"][:, 0] + 1),  # FITS (1-based) pixel of the sky images
            pyfits.Column("Y_PIX", "D", unit="pix", array=pt["px"][:, 1] + 1),
            *extra_columns,
        ], name=ext_name("POINTS", i, len(points)))
        for key, val in header_cards.items():
            table.header[key] = val
        return table

    def spectral_header(btype, label, what):
        """Image header of a spectral file and its primary header (global keywords only)."""
        hdr = fits_header(ref_freq, "", btype=btype)
        hdr0 = pyfits.Header()
        for key in ("ORIGIN", "OBSRA", "OBSDEC", "DATE-MAP", "EQUINOX", "PBCOR", "PBMODEL", "PBMIN", "PBMODE", "PBINMOD"):
            if key in hdr:
                hdr0[key] = hdr[key]
        hdr0["BTYPE"] = btype
        hdr0["REFFREQ"] = (ref_freq, f"[Hz] reference frequency of the {what}")
        hdr0["PRODUCT"] = label
        return hdr, hdr0

    # -----------------------------------------------------------------------------------------
    # Sky brightness [mJy/arcsec^2], one multi-extension file per frequency: the total sky
    # (sum of all components) in the primary HDU, then the component layers; the POINTS table
    # gives the flux density [mJy] per source.
    # -----------------------------------------------------------------------------------------
    def write_brightness(lay, tag, label):
        for fi, f in enumerate(freq_hz):
            hdr = fits_header(f, "mJy/arcsec^2")
            hdr0 = hdr.copy()
            hdr0["PRODUCT"] = label
            hdr0["COMMENT"] = f"Total sky brightness (sum of all components, {label})."
            hdr0["COMMENT"] = "Extensions: BG, O0..On (cutouts), TILES, TILECUBE; tables: TILEPOS, POINTS."
            hdus = [pyfits.PrimaryHDU(np.asarray(lay["sky"][fi], dtype="float32").T[None, None], header=hdr0)]
            hdus.append(image_hdu(lay["bg"][fi], hdr, "BG"))
            for i, (o, off) in enumerate(zip(lay["obj"], obj_offsets, strict=True)):
                hdus.append(image_hdu(o[fi], fits_header(f, "mJy/arcsec^2", offset=off), f"O{i}"))
            for i, (t, cube) in enumerate(zip(lay["tiles"], lay["tile_cube"], strict=True)):
                hdus.append(image_hdu(t[fi], hdr, ext_name("TILES", i, len(lay["tiles"]))))
                hdus.append(tilecube_hdu(
                    cube[:, fi], i, {"BUNIT": "mJy/arcsec^2", "FREQ": (float(f), "[Hz]")},
                    "Overlapping tiles add up: TILES = sum of the tiles placed via TILEPOS.",
                ))
                hdus.append(tilepos_hdu(i))
            for i, (flux, _, _) in enumerate(lay["points"]):
                hdus.append(point_table(
                    i, [pyfits.Column("FLUX", "E", unit="mJy", array=flux[:, fi])],
                    {"FREQ": (float(f), "[Hz] frequency of FLUX")},
                ))
            write_hdus(hdus, f"{prefix}_{round(f / 1e6)}MHz_posterior_{tag}.fits")

    # -----------------------------------------------------------------------------------------
    # Spectral index at nu_ref per component (multi-frequency models): the primary HDU holds
    # the global keywords only, then the component layers. Overlapping tiles are combined
    # brightness-weighted, sum_i S_i alpha_i / sum_i S_i; the POINTS table gives the flux
    # density at nu_ref and the spectral index.
    # -----------------------------------------------------------------------------------------
    def write_spectral_index(lay, tag, label):
        hdr, hdr0 = spectral_header("SPECTRAL INDEX", label, "spectral index")
        hdr0["COMMENT"] = f"Spectral index at REFFREQ per sky component ({label})."
        hdr0["COMMENT"] = "Extensions: BG, O0..On (cutouts), TILES, TILECUBE; tables: TILEPOS, POINTS."
        hdus = [pyfits.PrimaryHDU(header=hdr0)]
        hdus.append(image_hdu(lay["bg_alpha"], hdr, "BG"))
        for i, (alpha, off) in enumerate(zip(lay["obj_alpha"], obj_offsets, strict=True)):
            hdus.append(image_hdu(alpha, fits_header(ref_freq, "", btype="SPECTRAL INDEX", offset=off), f"O{i}"))
        for i, (alpha, cube) in enumerate(zip(lay["tiles_alpha"], lay["tile_cube_alpha"], strict=True)):
            hdus.append(image_hdu(alpha, hdr, ext_name("TILES", i, len(lay["tiles_alpha"]))))
            hdus.append(tilecube_hdu(
                cube, i, {"BUNIT": "", "BTYPE": "SPECTRAL INDEX", "REFFREQ": (ref_freq, "[Hz]")},
                "In overlaps TILES is the brightness-weighted mean of the tiles.",
            ))
            hdus.append(tilepos_hdu(i))
        for i, (flux, alpha, _) in enumerate(lay["points"]):
            hdus.append(point_table(
                i,
                [
                    pyfits.Column("FLUX_REF", "E", unit="mJy", array=flux[:, ref_idx]),
                    pyfits.Column("ALPHA", "E", array=alpha),
                ],
                {"REFFREQ": (ref_freq, "[Hz] frequency of FLUX_REF and ALPHA")},
            ))
        write_hdus(hdus, f"{prefix}_spectral_index_posterior_{tag}.fits")

    # -----------------------------------------------------------------------------------------
    # Spectral curvature per component (multi-frequency models with a curvature term), in the
    # layout of the spectral index file: beta is the coefficient of u^2 = ln(nu / nu_ref)^2 of
    # the model spectrum, ln S = ln I0 + alpha u + beta q(u) + deviations with
    # q(u) = u^2 - (sum u^3 / sum u^2) u (anchored at nu_ref). Components without a curvature
    # term are NaN.
    # -----------------------------------------------------------------------------------------
    def write_spectral_curvature(lay, tag, label):
        hdr, hdr0 = spectral_header("SPECTRAL CURVATURE", label, "curvature")
        hdr0["COMMENT"] = f"Spectral curvature (coefficient of ln(nu/REFFREQ)^2) per sky component ({label})."
        hdr0["COMMENT"] = "Components without a curvature term are NaN."
        hdr0["COMMENT"] = "Extensions: BG, O0..On (cutouts), TILES, TILECUBE; tables: TILEPOS, POINTS."
        hdus = [pyfits.PrimaryHDU(header=hdr0)]
        hdus.append(image_hdu(lay["bg_beta"], hdr, "BG"))
        for i, (beta, off) in enumerate(zip(lay["obj_beta"], obj_offsets, strict=True)):
            hdus.append(image_hdu(beta, fits_header(ref_freq, "", btype="SPECTRAL CURVATURE", offset=off), f"O{i}"))
        for i, (beta, cube) in enumerate(zip(lay["tiles_beta"], lay["tile_cube_beta"], strict=True)):
            hdus.append(image_hdu(beta, hdr, ext_name("TILES", i, len(lay["tiles_beta"]))))
            hdus.append(tilecube_hdu(
                cube, i, {"BUNIT": "", "BTYPE": "SPECTRAL CURVATURE", "REFFREQ": (ref_freq, "[Hz]")},
                "In overlaps TILES is the brightness-weighted mean of the tiles.",
            ))
            hdus.append(tilepos_hdu(i))
        for i, ((flux, _, _), beta) in enumerate(zip(lay["points"], lay["points_beta"], strict=True)):
            hdus.append(point_table(
                i,
                [
                    pyfits.Column("FLUX_REF", "E", unit="mJy", array=flux[:, ref_idx]),
                    pyfits.Column("BETA", "E", array=beta),
                ],
                {"REFFREQ": (ref_freq, "[Hz] frequency of FLUX_REF and BETA")},
            ))
        write_hdus(hdus, f"{prefix}_spectral_curvature_posterior_{tag}.fits")

    # Write the posterior samples (<...>_posterior_sample_<k>.fits) and their mean (<...>_posterior_mean.fits)
    for tag, label, lay in [(f"sample_{k}", f"posterior sample {k}", lay) for k, lay in enumerate(layers)] + [
        ("mean", "posterior mean", mean_layers)
    ]:
        write_brightness(lay, tag, label)
        if multi:
            write_spectral_index(lay, tag, label)
        if has_curvature:
            write_spectral_curvature(lay, tag, label)


def as_float32(tree):
    """Layer tree with every array as float32."""
    import numpy as np

    if isinstance(tree, dict):
        return {k: as_float32(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(as_float32(v) for v in tree)
    return np.asarray(tree, dtype="float32")


def average(trees):
    """Element-wise mean over a list of equally structured layer trees."""
    import numpy as np

    first = trees[0]
    if isinstance(first, dict):
        return {k: average([t[k] for t in trees]) for k in first}
    if isinstance(first, (list, tuple)):
        return type(first)(average([t[i] for t in trees]) for i in range(len(first)))
    return np.mean(trees, axis=0)


if __name__ == "__main__":
    main()
