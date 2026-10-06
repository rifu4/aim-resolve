# %%
# ---------------------------------------------------------------------------
# GPU / JAX environment setup (must run before importing jax).
# ---------------------------------------------------------------------------
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# %%
# ---------------------------------------------------------------------------
# Imports.
# ---------------------------------------------------------------------------
import pickle
import time

import astropy.io.fits as pyfits
import numpy as np
from astropy.time import Time
from astropy.wcs import WCS

import aim_resolve as aim
from aim_resolve.model.map import point_slice_indices, signal_slice_indices

# %%
# ---------------------------------------------------------------------------
# Export the full-sky (2 deg, 3072 x 3072) reconstruction to FITS, in the
# format of the fast-resolve products (eso136-006_1053MHz_posterior_mean.fits):
#   <PREFIX>_<nu>MHz_posterior_<tag>.fits       : sky brightness at each frequency
#   <PREFIX>_spectral_index_posterior_<tag>.fits: spectral index at nu_ref
# with <tag> = sample_<k> for every posterior sample and mean for their mean.
# Each is a multi-extension FITS file with one layer per sky component (see
# below); the brightness files hold the total sky in the primary HDU.
# Header and axis conventions follow `resolve.ubik_tools.fits.field2fits`,
# which wrote the fast-resolve files: 4D data (STOKES, FREQ, DEC, RA), array
# axis 0 -> RA (NAXIS1, CDELT1 < 0), axis 1 -> DEC. The phase centre is at
# the 0-based pixel N / 2, i.e. CRPIX = N / 2 + 1 (field2fits writes N / 2, which
# shifts the coordinates by one pixel; checked against the uSARA image).
# ---------------------------------------------------------------------------
dir = "/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b"

mf_rec = "4_rec_3z_4_6f_1_it_1_it_1"
mf_it = 6

ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
CONV_FACTOR = 1000 * AS2RAD**2  # Jy/sr -> mJy/arcsec^2
REF_IDX = 1  # reference-frequency index of the spectral model (ref_freq = freq[1])

PREFIX = "eso137"
odir = f"{dir}/opt/{mf_rec}/fits"

opt_yml = f"{dir}/opt/{mf_rec}/opt.yml"
print("load:", opt_yml)

optim_cfg = aim.OptimizeKLConfig.from_file(opt_yml, aim.get_builders)
sky_mf = optim_cfg.instantiate_sec(f"sky.{mf_it}")
print("sky components:", [c.prefix for c in sky_mf.models])

freq_hz = np.asarray(sky_mf.freq, dtype="float64")
print("sky freqs:", [f"{round(f / 1e6)} MHz" for f in freq_hz], "| ref freq:", f"{round(freq_hz[REF_IDX] / 1e6)} MHz")

with open(f"{dir}/opt/{mf_rec}/last.pkl", "rb") as f:
    samples_mf, *_ = pickle.load(f)
print("samples:", len(samples_mf))

# Phase centre (RA, DEC) [rad] from the FIELD table of the multi-frequency data.
data_fn = optim_cfg.sections[f"data.{mf_it}"]["fname"]
with np.load(data_fn, allow_pickle=True) as data:
    field_keys = list(data["auxtable_FIELD_0000"])
    phase_dir = data[f"auxtable_FIELD_{field_keys.index('PHASE_DIR') + 1:04d}"]
ra0, dec0 = np.asarray(phase_dir, dtype="float64").reshape(-1)[:2]
print(f"phase centre: RA {np.rad2deg(ra0):.6f} deg, DEC {np.rad2deg(dec0):.6f} deg")

# Pixel size [rad] of the full sky grid.
grid = sky_mf.grid
pix_rad = np.asarray(grid.dis, dtype="float64") / grid.fac
nx, ny = grid.shape
print(f"sky grid: {grid.shape} | pixel size: {np.rad2deg(pix_rad[0]) * 3600:.3f} arcsec")


# %%
# ---------------------------------------------------------------------------
# FITS helpers.
# ---------------------------------------------------------------------------
def fits_header(freq, bunit, btype=None, offset=(0, 0)):
    """Header of a single-frequency, single-Stokes sky image; `offset` is the
    0-based start pixel of a cutout on the full sky grid (shifts CRPIX)."""
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


# %%
# ---------------------------------------------------------------------------
# Component layers, shared by all files (one multi-extension FITS file each):
#   BG       background, full sky grid
#   O0..On   objects, each on its own (native) grid as a cutout with its own
#            WCS (CRPIX shifted by the cutout offset on the sky grid)
#   TILES    tiles on the full sky grid, NaN outside the tiles
#   TILECUBE the individual tiles as a cube (tile, x, y), e.g. for the tile
#            plots, since overlapping tiles cannot be separated in TILES
#   TILEPOS  table of the tile positions on the sky grid (row = cube index)
#   POINTS   table of the point sources. Each source covers 3 x 3 sub-pixels of
#            the point grid (grouped by their coarse pixel); positions are the
#            nu_ref-brightness-weighted sub-pixel centroids.
# ---------------------------------------------------------------------------
dvol = float(grid.dvol)  # pixel solid angle [sr]
REF_FREQ = freq_hz[REF_IDX]
wcs_sky = WCS(fits_header(REF_FREQ, "")).celestial
obj_offsets = [np.asarray(signal_slice_indices(o.grid, grid)[1]) for o in sky_mf.objects]
for i, (o, off) in enumerate(zip(sky_mf.objects, obj_offsets)):
    print(f"  O{i} ({o.prefix}): cutout {o.grid.shape} at sky pixel {tuple(off.tolist())}")
tile_offsets = []
for t in sky_mf.tiles:
    t_in, t_out = (np.asarray(v) for v in signal_slice_indices(t.tiles.grid, grid))
    if np.any(t_in != 0):
        raise ValueError("tiles reaching beyond the sky grid are not supported")
    tile_offsets.append(t_out)  # (n_tiles, 2) 0-based start pixel on the sky grid


def tilecube_hdu(cube, i, header_cards, comment):
    """Cube extension of the individual tiles (n_tiles, tx, ty), stored as
    (tile, DEC, RA) like the sky images; the tile index is the TILEPOS row."""
    h = pyfits.Header()
    h["EXTNAME"] = ext_name("TILECUBE", i, len(sky_mf.tiles))
    for key, val in header_cards.items():
        h[key] = val
    h["COMMENT"] = "Individual tiles: axis 3 = tile index (row of TILEPOS), axes 1 / 2 ="
    h["COMMENT"] = "tile pixels along RA / DEC (same orientation as the sky images)."
    h["COMMENT"] = comment
    return pyfits.ImageHDU(np.asarray(cube, dtype="float32").transpose(0, 2, 1), header=h)


def tilepos_hdu(i):
    """TILEPOS table: start pixel, size and centre of every tile on the sky grid."""
    off = tile_offsets[i]
    tx, ty = sky_mf.tiles[i].tiles.grid.shape
    ra, dec = wcs_sky.pixel_to_world_values(off[:, 0] + (tx - 1) / 2, off[:, 1] + (ty - 1) / 2)
    return pyfits.BinTableHDU.from_columns([
        pyfits.Column("TILE", "J", array=np.arange(len(off))),
        pyfits.Column("X0_PIX", "J", unit="pix", array=off[:, 0] + 1),  # FITS (1-based) sky pixel of tile pixel (1, 1)
        pyfits.Column("Y0_PIX", "J", unit="pix", array=off[:, 1] + 1),
        pyfits.Column("NX", "J", unit="pix", array=np.full(len(off), tx)),
        pyfits.Column("NY", "J", unit="pix", array=np.full(len(off), ty)),
        pyfits.Column("RA", "D", unit="deg", array=ra),   # tile centre
        pyfits.Column("DEC", "D", unit="deg", array=dec),
    ], name=ext_name("TILEPOS", i, len(sky_mf.tiles)))


def point_catalogue(m, x):
    """Per point source: flux density [mJy] at every frequency (n_src, nfreq),
    spectral index at nu_ref and centroid pixel (0-based, sky grid)."""
    pix = np.asarray(point_slice_indices(m.points.grid, grid)[1], dtype="float64")  # (n_sub, 2)
    coarse = np.floor(np.asarray(m.points.grid.coos)).astype(int)
    _, src = np.unique(coarse, axis=0, return_inverse=True)
    src = src.reshape(-1)
    n_src, n_sub = src.max() + 1, src.size

    def per_source(v):
        return np.bincount(src, v, n_src)

    s_freq = np.asarray(m(x, map=False)).reshape(n_sub, len(freq_hz))  # [Jy/sr] per sub-pixel
    flux = np.stack([per_source(s_freq[:, fi]) for fi in range(len(freq_hz))], axis=1) * dvol * 1e3
    s_ref = np.asarray(m.ref_freq_model(x, map=False)).reshape(-1)
    alpha = np.asarray(m.spectral_index(x, map=False)).reshape(-1)
    w = per_source(s_ref)
    alpha_src = per_source(s_ref * alpha) / w
    px = np.stack([per_source(s_ref * pix[:, i]) / w for i in range(2)], axis=1)
    return flux, alpha_src, px


def tile_spectral_index(m, x):
    """Spectral index of the tiles on the sky grid, NaN outside. Where tiles
    overlap it is brightness-weighted, sum_i S_i alpha_i / sum_i S_i; elsewhere
    (or if all weights vanish, e.g. far in the tapered tile wings) the plain
    tile index, so that the weights cannot underflow to 0 / 0."""
    ref, alpha = m.ref_freq_model, m.spectral_index
    s_ref, a_tile = ref(x, map=False), alpha(x, map=False)
    num = np.squeeze(np.asarray(ref.map_function(s_ref * a_tile)))
    den = np.squeeze(np.asarray(ref.map_function(s_ref)))
    a_sum = np.squeeze(np.asarray(alpha.map_function(a_tile)))
    n_cov = np.squeeze(np.asarray(alpha.map_function(np.ones_like(a_tile))))
    plain = np.where(n_cov > 0, a_sum / np.where(n_cov > 0, n_cov, 1.0), np.nan)
    weighted = num / np.where(den > 0, den, 1.0)
    return np.where((n_cov > 1) & (den > 0), weighted, plain)


# %%
# ---------------------------------------------------------------------------
# Evaluate every posterior sample once into all component layers; the posterior
# mean is the average of these layers over the samples. Brightness layers are in
# mJy/arcsec^2, point-source flux densities in mJy.
# ---------------------------------------------------------------------------
tile_masks = [np.asarray(t.mask) for t in sky_mf.tiles]


def sample_layers(x):
    """All component layers of one posterior sample `x` (float32)."""
    lay = dict(
        sky=np.asarray(sky_mf(x)) * CONV_FACTOR,                                  # (nfreq, nx, ny)
        bg=np.asarray(sky_mf.background(x)) * CONV_FACTOR,                        # (nfreq, nx, ny)
        obj=[np.asarray(o(x)) * CONV_FACTOR for o in sky_mf.objects],             # (nfreq, *cutout)
        tiles=[np.where(mk, np.asarray(t(x)) * CONV_FACTOR, np.nan) for t, mk in zip(sky_mf.tiles, tile_masks)],
        tile_cube=[np.asarray(t(x, map=False)) * CONV_FACTOR for t in sky_mf.tiles],  # (n_tiles, nfreq, tx, ty)
        bg_alpha=np.asarray(sky_mf.background.spectral_index(x)),
        obj_alpha=[np.asarray(o.spectral_index(x)) for o in sky_mf.objects],
        tiles_alpha=[tile_spectral_index(t, x) for t in sky_mf.tiles],
        tile_cube_alpha=[np.asarray(t.spectral_index(x, map=False)) for t in sky_mf.tiles],
        points=[point_catalogue(p, x) for p in sky_mf.points],                   # (flux, alpha, px)
    )
    return as_float32(lay)


def as_float32(tree):
    if isinstance(tree, dict):
        return {k: as_float32(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(as_float32(v) for v in tree)
    return np.asarray(tree, dtype="float32")


def average(trees):
    """Element-wise mean over a list of equally structured layer trees."""
    first = trees[0]
    if isinstance(first, dict):
        return {k: average([t[k] for t in trees]) for k in first}
    if isinstance(first, (list, tuple)):
        return type(first)(average([t[i] for t in trees]) for i in range(len(first)))
    return np.mean(trees, axis=0)


print("evaluating the posterior samples ...")
layers = [sample_layers(s) for s in samples_mf]
mean_layers = average(layers)

# Point-source positions: posterior-mean centroids, identical in every file, so
# that the POINTS rows match across frequencies and samples.
points = []
for p, (_, _, px) in zip(sky_mf.points, mean_layers["points"]):
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


# %%
# ---------------------------------------------------------------------------
# Sky brightness [mJy/arcsec^2], one multi-extension file per frequency:
#   PRIMARY  total sky (sum of all components), then the component layers
#            above; the POINTS table gives the flux density [mJy] per source.
# ---------------------------------------------------------------------------
def write_brightness(lay, tag, label):
    for fi, f in enumerate(freq_hz):
        hdr = fits_header(f, "mJy/arcsec^2")
        hdr0 = hdr.copy()
        hdr0["PRODUCT"] = label
        hdr0["COMMENT"] = f"Total sky brightness (sum of all components, {label})."
        hdr0["COMMENT"] = "Extensions: BG, O0..On (cutouts), TILES, TILECUBE; tables: TILEPOS, POINTS."
        hdus = [pyfits.PrimaryHDU(np.asarray(lay["sky"][fi], dtype="float32").T[None, None], header=hdr0)]
        hdus.append(image_hdu(lay["bg"][fi], hdr, "BG"))
        for i, (o, off) in enumerate(zip(lay["obj"], obj_offsets)):
            hdus.append(image_hdu(o[fi], fits_header(f, "mJy/arcsec^2", offset=off), f"O{i}"))
        for i, (t, cube) in enumerate(zip(lay["tiles"], lay["tile_cube"])):
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
        write_hdus(hdus, f"{PREFIX}_{round(f / 1e6)}MHz_posterior_{tag}.fits")


# %%
# ---------------------------------------------------------------------------
# Spectral index at nu_ref per component:
#   PRIMARY  empty (global keywords only), then the component layers above.
#   Overlapping tiles are combined brightness-weighted, sum_i S_i alpha_i / sum_i S_i;
#   the POINTS table gives the flux density at nu_ref and the spectral index.
# ---------------------------------------------------------------------------
def write_spectral_index(lay, tag, label):
    hdr = fits_header(REF_FREQ, "", btype="SPECTRAL INDEX")
    hdr0 = pyfits.Header()
    for key in ("ORIGIN", "OBSRA", "OBSDEC", "DATE-MAP", "EQUINOX"):
        hdr0[key] = hdr[key]
    hdr0["BTYPE"] = "SPECTRAL INDEX"
    hdr0["REFFREQ"] = (REF_FREQ, "[Hz] reference frequency of the spectral index")
    hdr0["PRODUCT"] = label
    hdr0["COMMENT"] = f"Spectral index at REFFREQ per sky component ({label})."
    hdr0["COMMENT"] = "Extensions: BG, O0..On (cutouts), TILES, TILECUBE; tables: TILEPOS, POINTS."
    hdus = [pyfits.PrimaryHDU(header=hdr0)]
    hdus.append(image_hdu(lay["bg_alpha"], hdr, "BG"))
    for i, (alpha, off) in enumerate(zip(lay["obj_alpha"], obj_offsets)):
        hdus.append(image_hdu(alpha, fits_header(REF_FREQ, "", btype="SPECTRAL INDEX", offset=off), f"O{i}"))
    for i, (alpha, cube) in enumerate(zip(lay["tiles_alpha"], lay["tile_cube_alpha"])):
        hdus.append(image_hdu(alpha, hdr, ext_name("TILES", i, len(lay["tiles_alpha"]))))
        hdus.append(tilecube_hdu(
            cube, i, {"BUNIT": "", "BTYPE": "SPECTRAL INDEX", "REFFREQ": (REF_FREQ, "[Hz]")},
            "In overlaps TILES is the brightness-weighted mean of the tiles.",
        ))
        hdus.append(tilepos_hdu(i))
    for i, (flux, alpha, _) in enumerate(lay["points"]):
        hdus.append(point_table(
            i,
            [
                pyfits.Column("FLUX_REF", "E", unit="mJy", array=flux[:, REF_IDX]),
                pyfits.Column("ALPHA", "E", array=alpha),
            ],
            {"REFFREQ": (REF_FREQ, "[Hz] frequency of FLUX_REF and ALPHA")},
        ))
    write_hdus(hdus, f"{PREFIX}_spectral_index_posterior_{tag}.fits")


# %%
# ---------------------------------------------------------------------------
# Write the posterior samples (<...>_posterior_sample_<k>.fits) and their mean
# (<...>_posterior_mean.fits).
# ---------------------------------------------------------------------------
for k, lay in enumerate(layers):
    write_brightness(lay, f"sample_{k}", f"posterior sample {k}")
    write_spectral_index(lay, f"sample_{k}", f"posterior sample {k}")
write_brightness(mean_layers, "mean", "posterior mean")
write_spectral_index(mean_layers, "mean", "posterior mean")
