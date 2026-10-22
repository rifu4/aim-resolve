# %%
# ---------------------------------------------------------------------------
# MeerKAT L-band primary beam (Stokes I) at every model frequency of the
# multi-frequency reconstruction, on the full sky grid (2 deg, 3072 x 3072).
#
# The forward model does not contain the primary beam, so the reconstructed
# sky is the *apparent* (beam-attenuated) sky. The intrinsic sky is
#   sky_true(nu) = sky_rec(nu) / PB(nu),
# which is only meaningful where PB is not too small (mask e.g. PB < 0.1).
#
# Beam model: resolve.library.meerkat_beam.JimBeam ("MKAT-AA-L-JIM-2020",
# Mauch et al. 2020 cos^3 fit, 900-1670 MHz). JimBeam.I expects offsets from
# the pointing centre in *degrees* and the frequency in *MHz*.
# The observation is a single pointing (DELAY_DIR = PHASE_DIR), so the beam is
# centred on the phase centre at 0-based pixel N / 2 (cf. save_fits.py).
#
# Output: <odir>/eso137_<nu>MHz_primary_beam.fits (same WCS as the sky files)
#         <odir>/eso137_primary_beam.npy          (nfreq, nx, ny), axis 0 -> RA
# ---------------------------------------------------------------------------
import os
import time

import astropy.io.fits as pyfits
import numpy as np
from astropy.time import Time
from resolve.library.meerkat_beam import JimBeam

import aim_resolve as aim

dir = "/scratch/users/rfuchs/packages/aim-resolve/steering/runs/fast_vi_1f_1024_1z_b"
mf_rec = "4_rec_3z_4_6f_1_it_1_it_1"
mf_it = 6
PREFIX = "eso137"
odir = f"{dir}/opt/{mf_rec}/fits"

optim_cfg = aim.OptimizeKLConfig.from_file(f"{dir}/opt/{mf_rec}/opt.yml", aim.get_builders)
sky_mf = optim_cfg.instantiate_sec(f"sky.{mf_it}")
freq_hz = np.asarray(sky_mf.freq, dtype="float64")

data_fn = optim_cfg.sections[f"data.{mf_it}"]["fname"]
with np.load(data_fn, allow_pickle=True) as data:
    field_keys = list(data["auxtable_FIELD_0000"])
    phase_dir = data[f"auxtable_FIELD_{field_keys.index('PHASE_DIR') + 1:04d}"]
    delay_dir = data[f"auxtable_FIELD_{field_keys.index('DELAY_DIR') + 1:04d}"]
assert np.allclose(phase_dir, delay_dir), "pointing != phase centre, beam would need an offset"
ra0, dec0 = np.asarray(phase_dir, dtype="float64").reshape(-1)[:2]

grid = sky_mf.grid
pix_deg = np.rad2deg(np.asarray(grid.dis, dtype="float64") / grid.fac)
nx, ny = grid.shape

# Offsets from the pointing centre [deg]; phase centre at 0-based pixel N / 2.
# (SIN projection: l, m are direction cosines; at <= 1.4 deg off-axis the
# difference to the true angular distance is negligible.)
x = (np.arange(nx) - nx / 2) * pix_deg[0]
y = (np.arange(ny) - ny / 2) * pix_deg[1]
xx, yy = np.meshgrid(x, y, indexing="ij")

beam = JimBeam("MKAT-AA-L-JIM-2020")
pb = np.stack([beam.I(xx, yy, f / 1e6) for f in freq_hz])  # (nfreq, nx, ny)
print("PB at centre:", pb[:, nx // 2, ny // 2])
print("PB at field edge (1 deg):", pb[:, 0, ny // 2])


# %%
def fits_header(freq):
    h = pyfits.Header()
    h["ORIGIN"] = "aim-resolve"
    h["BUNIT"] = ""
    h["BTYPE"] = "Primary beam (MKAT-AA-L-JIM-2020, Stokes I)"
    h["OBSRA"] = np.rad2deg(ra0)
    h["OBSDEC"] = np.rad2deg(dec0)
    h["CTYPE1"] = "RA---SIN"
    h["CRVAL1"] = np.rad2deg(ra0)
    h["CDELT1"] = -pix_deg[0]
    h["CRPIX1"] = nx / 2 + 1
    h["CUNIT1"] = "deg"
    h["CTYPE2"] = "DEC--SIN"
    h["CRVAL2"] = np.rad2deg(dec0)
    h["CDELT2"] = pix_deg[1]
    h["CRPIX2"] = ny / 2 + 1
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


os.makedirs(odir, exist_ok=True)
for f, p in zip(freq_hz, pb):
    fn = f"{odir}/{PREFIX}_{round(f / 1e6)}MHz_primary_beam.fits"
    pyfits.PrimaryHDU(p.astype("float32").T[None, None], header=fits_header(f)).writeto(fn, overwrite=True)
    print(f"saved {fn}")
np.save(f"{odir}/{PREFIX}_primary_beam.npy", pb.astype("float32"))
print(f"saved {odir}/{PREFIX}_primary_beam.npy")
