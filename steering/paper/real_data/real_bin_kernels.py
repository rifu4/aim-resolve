# PSF and noise kernels of the one-MF-bin SF data (eso_1007-1053mhz_wfix.npz): the zoomed SF stage of 4_pre_3z_14.yml with
# the data and kernel names replaced; the kernels are built on instantiation of its last likelihood section
import os, re, tempfile
ST = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(ST)
import aim_resolve as aim
S = tempfile.mkdtemp()  # the generated configs
t = open("runs/fast_vi_1f_1024_1z_b/files/4_pre_3z_14.yml").read()
t = re.sub(r"data/eso_986-1137mhz(?:_rows3pct)?\.npz", "data/eso_1007-1053mhz_wfix.npz", t)
t = re.sub(r"(kernels/[pn]k_eso_)986-1137mhz(_[^\s]*?)\.pkl", r"\g<1>1007-1053mhz\2_wfix.pkl", t)
tmp = f"{S}/kernel_cfg_bin.yml"; open(tmp, "w").write(t)
cfg = aim.OptimizeKLConfig.from_file(("runs/fast_vi_1f_1024_1z_b/files/base.yml", tmp), aim.get_builders)
sec = sorted((k for k in cfg.sections if k.startswith("lh.")), key=lambda k: int(k.split(".")[1]))[-1]
print(sec, cfg.sections[sec].get("psf_kernel_fn"), cfg.sections[sec].get("n_inv_kernel_fn"), "| data", [cfg.sections[k].get("fname") for k in cfg.sections if k.startswith("data")], flush=True)
lh = cfg.instantiate_sec(sec)
print("built", sec, flush=True)
