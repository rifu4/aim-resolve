# builds the PSF and noise kernels of the real data with the rescaled weights: the sections of an existing generated config, with
# the data and kernel names replaced; kernels are built when their file is missing
import os, re, sys, tempfile
os.environ.setdefault("JAX_PLATFORMS", "cpu")
ST = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(ST)
import aim_resolve as aim
S = tempfile.mkdtemp()  # the generated configs
for src in sys.argv[1:]:
    t = open(src).read()
    t = re.sub(r"(data/eso_[0-9-]+mhz)(?:_rows3pct)?\.npz", r"\1_wfix.npz", t)
    t = re.sub(r"(kernels/[pn]k_eso_[^\s]*?)\.pkl", r"\1_wfix.pkl", t)
    tmp = f"{S}/kernel_cfg_{os.path.basename(src)}"; open(tmp, "w").write(t)
    cfg = aim.OptimizeKLConfig.from_file(("runs/fast_vi_1f_1024_1z_b/files/base.yml", tmp), aim.get_builders)
    secs = sorted((k for k in cfg.sections if k.startswith("lh.")), key=lambda k: int(k.split(".")[1]))
    print(src, "sections", secs, flush=True)
    lh = cfg.instantiate_sec(secs[-1])
    print("built", secs[-1], cfg.sections[secs[-1]].get("psf_kernel_fn"), cfg.sections[secs[-1]].get("n_inv_kernel_fn"), flush=True)
