# %%
import numpy as np
import resolve as rve

import aim_resolve as aim

# %%
obs_dir = "/scratch/users/rfuchs/data/eso_data_from_oleg.npz"
obs0 = rve.Observation.load(obs_dir)
print(obs0)

# %%
obs0_i = obs0.restrict_to_stokesi()
obs0_i = obs0_i.average_stokesi()
print(obs0_i)

# %%
obs0_a, idx_a = obs0_i.restrict_by_freq(961e6, 1145e6, True)
print("index_a:", idx_a)
print(obs0_a)
# obs0_a.save('/scratch/users/rfuchs/data/eso_961-1145mhz.npz', False)

# %%
obs0_b, idx_b = obs0_i.restrict_by_freq(1295e6, 1503e6, True)
print("index_b:", idx_b)
print(obs0_b)
# obs0_b.save('/scratch/users/rfuchs/data/eso_1295-1503mhz.npz', False)

# %%
idx_c = [i for s in (idx_a, idx_b) for i in range(s.start, s.stop)]
print("index_c:", idx_c)

obs0_c = obs0_i.get_freqs(idx_c)
print(obs0_c)
obs0_c.save("/scratch/users/rfuchs/data/eso_961-1503mhz.npz", False)

# %%

# %%
obs_dir = "/scratch/users/rfuchs/data/eso_961-1145mhz.npz"
obs0 = rve.Observation.load(obs_dir)
print(obs0)

# %%
freq_a = [float(s.mean()) for s in np.array_split(obs0.freq, 3)]
idx_a = [len(s) for s in np.array_split(obs0.freq, 3)]
print("freq_a:", freq_a)
print("idx_a:", idx_a)
obs1 = obs0.get_freqs(range(74, 74 + 73))
print(obs1)

# %%
print(obs1.freq.mean())
obs1.save("/scratch/users/rfuchs/data/eso_1024-1084mhz.npz", False)

# %%
obs_dir = "/scratch/users/rfuchs/data/eso_961-1503mhz.npz"

obs1 = aim.radio_data(fname=obs_dir)
freq = obs1.freq

# %%
freq_a = freq[(freq < 1200e6)]
freq_b = freq[(freq >= 1200e6)]

freq_a = [float(s.mean()) for s in np.array_split(freq_a, 3)]
print("freq_a:", freq_a)
freq_b = [float(s.mean()) for s in np.array_split(freq_b, 3)]
print("freq_b:", freq_b)

freq = freq_a + freq_b
print("freq:", freq)

# %%
obs_dir = "/Users/rf/Development/data/eso_986-1137mhz.npz"

obs1 = aim.radio_data(fname=obs_dir)
freq = obs1.freq

for f in freq:
    print(f)

print([float(freq[:4].mean()), float(freq[4:].mean())])
print(freq.mean())

# %%
obs_dir = "/Users/rf/Development/data/eso_1356-1439mhz.npz"

obs2 = aim.radio_data(fname=obs_dir)
freq = obs2.freq

for f in freq:
    print(f)

print([float(freq[:4].mean()), float(freq[4:].mean())])
