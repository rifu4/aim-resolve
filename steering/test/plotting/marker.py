import numpy as np

from aim_resolve import plot_arrays
from aim_resolve.plot.util import to_shape

#%%
ps_map = np.zeros((10,10))
ps_map[2, 2] = 1
ps_map[8, 8] = 1

oj_map = np.zeros((10,10))
oj_map[4:7, 4:7] = 1
oj_map[5, 5] = 0

plot_arrays([ps_map, oj_map], dpi=50, rows=1)

#%%
px, py = np.argwhere(ps_map == 1).T
ps_mrk = dict(x=px, y=py, s=1, c='white', marker='+')
print(ps_mrk)

ox, oy = np.argwhere(oj_map == 1).T
oj_mrk = dict(x=ox, y=oy, s=1, c='white', marker=',')
print(oj_mrk)

#%%
marker = dict(ps_mrk=ps_mrk, oj_mrk=oj_mrk)

m0 = to_shape(marker, (1, 1))
print(tuple(m0[0,0].values()))

plot_arrays([np.ones((10, 10)), np.zeros((10, 10))], marker=[marker, oj_mrk], rows=1, dpi=100)

#%%
