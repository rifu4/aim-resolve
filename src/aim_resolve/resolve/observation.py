import pickle
import numpy as np
import pandas as pd
from nifty8 import makeOp

from .constants import SPEEDOFLIGHT, DEG2RAD, AS2RAD
from .fast import build_exact_responses


TABLE = {5: 'RR', 6: 'RL', 7: 'LR', 8: 'LL', 9: 'XX', 10: 'XY', 11: 'YX', 12: 'YY'}
INVTABLE = {val: key for key, val in TABLE.items()}



class Observation():
    '''A class to represent a radio interferometric observation. It is based on the resolve observation class.'''

    def __init__(self, pol, freq, antpos, vis, weight, name=None):
        '''
        Initialize the Observation class.

        Parameters
        ----------
        pol : array
            Polarizations.
        freq : array
            Frequencies.
        antpos : Pandas DataFrame
            Antenna positions.
        vis : Pandas DataFrame
            Visibilities.
        weight : Pandas DataFrame
            Weights.
        name : str
            Name of the observed source.
        '''
        self._pol = pol
        self._frq = freq
        self._antpos = antpos
        self._vis = vis
        self._wgt = weight
        self.name = name

    def save(self, fname):
        '''
        Save the observation to a pickle file.

        Parameters
        ----------
        fname : str
            File name of the observation file that is saved.
        '''
        dct = {
            'pol': self._pol, 
            'freq': self._frq, 
            'antpos': self._antpos, 
            'vis': self._vis, 
            'weight': self._wgt,
            'name': self.name,
        }
        if not '.pkl' in fname:
            fname += '.pkl'
        with open(fname, 'wb') as f:
            pickle.dump(dct, f)

    @classmethod
    def load(self, fname):
        '''
        Load an observation from either a pickle or npz file.
        
        Parameters
        ----------
        fname : str
            File name of the observation file that is imported.
        '''
        if '.npz' in fname:
            return self.load_npz(fname)
        elif '.pkl' in fname:
            return self.load_pkl(fname)
        else:
            raise ValueError('Unknown file format')
    
    @classmethod
    def load_pkl(self, fname):
        '''Load an observation from a pickle file.'''
        with open(fname, 'rb') as f:
            dct = pickle.load(f)
        return Observation(**dct)

    @classmethod
    def load_npz(self, fname):
        '''Load an observation from a npz file that was saved using the resolve observation class.'''
        dct = np.load(fname)
        pol = np.array([TABLE[ii] for ii in dct['polarization']]) if dct['polarization'].size > 0 else np.array(['I'])
        freq = dct['freq']
        dfs = []
        for a,k in zip([dct[f'antpos{i}'] for i in range(4)], [['u', 'v', 'w'], ['ant1'], ['ant2'], ['time']]):
            dfs.append(pd.DataFrame(a, columns=[k]))
        df_antpos = pd.concat(dfs, axis=1)
        idx = pd.MultiIndex.from_product([pol, freq], names=['pol', 'freq'])
        vis = np.transpose(dct['vis'], (1,0,2)).reshape((dct['vis'].shape[1], -1))
        df_vis = pd.DataFrame(vis, columns=idx)
        weight = np.transpose(dct['weight'], (1,0,2)).reshape((dct['weight'].shape[1], -1))
        df_weight = pd.DataFrame(weight, columns=idx)
        name = str(dct['auxtable_FIELD_0006'][0]) if 'auxtable_FIELD_0006' in dct else None
        return Observation(pol, freq, df_antpos, df_vis, df_weight, name)
    
    def to_resolve_obs(self):
        '''Convert the observation to a resolve observation.'''
        import resolve as rve

        obs = rve.Observation(
            antenna_positions=rve.AntennaPositions(self.uvw, self.ant1.astype(int), self.ant2.astype(int), self.time),
            vis=self.vis,
            weight=self.weight,
            polarization=rve.Polarization([INVTABLE[p] for p in self.pol if p != 'I']),
            freq=self.freq,
            auxiliary_tables=None,
        )
        return obs

    @property
    def pol(self):
        return self._pol
    
    @property
    def freq(self):
        return self._frq

    @property
    def vis(self):
        vis = self._vis.to_numpy()
        vis = vis.reshape(vis.shape[0], len(self._pol), len(self._frq)).transpose((1,0,2))
        return vis
    
    @property
    def weight(self):
        wgt = self._wgt.to_numpy()
        wgt = wgt.reshape(wgt.shape[0], len(self._pol), len(self._frq)).transpose((1,0,2))
        return wgt
    
    @property
    def uvw(self):
        return self._antpos[['u', 'v', 'w']].to_numpy()
    
    @property
    def uvwlen(self):
        return np.linalg.norm(self.uvw, axis=1)
    
    @property
    def ant1(self):
        return self._antpos['ant1'].to_numpy()[:,0]

    @property
    def ant2(self):
        return self._antpos['ant2'].to_numpy()[:,0]

    @property
    def time(self):
        return self._antpos['time'].to_numpy()[:,0]
    
    @property
    def npol(self):
        return self._pol.size
    
    @property
    def nfreq(self):
        return self._frq.size

    @property
    def nrow(self):
        return self._antpos.shape[0]
    
    @property
    def nvis(self):
        return self._vis.size
    
    @property
    def flags(self):
        return self._wgt == 0.0
    
    @property
    def mask(self):
        return self._wgt > 0.0
    
    @property
    def nvis_effective(self):
        return self.mask.sum().sum()

    @property
    def useful_fraction(self):
        return self.nvis_effective / self.nvis
    
    @property
    def baselines(self):
        return set((a1, a2) for a1, a2 in zip(self.ant1, self.ant2))

    @property
    def nbaselines(self):
        return len(self.baselines)
    
    @property
    def precision(self):
        if self.vis.dtype == np.complex64:
            return 'single'
        elif self.vis.dtype == np.complex128:
            return 'double'
        else:
            raise ValueError('unknown precision')
    
    def dirty_image(self, space):
        '''Compute the dirty image of the observation.'''
        obs = self.to_resolve_obs()
        N_inv = makeOp(obs.weight)
        R, *_ = build_exact_responses(obs, space)
        dirty = R.adjoint(N_inv(obs.vis))
        return np.array(dirty.val)
        
    def to_double_precision(self):
        return Observation(self._pol, self._frq, self._antpos, self._vis.astype(np.complex128), self._wgt.astype(np.float64), self.name)

    def to_single_precision(self):
        return Observation(self._pol, self._frq, self._antpos, self._vis.astype(np.complex64), self._wgt.astype(np.float32), self.name)
    
    def average_stokesi(self):
        if np.all(self.pol == 'I'):
            return self
        elif 'XX' in self.pol and 'YY' in self.pol:
            vis = self._vis['XX'] * self._wgt['XX'] + self._vis['YY'] * self._wgt['YY']
            wgt = self._wgt['XX'] + self._wgt['YY']
        elif 'RR' in self.pol and 'LL' in self.pol:
            vis = self._vis['RR'] * self._wgt['RR'] + self._vis['LL'] * self._wgt['LL']
            wgt = self._wgt['RR'] + self._wgt['LL']
        else:
            raise ValueError('polarizations cannot be averaged, only stokes I or XX/YY or RR/LL')
        invmask = wgt == 0.0
        vis /= wgt + np.ones_like(wgt) * invmask
        vis[invmask] = 0.0
        return Observation(np.array(['I']), self._frq, self._antpos, vis, wgt, self.name)
    
    def restrict_to_pol(self, pol):
        if pol not in self.pol:
            raise ValueError(f'polarization {pol} not present in observation')
        pol = np.array([pol])
        vis = self._vis.loc[:, self._vis.columns.get_level_values('pol').isin(pol)]
        wgt = self._wgt.loc[:, self._wgt.columns.get_level_values('pol').isin(pol)]
        return Observation(pol, self._frq, self._antpos, vis, wgt, self.name)
    
    def restrict_by_time(self, tmin, tmax, with_index=False):
        start, stop = np.searchsorted(self.time, [tmin, tmax])
        ind = slice(start, stop)
        res = self[ind]
        if with_index:
            return res, ind
        return res
    
    def get_freqs(self, freq_list):
        slc = np.zeros(self.nfreq, dtype=bool)
        slc[freq_list] = 1
        return self.get_freqs_by_slice(slc)
    
    def restrict_by_freq(self, fmin, fmax, with_index=False):
        start, stop = np.searchsorted(self.freq, [fmin, fmax])
        slc = slice(start, stop)
        res = self.get_freqs_by_slice(slc)
        if with_index:
            return res, slc
        return res

    def get_freqs_by_slice(self, slc):
        frq = self._frq[slc]
        vis = self._vis.loc[:, self._vis.columns.get_level_values('freq').isin(frq)]
        wgt = self._wgt.loc[:, self._wgt.columns.get_level_values('freq').isin(frq)]
        return Observation(self._pol, frq, self._antpos, vis, wgt, self.name)        
    
    def subsample_rows(self, n):
        if 0 < n < 1:
            n = int(n * self.nrow)
        elif 1 <= n <= self.nrow:
            n = int(n)
        else:
            raise ValueError('n must be positive and less than the number of rows')
        slc = np.random.choice(self.nrow, n)
        return Observation(self._pol, self._frq, self._antpos.iloc[slc], self._vis.iloc[slc], self._wgt.iloc[slc], self.name)
    
    def flags_to_nan(self):
        if self.useful_fraction == 1.0:
            return self
        vis = self._vis.copy()
        vis[self.flags] = np.nan
        return Observation(self._pol, self._frq, self._antpos, vis, self._wgt, self.name)
    
    def __repr__(self):
        short0 = self.uvwlen.min()
        long0 = self.uvwlen.max()
        short1 = 1/(short0*self.freq.min()/SPEEDOFLIGHT)
        long1 = 1/(long0*self.freq.max()/SPEEDOFLIGHT)
        s = [
            f'source name:\t\t{self.name}',
            f'visibilities shape:\t{self.vis.shape}',
            f'# visibilities:\t{self.nvis}',
            f'frequency range:\t{self.freq.min()*1e-6:.3f} -- {self.freq.max()*1e-6:.3f} MHz',
            f'polarizations:\t' + ', '.join(self.pol),
            f'shortest baseline:\t{short0:.1f} m -> {short1/DEG2RAD:.3f} deg',
            f'longest baseline:\t{long0:.1f} m -> {long1/AS2RAD:.3f} arcsec',
            f'flagged:\t\t{(1-self.useful_fraction)*100:.1f}%',
        ]
        return '\n'.join(['Observation:'] + [f'  {ss}' for ss in s])
