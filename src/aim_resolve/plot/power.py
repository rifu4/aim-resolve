import matplotlib.pyplot as plt
import numpy as np

from .util import plot_figure



def plot_power(
        array,
        axes = None, 
        label = None,
        name = None,
        odir = None,
        plot_label = True,
    ):
    '''
    Plot a 1D power spectrum of the nifty8.re correlated field model.
    
    Parameters
    ----------
    array : np.ndarray
        The array to plot.
    axes : list of plt.Axes, optional
        The axes to plot on. If not provided, a new figure will be created.
    label : str, optional
        The label of the plot. Default is None.
    name : str, optional
        The name of the plot. Default is None.
    odir : str, optional
        The output directory to save the plot. Default is None.
    plot_label : bool, optional
        Whether to plot the label as title. Default is True.
    '''
    plot_now = False
    if axes is None:
        figure = plt.figure(figsize=(5,5))
        axes = []
        axes.append(figure.add_subplot(1, 1, 1))
        plot_now = True

    array = np.array(array, dtype='float64')
    
    k = np.arange(0, array.size)

    axes[-1].loglog(k, array)

    if plot_label and label:
        axes[-1].set_title(label)

    if plot_now:
        plot_figure(figure, odir, name)
