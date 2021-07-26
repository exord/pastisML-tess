import numpy as np

from matplotlib import pyplot as plt


def plot_sample_lcs(f, n_plot=10):
    """Plot LCs."""
    # Randomly choose curves to plot
    ii = np.random.randint(len(f), size=n_plot)
    print(ii)

    f_to_plot = np.array(f)[ii]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    for ff in f_to_plot:
        tt = np.linspace(0, 1, len(ff[0]))
        ax.plot(tt, ff[0], alpha=0.8)

    return
