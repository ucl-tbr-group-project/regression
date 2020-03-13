import matplotlib.pyplot as plt


def set_plotting_style(dark=False):
    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 16,  # fontsize for x and y labels (was 10)
              'axes.titlesize': 18,
              'font.size': 14,  # was 10
              'legend.fontsize': 16,  # was 10
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'text.usetex': True,
              'figure.figsize': [9, 8],
              'font.family': 'serif'
              }

    plt.rcParams.update(params)

    if dark:
        plt.style.use('dark_background')
