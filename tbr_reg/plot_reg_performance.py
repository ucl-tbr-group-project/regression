import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.linear_model import Ridge

from .plot_utils import density_scatter
from .metric_loader import get_metric_factory


def plot_reg_performance(df, density_bins=80):
    '''
    Display true vs. predicted TBR in a density scatter plot.
    Requires a data frame with tbr and tbr_pred columns.
    If density_bins are None, then a regular (non-density-colored) plot is used.
    '''
    xs = df['tbr'].to_numpy()
    ys = df['tbr_pred'].to_numpy()

    linreg = Ridge().fit(xs.reshape(-1, 1), ys.reshape(-1, 1))
    ys_fit = linreg.predict(xs.reshape(-1, 1))

    fig, ax = plt.subplots()

    if density_bins is None:
        ax.scatter(xs, ys, s=5, label='Data')
    else:
        density_scatter(xs, ys, ax=ax, bins=density_bins, s=5, label='Data')

    ax.plot(xs, xs, '--', c='k', linewidth=1,
            dashes=[5, 10], label='Ideal model')
    ax.plot(xs, ys_fit, '--', c='r', linewidth=1,
            dashes=[5, 10], label='Trained model')

    ax.set_xlabel('True TBR')
    ax.set_ylabel('Predicted TBR')
    ax.legend(loc='lower right')

    metric_text = ''

    # TODO: this is imperfect as there may be other non-input columns in the data frame
    in_columns = list(df.columns)
    in_columns.remove('tbr')
    in_columns.remove('tbr_pred')

    for init_metric in get_metric_factory().values():
        metric = init_metric()

        if metric_text:
            metric_text += '\n'
        metric_value = metric.evaluate(
            df[in_columns], df['tbr'].to_numpy(), df['tbr_pred'].to_numpy())
        metric_text += f'${metric.latex_name}$ = {metric_value:0.06f}'

    anchored_text = AnchoredText(metric_text, loc=2)
    ax.add_artist(anchored_text)

    return fig, ax
