
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib.dates import date2num, num2date
from matplotlib.colors import ListedColormap
from matplotlib import dates as mdates
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from matplotlib import ticker
import datetime

# Plot Tong's default setting
SMALL_SIZE  = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE, family='sans-serif', serif='Open Sans')#'Arial') # controls default text sizes

plt.rc('axes',   titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes',   labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels"
plt.rc('ytick',  labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text')

from matplotlib.ticker import MaxNLocator
my_locator = MaxNLocator(6)

'''
colorWheel =['#329932', '#ff6961', 'b', '#6a3d9a', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
            '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#67001f', '#b2182b', '#d6604d',
            '#f4a582', '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de', '#4393c3', '#2166ac', '#053061']
dashesStyles = [[3,1],
            [1000,1],
            [2,1,10,1],
            [4, 1, 1, 1, 1, 1]]
'''

NUMOFAGENTS = 46850
color_list  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
color_list2 = ['#070707', '#ff3b3b']                       # ['#81b8df', '#fe817d']
color_list3 = ['#4d85bd', '#f7913d', '#59a95a']            # ['#d22026', '#385889', '#7fa5b7']
color_list4 = ['#2d3063', '#6dab7a', '#d4357a', '#e3ab12']
color_list5 = ['#015699', '#fabf0f', '#f3774a', '#5fc5c9', '#15ab30']#['#015699', '#fabf0f', '#f3774a', '#5fc5c9', '#4f596d']

def figure_size_setting(WIDTH):
    #WIDTH = 700.0  # the number latex spits out
    FACTOR = 0.8  # the fraction of the width you'd like the figure to occupy
    fig_width_pt  = WIDTH * FACTOR
    inches_per_pt = 1.0 / 72.27
    golden_ratio  = (np.sqrt(5) - 1.0) / 2.0      # because it looks good
    fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio   # figure height in inches
    fig_dims    = [fig_width_in, fig_height_in]   # fig dims as a list
    return fig_dims

def format_axis(ax, month=True):
    ax.tick_params(which='both', axis='x', labelrotation=90)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    if month:
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plot_convergence_plots(theta, params_df, param_label, prior_range, num_params=2, num_iter_if=50, title=None, path_to_save = None):
    fig, ax = plt.subplots(num_params, 1, figsize=(15.5, 12.2), sharex=True)
    for idx, axi in enumerate(ax.flatten()):
        param_range = prior_range[idx]
        p_lab       = param_label[idx]
        param_df    = params_df[idx]

        axi.plot(range(num_iter_if+1), theta[idx,:], color="k", lw=3, label="Mean")
        axi.fill_between(range(1,num_iter_if+1), param_df["low_95"], param_df["high_95"], color="gray", alpha=0.2, label="95% CI")
        axi.fill_between(range(1,num_iter_if+1), param_df["low_50"], param_df["high_50"], color="gray", alpha=0.4, label="50% CI")
        #axi.axhline(y=param_truth, color="red", linestyle="--", lw=2, label="Truth")
        axi.set_ylabel(p_lab)
        axi.legend(loc="upper left", ncol=1)
        axi.set_ylim(param_range)

    ax[-1].set_xlabel("IF iteration")
    fig.suptitle(title)

    plt.tight_layout()
    if path_to_save:
        fig.savefig(path_to_save, dpi=300, transparent=True)


def create_df_response(samples, time, date_init ='2020-03-06',  quantiles = [25, 50, 80, 95], forecast_horizon=27, dates=None, use_future=False):
    """[summary]

    Args:
        samples ([type]): [description]
        time ([type]): [description]
        date_init (str, optional): [description]. Defaults to '2020-03-06'.
        forecast_horizon (int, optional): [description]. Defaults to 27.
        use_future (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    if dates is not None:
        dates_fitted = dates
    else:
        dates_fitted   = pd.date_range(start=pd.to_datetime(date_init), periods=time)
        dates_forecast = pd.date_range(start=dates_fitted[-1]+datetime.timedelta(1), periods=forecast_horizon)

    dates = list(dates_fitted)
    types = ['estimate']*len(dates_fitted)
    if use_future:
        dates += list(dates_forecast)
        types  += ['forecast']*len(dates_forecast)

    results_df = pd.DataFrame(samples.T)
    df_response = pd.DataFrame(index=dates)
    # Calculate key statistics
    df_response['mean']        = results_df.mean(axis=1).values
    df_response['median']      = results_df.median(axis=1).values
    df_response['std']         = results_df.std(axis=1).values

    for quant in quantiles:
        low_q  = ((100-quant)/2)/100
        high_q = 1-low_q

        df_response[f'low_{quant}']  = results_df.quantile(q=low_q, axis=1).values
        df_response[f'up_{quant}'] = results_df.quantile(q=high_q, axis=1).values

    df_response['type']        =  types
    df_response.index.name = 'date'

    return df_response
