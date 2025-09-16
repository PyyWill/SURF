import sys
import os
sys.path.append('.')
from portfolio import synthetic, yfinance
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
pd.options.mode.copy_on_write = True
pio.renderers.default = 'svg'

if __name__ == '__main__':

    # generate scatter grid for synthetic data
    X, _, _, y, _, _ = synthetic.data_gen(0.1, 0.2)
    df = pd.DataFrame(np.concatenate((X.numpy(), y.numpy()), axis=1), columns=['X_0', 'X_1', 'y_1', 'y_2'])
    px.scatter_matrix(df).write_image('plots/synthetic_scatter_matrix.png') # change extension to svg for the paper
    
    # generate vol vs return plots for yfinance data
    brets = pd.read_csv('portfolio/data/yfinance/expected_return.csv', dtype_backend='pyarrow')
    sides = pd.read_csv('portfolio/data/yfinance/side_info.csv', dtype_backend='pyarrow')
    px.scatter(x=sides['AAPL_SI'], y=brets['AAPL'], labels={'x': 'Volume', 'y': 'Return'}).write_image('plots/yfinance_AAPL_vol_bret_scatter.png')
    px.density_contour(x=sides['AAPL_SI'], y=brets['AAPL'], marginal_x="histogram", marginal_y="histogram", labels={'x': 'Volume', 'y': 'Return'}).write_image('plots/yfinance_AAPL_vol_bret_density.png')

    # generate index vs return plot
    px.scatter(x=sides['DJI_SI'], y=brets['AAPL'], labels={'x': 'DJI Index', 'y': 'Return'}).write_image('plots/yfinance_DJI_AAPL_scatter.png')
    px.scatter(x=sides['VIX_SI'], y=brets['AAPL'], labels={'x': 'VIX Index', 'y': 'Return'}).write_image('plots/yfinance_VIX_AAPL_scatter.png')
    px.density_contour(x=sides['DJI_SI'], y=brets['AAPL'], marginal_x="histogram", marginal_y="histogram", labels={'x': 'DJI Index', 'y': 'Return'}).write_image('plots/yfinance_DJI_AAPL_density.png')
    px.density_contour(x=sides['VIX_SI'], y=brets['AAPL'], marginal_x="histogram", marginal_y="histogram", labels={'x': 'VIX Index', 'y': 'Return'}).write_image('plots/yfinance_VIX_AAPL_density.png')
