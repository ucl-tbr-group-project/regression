'''
Performs principal component analysis (PCA) on continuous parameters,
given discrete-sliced sample set.
'''

from tbr_reg import data_utils
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint
from skgstat import Variogram


def test_variogram():
    '''
    Performs PCA on each discrete-slice from run.
    '''
    
    var_descs = {}
    
    indir = "../../Data/run1dsort/"
    for filename in os.listdir(indir):
        if filename.endswith(".csv"):
    
            # load dataset into Pandas DataFrame
            fpath = os.path.join(indir,filename)
            runname = os.path.splitext(filename)[0]
            
            df = pd.read_csv(fpath, index_col=0)
    
            # separate out cont and disc features, and target
            c, d, y = data_utils.c_d_y_split(df)

            # construct pca space consisting of continuous features and target
            cy = pd.concat([c, y[['tbr']]], axis = 1)

            numcomp = y[['tbr']].values.size
            yvalues = [x[0] for x in y[['tbr']].values]
            
            # Calculate variogram of TBR over all parameter values
            V = Variogram(coordinates=c.values, values=yvalues,
                          model='matern', normalize=False,
                          maxlag=100, n_lags=100)
                          
            fig = V.plot()              
                    
            fig.savefig('var_out/var_plot' + runname + '.jpg')
            #fig.show()
            
            var_descs[runname] = V.describe()
            
            
    with open("var_out/vardescs.txt", 'w') as f:
        pprint(var_descs, f)
        
def test_variogram_all():
    '''
    Performs PCA on all discrete-slices from run, together.
    '''
    
    
    indir = "../../Data/run1dsort/"
    for filename in os.listdir(indir):
        if filename.endswith(".csv"):
    
            # load dataset into Pandas DataFrame
            fpath = os.path.join(indir,filename)
            runname = os.path.splitext(filename)[0]
            
            if 'df' in locals():
                df = df.append(pd.read_csv(fpath, index_col=0), ignore_index=True)
            else:
                df = pd.read_csv(fpath, index_col=0)
    
    # separate out cont and disc features, and target
    c, d, y = data_utils.c_d_y_split(df)

    # construct pca space consisting of continuous features and target
    cy = pd.concat([c, y[['tbr']]], axis = 1)
    numcomp = y[['tbr']].values.size
    yvalues = [x[0] for x in y[['tbr']].values]
            
    # Calculate variogram of TBR over all parameter values
    V = Variogram(coordinates=c.values, values=yvalues,
                  model='matern', normalize=False,
                  maxlag=200, n_lags=100)
                  
    fig = V.plot()  
    pause            
            
    fig.savefig('var_out/var_plot_all.jpg')
    #fig.show()
            
    with open("var_out/vardescs_all.txt", 'w') as f:
        pprint(V.describe(), f)
        
def rank_variogram():
    

if __name__ == '__main__':
    os.environ['SKG_SUPPRESS'] = "true"
    test_variogram()
