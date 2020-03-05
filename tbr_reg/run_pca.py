'''
Performs principal component analysis (PCA) on continuous parameters,
given discrete-sliced sample set.
'''

from tbr_reg import data_utils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def test_pca():
    '''
    Performs PCA on all discrete-slices from run.
    '''
    pcasumvar = {}
    
    indir = "../../Data/run1dsort/"
    for filename in os.listdir(indir):
        if filename.endswith(".csv"):
    
            # load dataset into Pandas DataFrame
            fpath = os.path.join(indir,filename)
            df = pd.read_csv(fpath, index_col=0)
    
            # separate out cont and disc features, and target
            c, d, y = data_utils.c_d_y_split(df)

            # construct pca space consisting of continuous features and target
            cy = pd.concat([c, y[['tbr']]], axis = 1)

            numcomp = cy.columns.values.size
            cy = StandardScaler().fit_transform(cy)
 
            pca = PCA(n_components=numcomp)
            principalComponents = pca.fit_transform(cy)
            principalDf = pd.DataFrame(data = principalComponents, 
			           columns = ['pc1', 'pc2', 'pc3', 'pc4',
				              'pc5', 'pc6', 'pc7', 'pc8',
					      'pc9', 'pc10','pc11','pc12',
					      'pc13'])

            #fig = plt.figure(figsize = (8,8))
            #ax = fig.add_subplot(1,1,1) 
            #ax.set_xlabel('Principal Component 1', fontsize = 15)
            #ax.set_ylabel('Principal Component 2', fontsize = 15)
            #ax.set_title('2 component PCA', fontsize = 20)
            #ax.scatter(principalDf['pc1'],
             #           principalDf['pc2'],
            #           s = 50)
            #ax.grid()	                    
            #fig.savefig('pca_plot.jpg')
    
            pcavar = np.array(pca.explained_variance_ratio_)
            this_pcasumvar = np.round(pcavar.cumsum()*100,decimals=1)
            pcasumvar[os.path.splitext(filename)[0]] = this_pcasumvar
    
            with open("pca_out/result" + filename, "ab") as f:
                np.savetxt(f, this_pcasumvar, delimiter=',')
                f.write(b"\n")
                np.savetxt(f, pca.components_, delimiter=',')
            
    with open("pca_out/pcasumvar.csv", 'w') as f:
        pprint.pprint(pcasumvar, f)

if __name__ == '__main__':
    test_pca()
