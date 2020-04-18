import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plotfiles = []

plotfiles.append({  'filename': 'EVAL_10000_100_v1_metrics.csv', 
                    'label':    'MCMC samples',
                    'color':    'b',
                    'scale':    100
                }) 
                
plotfiles.append({  'filename': 'EVAL_10000_100_v1_renorm20_metrics.csv', 
                    'label':    'MCMC samples, renormalise every 20 iter',
                    'color':    'k',
                    'scale':    100
                }) 

plotfiles.append({  'filename': 'EVAL_10000_100_fake_metrics.csv', 
                    'label':    'uniform random samples',
                    'color':    'r',
                    'scale':    100
                }) 
                                
                
args = []
args.append({   'name':     'E_MAE',
                'style':    '-',
           })
args.append({   'name':     'MAE',
                'style':    '--',
           })

           

domain = 'numdata'

fig, ax = plt.subplots(1, 1)

for f in plotfiles:
    data = pd.read_csv(f['filename'])
    data = data.head(150)
    for arg in args:
        data.plot(x=domain, y=arg['name'], style = f['color'] + arg['style'], ax=ax, label=arg['name'] + ' -- ' + f['label'])

fig.set_size_inches(18,10)
fig.set_dpi(300)

plt.xlabel("Number of Samples")
plt.ylabel("Error value")
plt.title("Varied Sampling Scheme for 10000 Initial, 100 Incremental Samples")
        
plt.savefig("10000_sampling.png")

