import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plotfiles = []

plotfiles.append({  'filename': '1000_50_v1_50kmcmc_metrics.csv', 
                    'label':    '50k MCMC run',
                    'color':    'r',
                    'scale':    50
                }) 
                
plotfiles.append({  'filename': '1000_50_v1_metrics.csv', 
                    'label':    '100k MCMC run',
                    'color':    'b',
                    'scale':    50
                }) 
                                
                
args = []
args.append({   'name':     'MAE',
                'style':    '-',
           })
args.append({   'name':     'S',
                'style':    '--',
           })
           

domain = 'numdata'

fig, ax = plt.subplots(1, 1)

for f in plotfiles:
    data = pd.read_csv(f['filename'])
    for arg in args:
        data.plot(x=domain, y=arg['name'], style = f['color'] + arg['style'], ax=ax, label=arg['name'] + ' -- ' + f['label'])

fig.set_size_inches(18,10)
fig.set_dpi(300)

plt.xlabel("Number of Samples")
plt.ylabel("Error")
plt.title("Varied MCMC runtime for 1000 Initial, 50 Incremental Samples")
        
plt.savefig("1000_mcmc_samp.png")
